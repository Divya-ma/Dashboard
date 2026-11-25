# QH API guide (for another Claude instance)

This explains how to call the **QH API** (`https://qh-api.corp.hertshtengroup.com/apis`) the way this project does. The
dashboard itself exposes **no API of its own**: it is a Dash web app. "The API" here is the external market-data service
the dashboard reads from. Swagger: `https://qh-api.corp.hertshtengroup.com/apis/swagger/`.

Sources of truth in the code (read these before changing anything):
`adapters/qh_api/api.py` (one method per endpoint, parameter checks), `adapters/qh_api/client.py` (HTTP, retries,
pagination), `adapters/qh_api/rate_limiter.py` (limits), `adapters/base.py` (symbol translation).
Everything below marked **(probed)** was found by live calls, not from the Swagger text; **(unverified)** was not confirmed.

---

## 1. The one rule that matters: never call it raw

Always go through the shared `QHApi` object (`adapters.qh_api.make_api(token_provider)`; the app builds ONE and shares it).
Never use `requests` directly and never build a second `QHApi` in a running app. Reason: rate limits are **per token per
endpoint**, so a second copy of the dashboard (or a second client) with the same token shares and can exhaust the budget,
for instance the live price poll's `/fairvalue/` budget. The user's own dashboard normally runs on port 8050; do not start a
second real instance while it is running. In tests use fakes (see `tests/test_adapters_live_fairvalue.py`).

```python
from adapters.qh_api import make_api
api = make_api(lambda: "<bearer token>")      # token comes from the app's Settings tab / the container
rows = api.ohlc("5M", instruments=["CLZ26"], start=1_760_000_000, end=1_760_086_400)
```

## 2. Authentication and transport (`QHClient`)

- Header `Authorization: Bearer <token>` on every request, `accept: application/json`. The token is read per request from
  `token_provider`, so a token changed in Settings applies at once. No token -> `AuthenticationError`.
- Public endpoints (no token, no limit): `/common/instruments/`, `/fundamental/codes/`, `/taq/products/` (the last one is
  documented public but **needs a token in practice (probed)**).
- One in-flight request per endpoint (a lock per path). Never call an endpoint from parallel threads.
- Retries: timeouts and connection errors retry after 1 s, 2 s, 4 s...; 5xx retry the same way; 429 waits `Retry-After` and
  retries (at most 3 times); at most 5 attempts per request. 401 raises at once. Per-request timeout 60 s.
- Errors (`adapters.base`): `AuthenticationError` (401 / no token), `RateLimitError` (429 persisted, or the local budget is
  empty for longer than 90 s), `APIServerError` (5xx), `APITimeoutError`, `APIConnectionError`, `APIError` (anything else,
  message carries the first 300 chars of the body; a missing instrument is an `APIError` with "not found" in it).
- Params with value `None` are dropped; lists are comma-joined; bools become `true`/`false`; dates become ISO strings.

## 3. Rate limits (per token, per endpoint path; GET and POST on one path share a budget)

| Endpoint group | per minute | per hour | per day |
|---|---|---|---|
| default (everything not listed) | 7 | 420 | 10,080 |
| `/ohlc/` | 30 | 1,800 | 43,200 |
| `/tas/` | 10 | 600 | 14,400 |
| `/common/instruments/`, `/fundamental/codes/`, `/taq/products/` | unlimited | | |

A 429 still counts against the limit, so the client records every request when sent and blocks (up to 90 s) until the minute,
hour and day windows all have room; server `X-RateLimit-Remaining/-Reset` headers override the documented numbers. When the
hour or day is spent it raises `RateLimitError` instead of freezing a Dash callback. `api.spare_requests("ohlc")` returns how
many calls can go out right now; optional background work should check it first. **The live price poll uses ~6 of
`/fairvalue/`'s 7 per minute**, so any other `/fairvalue/` caller queues behind it. Each paginated page is one request, so
`max_pages` caps are deliberate.

## 4. Symbols (very important, easy to get wrong)

Internal product codes differ from the API's (`adapters/base.py: PRODUCT_TO_API_CODE`):

| Internal | API code | Meaning |
|---|---|---|
| `CL` | `CL` | WTI crude (NYMEX) |
| `BRN` | `CO` | ICE Brent |
| `BZ` | `BZZ` | NYMEX Brent look-alike |
| `WBS` | `WTCL` | a WTI-linked product (cost table entry only; no price feed in the app) |
| `G` | `GO` | ICE gasoil |
| `CLBZ` | `WTCLCO` | CL - BZ "outbox": `WTCLCOF27` = CL Jan27 - BZ Jan27 (**unverified**: the code was never confirmed against the live API) |
| `CLBZS` | `CLBZZX` | CL - BZ spreadbox: `CLBZZXF27-G27` (**unverified**) |

`SymbolTranslator.internal_to_api("BRNZ26") == "COZ26"`. Month letters: F Jan, G Feb, H Mar, J Apr, K May, M Jun, N Jul,
Q Aug, U Sep, V Oct, X Nov, Z Dec, then a 2-digit year (`CLZ26` = CL December 2026).

Instruments you can ask for:
- Outright: `CLZ26`, `COZ26`.
- Spread: `COZ26-F27` (December 2026 minus January 2027). Fly: `COZ26-F27-G27`. **Month-code form only; `/fairvalue/` does not accept
  generic codes for spreads/flies (probed).**
- Generic positions (front of the curve): `CO1`, `CO1-2`, `CO1-2-3` (probed), accepted by `/generic-charts/generic/data/` only.
- A d-fly is not a quotable instrument: this app builds it as fly minus the next fly.
- `/fairvalue/` omits some far-month flies (e.g. `COV28-X28-Z28`), so the app derives them from spreads/outrights
  (`core/derived_prices.py`).

## 5. Endpoints the dashboard actually uses

### 5.1 `GET /ohlc/` — candles (the workhorse; 30/min)
`api.ohlc(interval, instruments=None, hg_instrument_ids=None, count=None, start=None, end=None, extra_fields=None)`
- `interval`: `1M`, `5M`, `1H`, `1D` (anything else raises `ValueError` before any request).
- Give **either** `instruments` (up to 50 QH codes) **or** `hg_instrument_ids`, never both or neither.
- `start`/`end`: **unix seconds**. `count` defaults to 50 per instrument. `extra_fields`: `buyvolume`, `sellvolume`.
- Reply: a list of `{product, time (ms), open, high, low, close, volume}` rows, or `{"status": "SUCCESS", "data": []}` when
  nothing traded in the range. Handle both (`core/atr_data.py::_fetch`, `core/journal_snapshot.py::candles_by_symbol`).
- Used for: daily candles for ATR (`core/atr_data.py`), 5-minute candles for Churn (`core/churn_data.py`), journal excursions.

### 5.2 `GET /fairvalue/` — live prices (7/min; polled every 10 s)
`api.fairvalue(products="*" | [codes], token=None, max_attempts=None)`; `POST /fairvalue/` takes `{"products": [...]}`.
- Rows are `{Contract, Price, Timestamp (ms)}` **(probed)**. Back-month prices can carry a batch timestamp hours old, so check
  the age before trusting one (the Churn page treats >2 minutes as stale).
- Accepts outrights, spreads and flies in month-code form. The poll (`adapters/live/fairvalue.py`) requests only legs of open
  structures, shell structures, the watchlist and active price alerts.

### 5.3 `GET /common/dailymarketdata/` — settlements, open interest, volume (default limit)
`api.daily_market_data(qhcode="CLZ26,COZ26", start="2026-01-01", end="2026-11-01", fields="oi,close,volume", limit=20000, offset=0)`
- `close` is the **settlement** (probed). `qhcode` takes a comma list. `gdcode` filters by generic code (e.g. `CO1`).
- Pages by `limit`/`offset`; the reply is a dict with `results` and `count`. **Page by `count`, not by `len(results)`**; a page
  limit of at least 20,000 works (probed). Latest row is the previous trading day (OI is end-of-day).
- Spread/fly codes return 404 "Instrument not found" (probed), so spread/fly settlements are derived from outrights.
- `oi` can be `None` (Brent's last day) or 0 (before trading starts). A contract's history starts ~1,850 calendar days before expiry.
- 260 contracts (CL + Brent, 2019 to +36 months) is ~270k rows = 22 requests, ~3 minutes. Used by `core/oi_data.py`.
- Without `fields`, rows include `gd_code` (e.g. `CO1`).
- `GET /common/dailymarketdata-summary/` (`product`, `start`, `end`, `count`, `limit`, `offset`) gives `total_oi` and
  `total_volume` per product per day.

### 5.4 `POST /generic-charts/generic/data/` — generic (rolling front) series
`api.generic_chart_data(groups, end_inclusive=1)` where each group is
`{product: "CO1", interval: "1D", resolve: 0/1, months: "F,G,H", gap: 0/1, end: <unix, intraday only>, count: 500}`.
- **`end` must be in milliseconds** (seconds returns "No data (query)") (probed).
- Reply is a dict keyed `"{product}_{interval}"` (e.g. `CO1_1D`) holding `df` rows `{Date (ms), Open, High, Low, Close, Volume, Product}`;
  `Product` is the real contract the generic position held that day. The last row is today's unfinished bar.
- Generic 1D `Close` differs slightly from the `dailymarketdata` settlement. Do not mix them in one series. Used by `core/curve_history.py`.

### 5.5 `POST /taq/` — trades and quotes (7/min)
`api.taq([{"id": "COZ26-F27", "dates": ["2026-11-12"], "start": "08:00:00", "end": "09:00:00"}], page=None)`
- Times and the row `timestamp` (microseconds) are **UTC**. Reply `{status, page, total_pages, page_size: 10000, data: [...]}` with rows
  `{timestamp, instrument, best_bid, best_ask, bid_size, ask_size, trade_price, trade_volume, trade_side, ...}`.
- Rows are sorted **oldest first across all products**, 10,000 per page, so page 1 holds the *oldest* rows and the newest are on the
  **last** page (`?page=N`, reply has `page`, `total_pages`, `next`). Reading only page 1 returns stale quotes. Keep windows short.
- **One row per trade, not per quote change (probed)**: the day's rows sum exactly to the 5-minute candles' volume and fall in the
  same buckets. TAQ adds the bid-ask width (flies ~1 tick, max 2) but no extra price path over `/ohlc/`.
- Today's data can start late (nothing before ~09:20 UTC on one probed day). It is historical by date, not a live feed.
- Spread/fly quotes are tight (median gap 0.01); back-month **outright** quotes are often junk-wide (median 0.17, up to 6.4), so
  never derive spreads or flies from wide outright quotes.
- `GET /taq/products/` lists products (public by docs, token needed in practice).

## 6. Other wrapped endpoints (ready, not currently used by the dashboard)

All return the parsed JSON untouched; unknown parameter names raise `ValueError` listing the allowed ones.

| Method | Path | Notes |
|---|---|---|
| `economies()` / `economy_premiums(economies="*")` | `/common/economies/`, `/common/economies/premiums/` | `economies` required for premiums |
| `fixings(...)` | `/common/fixings/` | `name, economy, start, end, page, page_size` |
| `instruments(...)` | `/common/instruments/` | public: `qh_codes, tt_instrument_ids, limit, offset` |
| `curves(...)` | `/curves/` | `products, strategies, range, start, end` |
| `fairvalue_historical(...)` | `/fairvalue/historical/` | up to 50 instruments; `timeinterval` 1m..1d; `fields` open/high/low; `start`/`end` unix seconds |
| `fundamental_codes(...)` | `/fundamental/codes/` | public; `page_size` max 5000 |
| `fundamental_data(qhcodes,...)` | `/fundamental/data/` | up to 10 codes, **semicolon-separated** on the wire; `page_size` max 1000 |
| `gtc(...)` | `/gtc/` | `products` (list or `*`), `date` |
| `vap(instruments, interval, ...)` | `/vap/` | volume-at-price, interval `5M`/`1D`, up to 50 |
| `options_generic_code`, `options_greeks`, `options_instruments`, `options_oi` | `/options/...` | `options_oi` takes max 25 contracts |
| `platts_*` (bot, commentary, ladder, ltp, market-data, market-product, moc snapshot v1/v2, nai, refinery-outage, summary) | `/platts/...` | filters like `date`, `date__gte`, `market`, `product`, `time__range`; v2 variants are ClickHouse |
| `seasonality(...)` | `/seasonality/` | `qhcodes, multipliers, field (close/oi/volume), years, datatype` |
| `tas(products)` | `POST /tas/` | time and sales, same body shape as TAQ, 10/min |
| `get_all(name, page_size=1000, max_pages=10, **params)` | any paginated endpoint | one request per page, capped |

`adapters.qh_api.list_endpoints()` returns every endpoint with its method, path, limit and pagination style.

## 7. Pagination styles
- `offset`: `limit` + `offset`; reply is usually `{results, count, next}`. Stop when a page is short or `next` is empty.
- `page`: `page` (1-based) + `page_size`.
- `/taq/`: `?page=N` on the POST URL (see 5.5).
Always cap pages; each page spends budget.

## 8. Practical recipes

- **Daily history of a contract:** `/common/dailymarketdata/` with `fields="oi,close,volume"` (settlement-accurate, gap-free), not `/ohlc/` 1D (sparse).
- **Intraday path:** `/ohlc/` 5M, batch up to 50 symbols per call, start/end in unix seconds, request end = day + 1.
- **Spread/fly history:** derive from outrights' settlements, or `/ohlc/` with the spread code for intraday.
- **Live price of a structure:** one `/fairvalue/` call for all legs; derive composites that are missing (`core/derived_prices.py`).
- **Brent proxy:** the settlement store treats BZ as proxied by BRN; the stored CL-BZ outbox contract (`WTCLCO<month>`) stops trading
  at the Brent expiry of that month (CL expires ~20th, Brent ~month end).
- **Expiry handling:** a front contract rolls; the first trading day after a CL or Brent expiry is a contract switch, not a price move.

## 9. Pitfalls checklist
1. Seconds vs milliseconds: `/ohlc/` takes unix **seconds**; `/generic-charts/.../data/` `end` takes **milliseconds**.
2. Internal `BRN` is API `CO`; `BZ` is `BZZ`. Translate with `SymbolTranslator`.
3. Spreads/flies exist only in month-code form for `/fairvalue/` and `/ohlc/`; settlements for them must be derived.
4. Page `/dailymarketdata/` by the reply's `count`; page `/taq/` to the **last** page for fresh quotes.
5. `/ohlc/` can return `{"status": "SUCCESS", "data": []}` (a dict) instead of a list.
6. Do not run a second app instance or a second `QHApi` with the same token.
7. Do not spend the hourly/daily budget on speculative downloads; check `spare_requests` for optional background work.
8. Never print or commit the Bearer token; it lives in the app's Settings, not in the repo.
9. Do not import `ui.app` in checks: it runs the real start-up on the live database. Test with fakes and module imports.
