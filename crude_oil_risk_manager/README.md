# Crude Oil Risk Manager

A single-user trading and risk dashboard for a crude oil / gasoil futures spread trader. It tracks open structures
(outrights, spreads, flies, d-flies and custom combinations) of **WTI (CL)**, **ICE Brent (BRN)** and **ICE Low Sulphur
Gasoil (G)**, prices them live, and builds on that: P&L, exposure, correlation, Value-at-Risk, curve-kink detection,
curve comparison, open-interest and ATR research, an intraday "churn" screen, and a trade journal with analysis.

It is a Dash (Plotly) web application backed by SQLite and Parquet files, fed by the QH market-data API. It runs
locally (`python ui/app.py`), in Docker, or as a Windows service.

This README describes everything the dashboard does today and how each number is calculated. Where a number depends on
a setting, the default is given.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Architecture](#2-architecture)
3. [Products, symbols and conventions](#3-products-symbols-and-conventions)
4. [Market data: sources, rate limits, background jobs](#4-market-data-sources-rate-limits-background-jobs)
5. [Core calculations](#5-core-calculations)
6. [The pages](#6-the-pages)
   - [Sidebar and shell](#61-sidebar-and-shell)
   - [Home](#62-home)
   - [Structures](#63-structures)
   - [Correlation](#64-correlation)
   - [Exposure Map](#65-exposure-map)
   - [VaR & Scenarios](#66-var--scenarios)
   - [Trade Analyzer (Curve Kinks)](#67-trade-analyzer-curve-kinks)
   - [Curve Compare](#68-curve-compare)
   - [OI · Volume · ATR](#69-oi--volume--atr)
   - [Churn](#610-churn)
   - [Archive and the trade journal](#611-archive-and-the-trade-journal)
   - [Settings](#612-settings)
   - [Seasonality](#613-seasonality)
7. [Alerts](#7-alerts)
8. [Storage: database, files, settings keys](#8-storage-database-files-settings-keys)
9. [Configuration (.env)](#9-configuration-env)
10. [Deployment](#10-deployment)
11. [Development](#11-development)
12. [Assumptions and known limits](#12-assumptions-and-known-limits)

---

## 1. Quick start

```bash
cp .env.example .env            # then fill it in (see section 9)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python ui/app.py                # http://localhost:8050
```

1. Open **Settings → API Configuration**, paste the QH API Bearer token, and press *Save Token*. Until a token is saved
   the app shows a "setup required" banner and fetches nothing.
2. On first start the app downloads history in the background (settlements and open interest from 2019, ATR candles,
   generic curve history, five-minute candles). Pages show "downloading" messages and fill in as data arrives.
3. Create structures with **Structures → + New Structure**, enter trades on them, and watch Home.

Python 3.11+ is used in Docker; 3.14 is used on the development machine.

---

## 2. Architecture

```
crude_oil_risk_manager/
├── ui/                  Dash app: app.py (startup), layouts/ (static page structure), callbacks/ (wiring), container.py
├── core/                All business logic and maths (no Dash, no SQL)
│   └── models/          Pydantic domain models: Contract, Leg, Structure, Trade, PnLRecord, Alert
├── adapters/
│   ├── qh_api/          Rate-limited client + one wrapper per API endpoint
│   ├── live/            /fairvalue/ live price adapter (10 s poll)
│   ├── historical/      Daily-candle adapter with a local Parquet cache
│   └── mock/            Offline adapters for development
├── db/                  schema.sql and repository.py (the only SQL in the app)
├── config/settings.py   Environment configuration (pydantic-settings)
├── data/                Parquet / JSON stores (created at runtime)
├── service/             NSSM scripts for the Windows service
├── docs/                Worked notes (neighbour interpolation)
└── tests/               ~1,300 tests (pytest)
```

Design rules that hold everywhere:

- **Layouts are static, callbacks fill them.** `ui/layouts/*` build component trees; `ui/callbacks/*` load data, call
  `core/`, and render the result. Calculations are never done in a layout.
- **`core/` is pure where possible.** Maths functions take data in and return numbers out. I/O lives in `*_data.py` /
  `*_store.py` / `*_service.py` modules and in `db/repository.py`.
- **One source of truth per rule.** Position sign rules live in `core/positions.py`; contract sizes in
  `core/contract_specs.py`; the curve calendar in `core/curve_calendar.py`; the hedge ratio in `core/hedge.py`; the
  correlation rule in `core/correlation.py`; P&L in `core/pnl.py`. Every page that needs them imports them.
- **`ui/container.py`** is the dependency container filled once in `ui/app.py` (repository, adapters, services). Callback
  modules import `container`; nothing imports `ui.app`.
- **Dash lazy-loads page layouts**, so every Dash component library a page uses is imported at startup in `ui/app.py`
  (Dash refuses a library first imported inside a callback).

### Data flow in one picture

```
QH API ──(rate-limited client)──┬─ /fairvalue/ 10 s ─► live prices ─► portfolio P&L, packets, curve engine, stop/target alerts
                                ├─ /common/dailymarketdata/ ─► settlements + OI store ─► correlation, VaR, curve settlements
                                ├─ /ohlc/ 1D ─► ATR store;  /ohlc/ 5M ─► churn store;  /ohlc/ 1M-1H-1D ─► journal snapshots
                                ├─ /generic-charts/ ─► curve history (Trade Analyzer)
                                └─ /taq/ ─► quotes (Curve Compare, journal bid-ask)
SQLite ◄─► repository ◄─ structures, legs, trades, P&L snapshots, alerts, settings, kink events, trade journal
```

---

## 3. Products, symbols and conventions

### Products

| Product | Internal code | API code | Contract | Tick | Tick value | $ per 1.0 point | Expiry rule |
|---|---|---|---|---|---|---|---|
| WTI crude | `CL` | `CL` | 1,000 bbl | 0.01 | $10 | $1,000 | 3 business days before the 25th of the month before delivery (NYMEX calendar) |
| ICE Brent | `BRN` | `CO` | 1,000 bbl | 0.01 | $10 | $1,000 | Last business day of the second month before delivery (UK calendar) |
| ICE Low Sulphur Gasoil | `G` | `GO` | 100 tonnes | 0.25 | $25 | $100 | 2 business days before the 14th of the **delivery** month (UK calendar) |
| NYMEX Brent | `BZ` | `BZZ` | 1,000 bbl | 0.01 | $10 | $1,000 | (positions/history only) |
| WBS | `WBS` | `WTCL` | 1,000 bbl (assumed) | 0.01 | $10 | $1,000 | (positions / candle history only; not on the curve pages) |

`core/contract_specs.py` holds these numbers; `core/curve_calendar.PRODUCTS` registers the three curve products
(CL, BRN, G) that the curve-based pages (Trade Analyzer, Curve Compare, OI, ATR, Churn, Correlation matrix, journal)
loop over. **Adding a product is one entry in `PRODUCTS` (plus its expiry rule) and one in `contract_specs`.**

Gasoil is priced in USD per tonne (about 1,400), so a "point" is $1 per tonne and the 0.25 tick is a quarter dollar.
Price-point limits in the code (for example "widest usable bid-ask") are written in CL ticks and scaled by the product's
tick size, so gasoil gets the same limit in ticks.

### Contract symbols

- Internal symbols: product + month letter + 2-digit year: `CLZ26`, `BRNF27`, `GZ26`. A month letter is F G H J K M N Q U V X Z
  (Jan … Dec).
- API symbols swap the product code: `BRNZ26 → COZ26`, `GZ26 → GOZ26`, `BZZ26` stays. `SymbolTranslator`
  (`adapters/base.py`) is the single translator.
- Spread: `CLZ26-F27` = Dec26 − Jan27. Fly: `CLZ26-F27-G27`. Condor: `CLZ26-F27-G27+H27`.
- A **d-fly** has no exchange symbol. It is `fly(k) − fly(k+1) = M1 − 3·M2 + 3·M3 − M4` and is built from two flies (or
  from its four outrights).

### Structure shapes (everywhere in the app)

| Family | Legs | Weights on consecutive months |
|---|---|---|
| outright | 1 | +1 |
| spread | 2 | +1 −1 |
| fly | 3 | +1 −2 +1 |
| d-fly | 4 | +1 −3 +3 −1 |

### Sign convention

A traded leg contributes **ratio × side × lots**, side = +1 for a buy and −1 for a sell. The ratio carries the leg's sign
inside its structure (a fly body is −2); the position's `direction` (buy/sell) flips a position entered as a sell.
Only legs with lots and an entry price count, and only structures that are `OPEN` or `PARTIALLY_CLOSED`.

### Statuses

`SHELL` (defined, no trade yet) → `OPEN` (traded) → `CLOSED`. `PARTIALLY_CLOSED` is legacy: it appears when a per-leg
exit leaves legs at unequal lots, and is treated as open.

### Price differences, not returns

All statistics (correlation, hedge ratios, VaR, ATR) use **daily price differences**, never percentage returns: spreads
and flies can be zero or negative, so returns are undefined for them.

---

## 4. Market data: sources, rate limits, background jobs

### The QH API client (`adapters/qh_api`)

One shared client wraps every endpoint (`api.py`) over a transport (`client.py`) that:

- sends the Bearer token on every request (read from the settings table, so changing it takes effect at once);
- enforces **per-endpoint minute / hour / day budgets** with a sliding-window limiter (`rate_limiter.py`) — every request is
  recorded when sent and blocks until all three windows have room; server `X-RateLimit-*` headers override the
  assumptions;
- serialises calls to one endpoint with a lock (the API forbids parallel calls to one endpoint);
- retries 429 after `Retry-After`, and 5xx / timeouts with 1 s, 2 s, 4 s … waits;
- raises `RateLimitError` instead of freezing a callback when a budget is exhausted for longer than a few seconds;
- offers `api.spare_requests(endpoint)` (the least room left in the minute / hour / day windows) so optional background
  work can run only when the budget is free.

| Endpoint | Used for | Budget (per minute / hour / day) |
|---|---|---|
| `/fairvalue/` | live prices, curve ladders (10 s poll) | 7 / 420 / 10,080 |
| `/ohlc/` | daily candles (ATR), 5-minute candles (Churn), 1M/1H/1D candles (journal snapshots, excursion) | 30 / 1,800 / 43,200 |
| `/taq/` | quotes for Curve Compare and the journal's bid-ask | 7 / 420 / 10,080 |
| `/common/dailymarketdata/` | settlement closes, open interest, volume | 7 / 420 / 10,080 |
| `/generic-charts/generic/data/` | generic-position history for the Trade Analyzer | 7 / 420 / 10,080 |
| `/tas/` | trade and sales (wrapped, not used by a page) | 10 / 600 / 14,400 |

GET and POST on the same path share one budget. Public endpoints (`/common/instruments/`, `/taq/products/`) are
unlimited. Because every consumer shares the client, the minute budget is also shared **across running copies of the
app** using the same token.

### Price sources by purpose

| Need | Source | Why |
|---|---|---|
| Live price of a position | `/fairvalue/` every 10 s; far-month composites derived from simpler contracts | one cheap call prices the whole book and the curve ladder |
| Long, gap-free price history (correlation, VaR, beta, builder, hedge) | **Settlements** from `/common/dailymarketdata/` (`SettlementFirstLoader`) | candles exist only on days a contract traded, so back months have few rows; settlements have none missing |
| History of symbols with no settlements (WBS, BZ, CL/BRN/G contracts the store lacks) | daily candle Parquet cache (`DataLoader`) | fallback |
| Last-traded outright prices (journal) | `/ohlc/` candles | `/fairvalue/` is unreliable for outrights |
| Curve Compare live price | TAQ best bid/ask VWAP, `/fairvalue/` fallback | see 6.8 |

`SettlementFirstLoader` reads a CL / BRN / G symbol from the settlements — an outright is its settlement; a spread, fly or
condor is built from its outrights' settlements on the dates every outright settled — and everything else from the
candles. A symbol is never half settlement and half candle. **BZ is not proxied** by Brent in the shared loader (its basis
to Brent is real risk); only the New Structure builder's beta panel may proxy it, and it says so.

### Background jobs

| Job | When | What |
|---|---|---|
| Live poll | every 10 s | `/fairvalue/` for open-structure legs, shell-structure legs and the whole curve ladders; derives missing composites |
| Portfolio P&L refresh | every 5 s | recomputes P&L from cached prices (no API call) |
| Settlement / OI update | daily (and at start if stale; also run from the live poll) | `/common/dailymarketdata/` for every CL, BRN and G outright from 2019; ATR candles |
| Curve history refresh | daily; at once if a product has none | `/generic-charts/` for 15 series per product |
| Previous settlements | daily at the opening time set in Settings | previous trading day's settlements for the Curve pages |
| Morning sync | at start / on demand | latest daily candle for tracked symbols; **skips symbols read from settlements** |
| Churn candles | while the Churn page is open | 5-minute candles of the live ladder, refreshed every 10 minutes |
| Quotes (TAQ, fair value) | only while Curve Compare is open | demand lapses shortly after the page closes |
| Journal snapshots | after a trade is entered / exited | background thread; never blocks the trade |

Downloads are chunked so an interrupted run resumes where it stopped; a failed run waits (5–10 minutes) before retrying.

---

## 5. Core calculations

### 5.1 P&L (`core/pnl.py`)

- **Leg P&L** = `(price − entry) × direction × ratio × lots × multiplier`, `direction` +1 for a buy, −1 for a sell.
- **Average entry** after an add = `(lots₀·p₀ + lots₁·p₁) / (lots₀ + lots₁)`: the entry price of the whole open position.
- **First in, first out (`core/fifo.py`).** An exit takes its lots from the **earliest entry lots still open**, each at its own entry
  price: the realized P&L of an exit is the sum over those lots of `(exit − that entry's price) × lots × multiplier`. The lots left
  open then carry the **entry price of the lots not yet taken** (the legs' entry price is moved to it), so unrealized P&L, the
  next exit and the entry price shown on the Structures page all follow. Example: 3 lots bought at 0.50 and 2 at 0.60; exiting 4 at
  0.80 realizes 3×0.30 + 1×0.20 = 1,100 (not 4 × (0.80 − 0.54) = 1,040) and the 1 lot left is held at 0.60. With one entry price
  FIFO and an average are the same number, and realized + unrealized is the same either way; FIFO only changes how much of it is
  already realized. If the legs do not match the entry trades (real per-leg fills typed at entry, a hand-corrected entry price),
  or a per-leg exit was used, the engine keeps the older average-cost figure for that structure.
- **Structure price (as quoted)** = `sign(first ratio) × Σ ratio × leg price`. This reproduces the quoted spread/fly from
  its legs. A structure is entered at one structure price and one lot count; leg entry prices are allocated so the per-leg
  engine reproduces the structure P&L exactly (every leg but the first anchored at its live price, the first absorbing the
  difference) — unless you type real per-leg fills, which are then used as they are.
- **Unrealized** = Σ leg P&L over traded legs with a live price (a leg with no price is excluded and listed as missing —
  never treated as zero).
- **Realized** = Σ `realized_pnl` of the structure's exit trades (stored when the exit was made, never recomputed).
- **Total (per structure)** = unrealized + realized − transaction costs booked so far.

### 5.2 Transaction costs, split by lots

Every trade stores its own cost: entry or add = Σ over legs of `cost per lot × lots`; an exit is charged on the exited
lots; a dissolve is charged **nothing** (see 6.3).

Costs are attributed **by lots, first in, first out**, so a partial exit or dissolve only takes the entry cost of the lots it takes:

- *entry cost per lot* of an entry = that entry's cost ÷ its lots;
- *entry cost of N lots* = the cost of the N earliest entry lots still open, each at its own entry's cost per lot (a history that
  cannot be matched lot by lot uses the average per entered lot);
- **realized cost** = all exit costs + the entry cost of the lots already exited; **open cost** = the entry cost of the lots
  still open. They always add up to the structure's whole cost.

Used in: the exit dialog and toast (net P&L of the exited lots), Home's realized P&L, and dissolve.

### 5.3 Portfolio P&L (`calculate_portfolio_pnl`)

- `total_unrealized`: gross mark-to-market of open positions.
- `total_realized_all_time` = open structures' realized + closed structures' net realized − the **realized cost** of open
  structures. (Open lots' entry cost is *not* in it; it stays with the position.)
- `total_unrealized_net_of_tc` = unrealized − the **open cost**.
- `total_pnl` = unrealized + realized − every cost.
- Net lots by product, largest winner / loser, stale and missing-price flags.
- **Account reset** (Settings): subtracts a stored baseline from `total_pnl` and from `total_realized_all_time` so Home shows
  zero from the reset. Stored structures and trades are untouched.

### 5.4 Positions and dollar weights (`core/positions.py`)

Dollar weight of a symbol = Σ `ratio × side × lots × multiplier` = P&L per 1.0-point move. Two views: **as quoted** (a fly
stays a fly; VaR uses this) and **outrights** (each leg split into the outright contracts it is made of; the Exposure Map
and beta/correlation against the book use this). Candidate (unsaved) structures use the same rule with the product's own
multiplier when none is typed.

### 5.5 Correlation (`core/correlation.py`)

Everywhere a single correlation number is shown:

1. **Rolling window R** (default 5 days, Settings): for each day, the Pearson correlation of the two series' price
   differences over the R days ending that day.
2. **Lookback L** (default 20): the plain average of those daily values over the last L days.

A day whose window has no variance has no value and is skipped. A pair needs at least **20 shared days**
(`MIN_COMMON_DAYS`). Pairs are aligned on their **own** common dates, so a heatmap cell does not change when another item is
added. A structure's price is built first from all its legs (on the dates every leg has a close) and *then* differenced, so
a leg with a missing day cannot mix a one-day change with a two-day one.

A trade structure's price series for correlation is built from its **outright legs' settlements**; BZ legs use the
settlement of the ICE Brent contract of the same month and the result says so.

### 5.6 Hedge ratio (`core/hedge.py`)

For daily changes *y* of a position and *x* of the hedge instrument: **β = cov(y, x) / var(x)** (sample statistics). To hedge
one unit of *y* trade β units of *x* the opposite way; the share of variance removed is the squared correlation. This one
definition is used by the ATR pair hedge, the correlation page's rolling hedge ratio, and the Trade Analyzer hedge
(`core/regression.py` adds standard error and p-value for panels that show them).

**Across products the ratio is converted to lots**: β is in points of *y* per point of *x*, so lots of *x* per lot of *y* =
β × (multiplier of *y* ÷ multiplier of *x*). A gasoil lot is a tenth the size of a Brent lot.

The beta window used by the builder and VaR panels is `max(30, R + L − 1)` days.

### 5.7 Value at Risk (`core/var.py`)

- **Historical simulation**: the book's P&L for each of the last N days = Σ (price change × dollar weight); VaR is the
  loss at the (1 − confidence) percentile — always a positive number. Only dates on which every symbol has a price count.
  Fewer than 10 observations still produce a number with an "insufficient data" warning.
- **Monte Carlo (correlated)**: `move_i = σ_i × (correlated standard draw) + mean_i`; P&L = Σ weight_i × move_i.
  - The correlation matrix is the app-wide one (5.5), repaired to a valid (positive semi-definite) correlation matrix.
  - σ_i is, by default, each symbol's **own standard deviation of daily changes over the last N days** (the same N and
    prices the historical method uses), so a fly is shaken less than an outright and the two methods agree on near-normal
    data. The alternative *fixed* mode uses one number of points for every symbol (default 0.5), which is blunt.
  - Normal draws have standard deviation σ_i; Uniform draws are scaled to the same σ_i (history) or run from Low to High
    (fixed). The mean shift is applied after the correlation, not through it.
  - Defaults: 10,000 simulations, N = 60 days.
- The VaR page's settings are saved, and the **New Structure builder and Home packets use the same maths**.

### 5.8 ATR (`core/atr_calc.py`)

True range = `max(high − low, |high − previous close|, |low − previous close|)`; **ATR = simple average of the last 14
true ranges**. Days with no volume (a thin contract printing a flat line) are skipped when "skip idle" is on: they neither
enter the average nor become the "previous close". ATR as a % of price is shown for outrights only.

### 5.9 Open-interest roll attribution (`core/oi_roll.py`)

From one day's change in OI per contract: donors (OI fell, total D), receivers (OI rose, total U); rolled R = min(D, U);
each donor's rolled-out amount is its share of D × R, spread over receivers in proportion to their increases. What remains
is exit (donor) or fresh OI (receiver). It is an estimate — exchange OI cannot say who rolled.

---

## 6. The pages

### 6.1 Sidebar and shell

Navigation: **Home, Structures, Correlation, Exposure Map, VaR & Scenarios, Trade Analyzer, Curve Compare, Seasonality,
OI · Volume · ATR, Churn, Archive, Settings.** The sidebar also shows the time of the last price update, a stale-data
indicator and a countdown. A banner appears until an API token is configured. Alert toasts appear on screen.

Three clocks drive the app: a 10 s live-price poll, a 5 s portfolio refresh, a 1 s countdown.

### 6.2 Home

- **Realized bar:** *Total realized P&L (all time)* and *Today's realized P&L* (the realized P&L of exits dated today, UTC).
  A note appears when an account reset is active.
- **Cards:** *Unrealized P&L* (gross, with "net of TC" for the open lots' cost), *Open structures*, *Open legs*, *Net lots
  by product* (sold legs count negative), *Margin used* (usage is not tracked; the configured limit is shown),
  *Largest winner*, *Largest loser*.
- **Stale / missing data banner:** lists symbols in use whose price is older than the staleness threshold and symbols with
  no live price (excluded from P&L).
- **Live structures as packets:** one card per open structure, biggest absolute unrealized P&L first. The header shows the
  name (click to open the structure), **Side**, **Lots**, **Entry**, **Live**, **Target**, **Stop**, **Risk** and
  **Unrealized**; beneath is a row per leg (contract, side, lots, entry, live, risk, unrealized).
  - *Entry / Live* = structure price from leg prices (average entry / live).
  - *Target / Stop* = the values you typed with the most recent trade that set one (blank if none).
  - *Risk* = 1-day historical VaR at 95% over 250 days of that structure alone (settlement history); a leg's risk is that
    leg alone, so legs' risks add up to more than the structure's (they offset inside it). Cached until the book or the
    settlements change.

### 6.3 Structures

**Active grid** (shell and open structures) and a collapsible **closed grid**. Filters: *All active / Shell only / Open
only*, product (CL, BRN, BZ, WBS, G), sort (total P&L, unrealized P&L, days held, name). Row actions open the detail window.

**+ New Structure** (4-step builder):

1. *Template*: Outright, Spread (+1 −1), Fly (+1 −2 +1), Custom (any ratios).
2. *Legs*: symbols and ratios, with live validation (product recognised, month/year, consecutive months, standard ratios).
   **Portfolio impact analysis** (button-triggered):
   - *Correlation check* against every open structure and the whole portfolio, using each structure's outright
     decomposition on settlement data; absolute correlation above **0.35** is flagged (positive = adds to risk,
     negative = hedges).
   - *VaR of 1 lot* of the candidate by **both** the historical and the Monte Carlo method, with the VaR page's settings.
   - *Beta / position sizing* of the candidate against the portfolio: lots to hedge (−β) or match (+β).
3. *Details*: name, **structure type**, multiplier, tick size, tick value, notes. The structure type is what you call the trade, and it is what the Archive, its filter and the analysis group by. It starts as the template's own type; type or pick another (outright, spread, fly, dfly, and the box and diff forms of each, for example `dflybox`, `spreaddiff`). A d-fly traded as two flies is built from the Fly template and stored as a *dfly*. **Leave the numbers blank to use each product's own**
   (CL/Brent 1,000 / 0.01 / $10; gasoil 100 / 0.25 / $25) — a cross-product structure gets each leg its own. Anything typed
   still wins.
4. *Review* then save. History for new symbols is backfilled in the background.

**Structure detail window** (click a row):

- *P&L summary:* unrealized, realized, transaction cost, total, entry, live.
- *Legs:* per-leg lots, entry, live, naked flag, cost per lot, net outright equivalent.
- *Trade history* with delete (deleting a trade replays the structure's remaining trades to recompute lots, average
  price, status and exit P&L).
- **Enter trade / add to position:** structure price, lots, direction, optional **stop** and **target** (structure prices;
  the alert engine fires when the live structure price crosses them), optional *individual leg prices* (used instead of the
  automatic allocation; a leg that is itself a spread or fly may be zero or negative), *trade ideas*, *conviction* (1–5),
  *payups* (whole structure and optional per leg), notes. An add must be the same direction. If
  *Settings → "A first entry must have a stop and a target"* is on (it is on after first start), a new structure cannot be
  entered without both.
- **Exit:** any number of lots (up to what is open) at one structure price. All lots close the structure; fewer leaves the rest
  open on every leg (a *structure partial exit*). The confirmation shows the realized P&L and the transaction cost of
  **only those lots** (their share of the entry cost plus the exit cost). Optional *exit reason* and *payups*.
- **Exit specific legs** (per-leg exit at each leg's own price) is kept for the case where a leg must be closed alone; it can
  leave legs at unequal lots (flagged "naked"), after which the simple exit is replaced by this form.
- **Edit mode:** change leg symbols, ratios and (traded legs) entry prices, with an audit line.
- **Reuse as shell:** a closed structure can be copied back as a fresh shell.

**🔗 Dissolve** (next to + New Structure) closes several structures **against each other with no exit cost** — the case where
you cut a Jan d-fly by trading two flies, so nothing was traded to exit the d-fly itself.

- Tick the structures; set lots (one box for all, or per structure) for a partial dissolve.
- Gross P&L comes from the **leg prices** (one price per leg contract shared by every structure that has it, each
  structure's P&L = Σ leg realized P&L at those prices) **or** a **typed gross P&L** (split between the structures in
  proportion to their entry costs, equally if none, so the total is exactly what you typed).
- Net P&L = gross − the entry cost of the **dissolved lots**. A warning appears if the group does not net to zero in every
  outright contract. At least two structures are required.
- Recorded as a `DISSOLVE` trade (cost 0) on each structure; the structure closes when its lots reach zero (close trigger
  "dissolved"). It counts in today's realized P&L and the archive, and is replayed correctly if an earlier trade is deleted.

### 6.4 Correlation

**Structure correlation matrix (top).** Choose a **target** curve and a **base** curve (CL / BRN / G × outright, spread, fly,
d-fly; default BRN fly vs BRN fly), the number of front structures *N* (default 15), the rolling window and lookback (defaults
from Settings), and a mode:

- *Latest*: each structure is a fixed set of contracts (the front N today), valued every day from **settlements**, so its
  series has no roll jumps; the matrix is target (rows) against base (columns) with the correlation of 5.5.
- *Historical*: pick a date range; the lookback becomes the range and every daily value inside it is averaged (the daily
  values use the days before the range too).

Click a cell for the **correlation history** of that pair (default last 120 days, with the lookback average line). Pick two
structures for their **hedge ratio** (β over a window, default `max(30, R+L−1)` days, shown for the last 60 days) in lots of
the base per lot of the target — converted for contract size across products (5.6) — with its path and the correlation.

**Portfolio positions matrix.** Every open structure against every other, on the daily P&L changes of each position **as held**
(a sold structure is flipped), labelled like `name (BUY 5)`. A negative cell means two positions offset. Same windows; also
computed when the tab opens.

**Custom watchlist** (collapsible; the original tool). Add instruments (typed symbols), open structures, or columns from an
uploaded Excel workbook (a fallback when the API lacks a series: a `Timestamp` column followed by price-level columns,
differenced like everything else). Four views share the list: *Heatmap* (with an as-of date), *Time Series* (rolling
correlation of a pair for several windows), *Year Overlay* (the same pair across years by day of year), *Summary table*
(lookback average, std, min, max, last, beta, R², observations for every pair). A "Backfill missing data" button downloads
history for symbols with none (the only place this tab calls the API).

### 6.5 Exposure Map

Net lots per outright contract across open structures (legs split into outrights, sign rule 3). A bar chart, a table (net,
gross long, gross short, contributing structures) and **concentration warnings**: every delivery month whose absolute net
lots exceed the threshold (default **10 lots**, editable on the page) gets an amber alert naming the structures behind it.

### 6.6 VaR & Scenarios

- **Value at Risk (1-day).** Method: *Historical simulation* or *Monte Carlo (correlated)*. Lookback 30 / 60 / 90 / 252 days.
  Monte Carlo options: distribution (Normal / Uniform), mean, std or low/high, number of simulations, and volatility mode
  (each symbol's own history, or a fixed size). Cards show VaR at 95% and 99%, the lookback and the observations; a
  histogram of the simulated / historical P&L with the VaR cut-offs. Settings are saved and shared with the builder.
- **Scenario analysis (ATR-based shocks).** For each instrument the shock is its **previous completed day's true range**
  (from the daily candle Parquet); the P&L impact of ±1 shock = shock × the instrument's dollar weight. Rows: each
  instrument up and down, then *All instruments* together in each direction (each instrument shocked by its own ATR).
  Instruments with no ATR are excluded from the portfolio-wide rows and named in a note.
- **Beta vs portfolio — position sizing.** Pick a candidate, a window and see β of the candidate on the book's outright P&L,
  with the lots to hedge or match.

### 6.7 Trade Analyzer (Curve Kinks)

Route `/trade-analyzer`. Nothing is typed in to analyse: the engine (`core/curve_service.py`) prices the whole 15-month
ladder (outrights, spreads, flies, d-flies) of CL, BRN and gasoil on every 10 s poll, runs the kink maths on every curve in
the background whether or not the page is open, raises alerts, logs events, saves daily snapshots and keeps data-quality
flags. The page only chooses which curve to display.

**What is a kink.** A contract (or spread / fly / d-fly) that sits away from the rest of its curve. Four methods each produce
a z-score per position:

| Method | Definition |
|---|---|
| **fit** | robust polynomial (degree 3, Tukey-biweight IRLS so a kink does not drag the fit) across the curve; residual ÷ the larger of today's cross-sectional robust scale and that position's historical fit-residual scale |
| **neighbour** | leave-one-out interpolation (cubic Lagrange inside, quadratic near the ends) — equivalent to a second difference; residual ÷ that position's own historical residual scale. **Not scored at the first and last contract** (it extrapolates there) unless `ends_use_neighbour` is on |
| **pca** | the curve rebuilt from the top 3 principal components of its history (level, slope, curvature); leftover ÷ the historical leftover scale |
| **history** | value against its own history at the same generic position (median / robust σ); spreads, flies and d-flies only. Judged **relative to the curve's common move today** (the median z across the curve is removed), so a whole curve shifting is not N kinks |

Scales are `max(1.4826·MAD, standard deviation)` (the MAD alone ignores the fat tails and made z-scores 2–3× too big) and are
**never below the same measure over the last 60 days** (`recent_scale_days`), so a market that has become more volatile does
not flag everything. All scales have a floor (`min_scale`, 0.005 points). A method **flags** a position when |z| exceeds its
threshold (3.0 each). Agreement sets the priority: **HIGH** needs 2 methods, **MEDIUM** one. A kink that is the
neighbour-method echo of a stronger neighbouring kink is recognised and suppressed (the strongest flagged point is masked,
all methods re-run, repeat up to 6 times). A curve with more than 40% of its points flagged is treated as a data mismatch or
a whole-curve move and reports no kinks. **Seasonality:** the neighbour residual of the same delivery month at the same time of
year (±15 days) in earlier years gives a seasonal z; if |z| ≤ 1.5 the kink is "seasonally normal" and drops one priority level.

**Direction.** A contract rich against its curve is a **SELL**, a cheap one a **BUY** (`trade_side`).

**Tabs:** *Kinks* (the current kinks, strongest first, with z by method and fair values), *Change vs settlement*, *Trade
plans*, *Data quality*, *History log* (every kink event with its detail), *Backtest*, *Thresholds* (editable).

**Trade plan** for a kink (`core/curve_trade.py`):

- *Entry*: live price of the structure at alert time.
- *Stop*: `min(vol_stop_mult × daily move, risk_per_lot ÷ point_value)`, never narrower than one tick; the plan says when the
  per-lot cap wins.
- *Lots*: `floor(risk_per_trade ÷ (stop distance × point value))`, capped at `max_lots`. Point value scales with the product's
  contract size (gasoil is a tenth).
- *Target*: `entry + (fair − entry) × share`, fair = median fair value of the methods that flagged; the share is what such
  kinks have historically closed within `reversion_horizon` days (backtest), floored at `min_reversion_fraction`.
- *Time stop*: about 2 half-lives (AR(1) on the neighbour residual history).
- *Hedges*: candidate curve types per kink type (Settings), same product, optionally not sharing a contract; each scored on
  the last 120 days of changes; the ratio minimising the 1-day historical 95% VaR of trade + h × hedge is chosen, with the
  minimum-variance ratio shown beside it; the best two with |correlation| ≥ 0.5 are listed.

**Backtest.** For each of the last N days the detector runs exactly as live but fitted only on history before that day; each
flagged kink is followed forward for `reversion_horizon` days and its residual compared with the flag-day residual (reverted
= shrank below half). Read across groups (1 vs 2 vs 3+ methods), not as proof.

**Alerts.** A kink alerts when it first appears at or above the minimum priority (MEDIUM), when its priority rises, or when it
reappears after the cooldown (60 min). It must be present on **3 polls in a row** before it alerts (a one-poll flicker never
does), a kink resting on a price older than the stale limit (120 s) never alerts (it is still shown), and one alert per product
per poll lists related kinks. **By default only d-flies alert** (ticks per product and curve type on the Thresholds tab;
others are still shown and logged). Alerts also fire on a product's "buffer day" (a contract treated as rolled out one business
day before its rule-based expiry) asking you to confirm the expiry date.

**Calendar.** Contracts are live while `as_of < buffer day` (one business day before the rule-based expiry). The gasoil rule
was checked against the exchange's own front-contract history and agrees on every day of the last 260.

Defaults (all editable on the *Thresholds* tab): z 3.0 ×4; methods for HIGH 2; max kinked share 0.4; seasonal z 1.5, window 15
days; lookback 504 days; PCA components 3; polynomial degree 3; min scale 0.005; alert from MEDIUM; cooldown 60 min;
persistence 3 polls; recent-volatility window 60 days; stale 120 s; snapshot every 15 min; 5 years of history. Trade plan:
risk per trade $5,000, per lot $1,000, point value $1,000 (per 1,000-unit contract), max 100 lots, volatility stop 1.5 × the
60-day daily move, reversion horizon 10 days, minimum reversion share 0.5, minimum reward:risk 0 (off).

### 6.8 Curve Compare

Route `/curve-compare`. Pick a product (CL / BRN / G), months out (1–36), curve type, and for a d-fly how its price is built.
The chart shows the **live curve** against the **previous settlement**; points built from other quotes are hollow diamonds,
and a curve stops where nothing can be priced. A table lists live, previous settlement and the change.

- **Live price of a symbol** = size-weighted mid of the best bid and ask: `(bid × ask size + ask × bid size) ÷ (bid size +
  ask size)` (the side with less size trades next); plain mid with no sizes. No usable quote → `/fairvalue/`.
- **Quote quality:** a quote is used only if its bid-ask gap is at most 25 (outright) or 10 (spread/fly) **CL-ticks** — scaled
  by the product's tick, so for gasoil 25 / 10 of its own ticks. Back-month outright quotes are often junk, and a spread
  built from one would be wildly off, so a wide quote is ignored and the contract priced from its fair value.
- **Where a point's price comes from:** its own price (the newer of quote VWAP and fair value), or one built from other points
  (spread k = outright k − outright k+1; fly k = quoted fly, else spread k − spread k+1; d-fly k = fly k − fly k+1, or the
  3-month-spread or 1-month-spreads variants). The pieces of one built value must have been priced within 30 minutes of each
  other; a built price replaces an own price only if it is more than 5 minutes newer; an own price older than a day is used
  only when nothing can be built ("old price"), older than a week not at all. Every value carries its age.
- **Quotes** come from `/taq/` (paged, oldest first, so the newest quotes are on the last page — the store reads backwards until
  every symbol asked for has been seen). Polling happens only while the page is open. Gasoil has quotes for some contracts
  only (the front November contract had none when checked); those points use fair value.
- Previous settlements (from `/common/dailymarketdata/`) are shared with the Trade Analyzer; spread/fly settlements are built
  from outright settlements with the family's weights.

### 6.9 OI · Volume · ATR

Three tabs on one page, per product (CL, BRN, G).

**Live monitor.** For each of the next N live contracts (default 15) the latest OI (previous trading day's) is compared with
what **the same delivery month's finished contracts had at the same number of calendar days before their expiry**, over the
last 5 years — a January against past Januaries, never other months. Measures: absolute OI (while building), % of the
contract's peak so far (once declining), and the share of the product's total OI (context). **z = (x − median) ÷ scale**, scale =
the largest of 1.4826·MAD, the standard deviation and 10% of the median (with five past contracts a MAD can be tiny by chance).
A day is *deviating* when |z| ≥ 2.5 **and** OI ≥ 5,000 **and** the implied OI gap ≥ 2,000. The run of consecutive deviating
days: 1–5 trading days is **SUDDEN** (alerts), longer is **SUSTAINED** (shown, no alert). Also: a roll estimate (5.9), a chart of
one contract against its baselines, and a **front-contract check** (in the last 30 days, is it draining like finished contracts
did — how much is still open, how much has dropped, how much rolled into the next contract). Alerts go out once a day when fresh
data arrives; each deviation alerts once.

**Research** (finished contracts only, same delivery month together): the peak OI of each contract and how many days before
expiry it came; how long before expiry it reaches 10 / 25 / 50 / 75 / 90 / 100% of its peak and what share is still open N days
out; the last month day by day (share of peak open, the fall that day split into rolled and exited, and which following months
the rolled OI went to); and statistics by delivery month. Gasoil expires mid-delivery-month, so its cycle differs from CL/Brent.

**ATR.** Parameters: period (14), contracts (15), years (5), z threshold (2.5), skip idle days. Views, per product and family
(outright / spread / fly): a *live table* (today's ATR against the same month's usual at the same days to expiry), a *cycle
chart* (one contract's ATR path against past years), *month tables* (ATR N days before expiry by delivery month, in points, % of
price, or a multiple of the contract's ATR when it became the front), a *hedge table* (ratio of the k-th contract's ATR to the
front's, plus the minimum-variance hedge ratio and correlation) and a **Hedge calculator** — pick any two contracts (any
product, any family) and a lookback (default 60 shared days): the answer is the minimum-variance lots of B per lot of A (β),
converted for contract size across products, the correlation, the daily risk before and after hedging, and an ATR-ratio
alternative (which matches the size of daily moves but not their direction and is not scaled for contract size).

Data: OI / settlements download from January 2019 for 36 months ahead (history window 1,500 days before expiry, Settings);
ATR candles for outrights, consecutive spreads and consecutive flies (400 days).

### 6.10 Churn

Route `/churn`. "Which contracts offer the best intraday small-target churning?" For every live outright, spread, fly and d-fly
of CL, BRN and gasoil, long and short, it simulates the trade you describe from **five-minute candles** and ranks them.

**The trade (`core/churn_engine.py`).** Enter at a candle close by crossing the spread (fill = close + half the spread, +
"worse" ticks when testing a worse fill). Target = `target` ticks above the fill, stop = a multiple (default 3×) below. A later
candle's high at the target is a win (+target); its low at the stop is a stop-out (−stop and the spread to cross; if one candle
holds both, the stop is assumed first). A trade open at the end of the window is closed at the close. Nothing is held outside
**your trading window (London time, default 09:00–19:00; candles are UTC and converted, so summer time is handled)**.

**The day ladder** (headline figures): buy at the market at the start of the window; that fill is the *level*. After each win,
sell and wait up to `patience` minutes (60) for the price to return to the level, then buy there again; if it does not come
back, or after a stop-out, move on and take the new fill as the new level. Net per day, trades per day, hit probability,
positive days and the blow-up rate come from this, day by day. Days with no trading are excluded; an instrument that printed
nothing on a day counts as a flat day.

**Out of sample.** The target is picked on the first 60% of days; every ranking figure comes from the last 40% (no split with
fewer than 15 days).

**Costs (`core/churn_costs.py`).** Per lot per side: transaction cost (TC) minus rebate (rebates are **off** by default). A
structure is charged on its **net position per contract month** (a spread is two lots, a fly four, a d-fly eight; a month two
structures hold in opposite directions nets down). Default table: CL TC 0.89 (rebate 0.20), BZ 0.89 (0.77), BRN 0.95 (0.25),
WBS 1.02 (0.70), gasoil 1.05 (0.25); tick values $10 (gasoil $25). Quoted CL/BZ combos use rebates 0.87 / 0.81. The table is
editable on the page. A target must earn at least the round-trip cost plus a margin ($3 per lot) to be allowed.

**Churn score (0–100)** = 100 × the geometric mean of five 0–1 parts (none taken below 0.05): **edge** (net $ per day per lot,
ranked among eligible rows; rows that lose money score 0), **hit** (share of ladder trades that reached target), **worse fill**
(how much of the net per day survives an entry 1 and 2 ticks worse), **stability** (share of positive days), **tail** (1 − blow-up
rate; blow-up = a day losing more than three stop-outs' worth). A row below `min_volume` (100 lots/day) is shown but not scored.

**Table columns:** Contract, Live Price, Volume/day, ATR, Cost, Per-lot net profit, Number of oscillations, Score, Side, Worse
price score. *Oscillations* = from the live price, how many times in the last 5 trading days the market came back to that level
and then moved the target in the trade's favour, in one day. The Score's colour and tooltip carry the **live verdict** — GO if
the price is not on the bad side of its last-day average (z), volatility is within 0.5–2× usual, the current London hour is not
clearly weak and the price is fresh; WAIT with reasons otherwise; CLOSED outside your window; none without a price younger than
2 minutes. Below: worse-fill survival, best hour of the day, and the cost table.

**D-flies** have no quote; their candles are derived (`fly(k) − fly(k+1)`, close to close, so extremes inside a candle are not
seen — a target is slightly harder to hit than on a real instrument) and they are priced with no bid-ask.
Settings: contracts per family (12), window 60 calendar days, largest target 6 ticks, spreads per family (1 tick; d-fly 0),
start-from contract (4), oscillation days (5). Candles refresh every 10 minutes while the page is open.

### 6.11 Archive and the trade journal

**Archive → Trades: one row per exit.** An exit of some of a structure's lots is in the Archive as soon as it is made (the rest
stays open on the Structures page and in *Lots Still Open*); exiting the remainder adds a second row. A structure exited in two
steps is two rows; its first row keeps the structure's id, later ones are `<id>#2`, `#3`. **Each exit takes its lots from the
earliest entries first (first in, first out, `core/fifo.py`)**, which gives that row: its *entry price* (the lots-weighted price
of the entries it took lots from), *entry date*, the *entry transaction cost of exactly those lots*, the stops and targets typed on
those entries (risk:reward, risk at stop, R) and the market snapshot taken at that entry. Example: 3 lots bought Monday at 0.50 and 2
bought Tuesday at 0.60; an exit of 4 lots is entry 0.525, Monday; the exit of the last lot is entry 0.60, Tuesday. The row's realized
PnL is the figure stored on the exit trade (first in, first out as well: see 5.1; what Home uses, and what the editor corrects), net
of the exit's cost and those entry costs, so a structure's rows always add up to its total. Every entry and every exit now has its own market snapshot (kept by trade); *Fill market data for old trades* fills the
ones that are missing. Structures with a per-leg exit keep one row once closed. *Delete selected* removes a structure with all its
exit rows and only once it is closed.

**Archive → Trades.** Closed structures with filters (closed between, type, symbol contains, winners/losers) and summary cards
(trades, winners, losers, win rate, total / average / best / worst P&L, all **net of transaction costs**). Columns: name, type,
legs, entry and exit date, days held, entry and exit price, lots, transaction cost, realized P&L, result, then the journal
columns below. Row detail shows the legs, the facts, the journal, and an editor for ideas, payups, conviction, exit reason and
mistakes.

- **Delete selected** permanently deletes ticked closed trades (with their legs, trades and history). Home's realized P&L (all
  time and today's) drops by their net P&L automatically. If an Account Reset is active, a tickbox keeps the shown figure
  unchanged for trades closed before the reset (the baseline moves with them).
- **⬇ Export to Excel** downloads exactly the trades the table shows: sheets *Trades* (every column, real numbers and dates,
  frozen header, filters, $ formats), *Legs* (a row per leg), *Summary*.
- **📊 Fill market data for old trades** backfills snapshots (3 trades per click; each trade costs 6 `/ohlc/` requests) and
  excursions.

**✏️ Edit this trade (every column).** The detail panel of a trade has an *Edit this trade* button with four tabs, so any archive
column can be corrected afterwards (`core/archive_edit.py`). Each tab saves on its own, writes one audit line to the structure's
notes (`archive edit: ...`), and the table, stats and analysis reload:

| Tab | Changes | Archive columns it drives |
|---|---|---|
| **Structure** | name, **structure type** (any of the 12, or Custom), notes | Structure Name, Type |
| **Trades** | per trade: date and time (UTC), price, lots, buy / sell, transaction cost, realized P&L (gross), stop, target | Entry / Exit Date, Days Held, Entry / Exit Price, Lots, Transaction Cost, Realized PnL, Result, Risk : Reward, Risk at Stop, Result in R |
| **Legs** | symbol and ratio of each leg (the number of legs stays; new contracts are created as in the builder) | Legs |
| **Market data** | the 15 entry and exit snapshot cells (prices, day / 1-minute move, curve shapes, ATR, volatility, volume, bid-ask, Brent-WTI, crack, curve z) and the four MFE / MAE figures | the Entry / Exit market columns, MFE / MAE |

Ideas, payups, conviction, exit reason and mistakes are edited in the journal editor above it. Columns that are worked out (Days
Held, Result, Risk : Reward, Risk at Stop, Result in R) have no value of their own: change what they are worked out from.

Prices and P&L: changing a trade's price, lots or buy / sell (with *Recalculate* ticked, the default) replays the structure's
trades exactly as deleting a trade does: the legs' entry prices and each exit's realized P&L are recalculated, and a cost is scaled
in proportion to new lots (a price never changes a cost). A P&L or cost typed in the same save is kept as typed; with
*Recalculate* off everything is stored as typed. An edit that would leave lots open, or put a stop or target on the wrong side of
the price, is refused and nothing is changed. Realized P&L feeds Home's totals, so a corrected trade moves them too.

**The journal** is one JSON document per structure (`trade_journal` table):

| Column group | What is recorded | How |
|---|---|---|
| **Outright prices** (CO, CL, GO) | last traded price of each product's front outright at entry and at exit | last 1-minute close, else last hourly close (within 3 days), else the latest daily close; from `/ohlc/`, not `/fairvalue/` |
| **Day move / 1-minute move** | Bullish / Bearish / Stalemate for the front outright of the structure's own product | *Day*: the day so far (hourly candles from 00:00 UTC to the trade, so a past trade is judged on what was known then). *1-minute*: the last one-minute candle. Rule: close above open = Bullish, below = Bearish; **Stalemate** if high − low ≥ 5 × |open − close|; but if |open − close| ≥ 50 ticks the direction counts regardless |
| **Curve shape** (CL, BRN, GO) | Contango / Backwardation / Contango – Backwardation / Backwardation – Contango (or Flat) | first 6 values of the structure's curve type — **fly if any leg is a fly, else spread, else outright** — built from outright prices (spread = M1−M2…, fly = M1−2M2+M3…). Rising with maturity = Contango; a rise that turns into a fall = Contango – Backwardation. Moves under 5% of the range (or a tick) are not a turn; a zig-zag is decided by the bigger turn |
| **ATR** | ATR of the whole structure per lot | a quoted spread/fly/outright → its own ATR (as on the ATR page); else the structure's price built from the legs' stored closes and the 14-day mean absolute close-to-close change |
| **Volatility / volume vs normal** | last 24 hourly candles of the front contract vs the preceding 10 days | st.dev. of hourly close changes, and volume per 24 candles; 1.0 = normal |
| **Brent–WTI** and **gasoil crack** | from the prices above | CO − CL; GO ÷ 7.45 − CO |
| **Bid-ask (ticks)** | live trades only | one `/taq/` request for the quoted structure, when `/taq/` has ≥ 3 spare requests |
| **Curve z-score** | live trades only | the Trade Analyzer's largest |z| for a structure on its ladder (+ rich, − cheap) |
| **Risk : Reward** | not stored; worked out | entries' average stop and average target (plain averages) against the lots-weighted entry price: `reward ÷ risk`, shown `1 : 2.00` |
| **Risk at stop, Result in R** | not stored | risk $ = \|entry − stop\| × lots × multiplier; R = net P&L ÷ risk $ |
| **Trade ideas** | any number per trade, from the list in Settings | multi-select at entry or in the Archive |
| **Payups** (entry / exit, optional per leg) | ticks paid up before the fill, typed by hand | e.g. order 0.08, fill 0.09 = 1; totals add up across adds / exits; only recorded, no P&L effect |
| **Conviction, exit reason, mistake tags** | 1–5; reason from a list; tags from a list | lists editable in Settings |
| **MFE / MAE** (best / worst open P&L) | in $ at the size entered, intraday **and** daily | marks the position at every point from first entry to last exit at the lots-weighted entry price; *daily*: structure price from settlement closes (free); *intraday*: leg candle closes (1M up to 36 h, 5M up to 7 days, else 1H), one `/ohlc/` request, **only when `/ohlc/` has ≥ 8 spare requests** (otherwise daily only, and the backfill adds the intraday later) |

Every market column is recorded **at entry** (the first entry; adds do not replace it) **and at exit** (each later exit
replaces it, including a dissolve). The snapshot is taken on a background thread after the trade is saved and costs three
`/ohlc/` requests; nothing blocks the trade.

**Archive → Analysis.** Built on the trades the filters leave; every figure is per trade, **net of costs**.

- *Headline cards:* trades (and how many have an R), win rate, expectancy per trade ($ and R), profit factor (gross wins ÷
  gross losses), payoff (average win ÷ average loss), total net P&L and costs paid, worst drawdown with the longest losing and
  winning runs, average payups.
- *Equity curve and drawdown* (cumulative net P&L in exit order; the start counts as a peak).
- *Group-by table and chart:* pick one column — structure type, product, trade idea, curve shape, outright price range (thirds
  of each product's own prices), day move, 1-minute move, stop distance in ATRs, planned risk:reward, conviction, exit reason,
  mistake tag, entry hour (London), weekday, days held, payups, volatility, volume, bid-ask, Brent–WTI, gasoil crack, and whether
  the trade was entered with or against the curve's mispricing — and see trades, win rate, average win/loss, payoff,
  expectancy ($, R), profit factor, total and average payups. Idea and mistake tags count a trade under each tag.
- **A group with fewer than 15 trades is marked "low" (greyed)**: with so few trades a win rate or expectancy is mostly luck.
- *Two columns together:* every pair of groups of two columns.
- *What separates winners from losers:* for each column, a chi-squared test of whether the win rate differs between its groups
  (groups under 5 trades left out), the spread of win rates, **Cramér's V** (0 = the column says nothing about winning, 1 = it
  decides it) and the p-value. Because many columns are tried, a column **"stands out"** only if its p-value beats 5% divided by
  the number of columns tested (Bonferroni); one that only beats 5% is **"worth a look"**. Below 100 trades the table is
  labelled a pointer, not a result.

### 6.12 Settings

| Section | Contents |
|---|---|
| **API Configuration** | Bearer token (stored in the settings table, shown masked), *Save Token*, *Test Connection* |
| **Risk & Alert Thresholds** | portfolio P&L stop (default −$50,000), structure max loss (−$10,000, stored but not evaluated by an alert today), staleness threshold (10 s), portfolio margin limit ($1,000,000, shown on Home) |
| **Analysis Defaults** | VaR confidence (95%, stored), correlation rolling window (5) and lookback (20) |
| **Curve Kinks** | opening time and time zone of the daily previous-settlement fetch; trade-plan settings (risk, stop multiple, volatility window, hedge types…) |
| **OI / Volume / ATR** | OI history length (1,500 calendar days before expiry), ATR history length (400) |
| **Trade Journal lists** | trade ideas, exit reasons and mistake tags (one per line), and the switch "a first entry must have a stop and a target" |
| **Microsoft Teams Alerts** | webhook URL (masked, https only), enable switch, *Test*, *Save* |
| **Account Reset** | *Reset total P&L to zero* (stores baselines for total and realized P&L, confirm first) and *Clear reset* |
| **Data Management** | last sync summary (symbols tracked / ok / error), stored data size, *Run sync now* |

Validators reject bad input with a readable message (text in a number field, a positive P&L stop, a non-https webhook) before
anything is stored. The token and the webhook are never logged.

### 6.13 Seasonality

Route `/seasonality`. Where is a structure today compared with the same structure in earlier years at the same point of its
life? For one product (CL / BRN / G) and curve type (outright / spread / fly / d-fly) the table lists the structures of the
first N contracts from the front (default 15, up to 36); clicking a row plots them.

**Comparison (`core/seasonality.py`).** The Apr26 d-fly is set against the Apr25, Apr24 ... d-flies (default 7 earlier
years; the settlement store starts in 2019) at the same number of **trading days before expiry** — the exchange business days
to the expiry of the structure's *front* leg (0 on the expiry day). A finished contract's expiry is the last day its front leg
settled; a live one uses the rule-based date. Prices are settlements; a spread, fly or d-fly is the weighted sum of its outrights'
settlements on the days every outright settled. A gap of up to 3 days is carried forward when years are lined up.

**Band.** At each day before expiry: the mean ± *z* sample standard deviations (default z = 2) of the earlier years, or the
robust median ± *z* × 1.4826 × MAD. The spread is never below one tick; a band needs at least two years. Any delivery year can be
left out of the average (it stays on the chart, dotted and grey). The choices are not stored: they reset on every visit.

**Status.** The live price (Curve Compare's quote VWAP, fair value as fallback; the last settlement if there is none) is
scored as z = (price − centre) ÷ spread. Outside the band on one side for the chosen number of days in a row (default 3: the
run of settlements ending at the last one, plus today when the live price is on the same side) = **FLAGGED RICH** (above, a
sell candidate) or **FLAGGED CHEAP**; outside for fewer days = **WATCH**. A note appears when most of the table is outside the
band on one side — that is the whole market in a different regime, not individual mispricings.

**Table columns:** days to expiry, live price and source, last settlement, live − settlement, the usual price at this time of
year, deviation in ticks and z, how many earlier years were priced below today, days outside the band, status, **same move in
earlier years**, volume of the thinnest leg on the last settlement day, the usual volume at this point, and their ratio.
*Same move in earlier years* (shown for WATCH / FLAGGED rows) lists each other year that left a band built from the remaining
years, on the same side, within 10 trading days of today's point, with its peak z and whether it was back inside 20 trading
days later ("back" / "still out").

**Chart.** Every earlier year against days before expiry (axis runs right to left), this year's settlements in white, the
band and average (tick box), and a star for today's price. The y-axis is sized to the 2nd–98th percentile of the plotted values,
because the last days before expiry swing far more than the rest; zoom out to see them.

**Rebasing.** *Show → Change since a reference day* plots each curve as the change since N trading days before expiry (default
250). It removes the oil-price level (Brent at 20 dollars in 2020, 120 in 2022) that dominates an outright; a structure that is
nearer expiry than the reference day cannot be rebased and shows no data.

**Depth.** The watch window (days before expiry) reaches as far back as the stored settlements: each contract keeps
`OI history` calendar days (Settings, default 1,500, about 1,030 trading days). The page says so when the window asks for more.
No extra API calls are made by this page itself except Curve Compare's quote polling while it is open.

---

## 7. Alerts

`AlertManager` persists every alert (in-app) and posts to Microsoft Teams when a webhook is set and enabled. It never raises;
the webhook URL is a secret and is never logged.

| Alert | When | Level |
|---|---|---|
| Portfolio P&L stop | total P&L falls below the stop; once per breach | critical |
| Stop-loss / target | a trade's structure price crosses the stop or target typed with the entry (buy: stop at or below, target at or above; sell: reverse); repeats every 5 minutes until you stop it on the structure; the stop toast stays until dismissed, the target toast closes after 8 s | critical / info |
| Curve kink | see 6.7 (persistence, d-fly default, cooldown) | warning (HIGH) / info |
| Contract buffer day | a contract treated as rolled out one business day early | warning |
| OI deviation | a contract has just become SUDDEN outside its same-month baseline; one alert per product, once per deviation | warning |
| Stale data | shown on Home and the sidebar, not sent | — |

Titles are cut to 100 characters and bodies to 500 (Curve Kink bodies are allowed longer).

---

## 8. Storage: database, files, settings keys

### SQLite (`db/crude_oil_risk.db`, `db/schema.sql`; the only SQL is in `db/repository.py`)

`contracts`, `structures`, `legs`, `trades` (with `stop_loss_price`, `target_price`, `transaction_cost`), `pnl_records`
(periodic snapshots per structure, used for today's P&L), `alerts`, `settings` (JSON values keyed by name),
`curve_kink_events` (the Trade Analyzer log), `trade_journal` (one JSON document per structure). New columns are added by safe
`ALTER`s at start.

Trade event types: `trade`, `add`, `partial_exit` (one leg), `structure_partial_exit` (some lots at one structure price),
`full_exit`, `dissolve`, `roll`.

### Files (`data/`)

| Path | Contents |
|---|---|
| `data/historical/<API code>/<symbol>.parquet` | daily candle cache |
| `data/oi/` | `<product>.parquet` (contract, date, oi, close, volume) + `meta.json` |
| `data/atr/` | `<product>.parquet` (symbol, date, OHLC, volume) + `meta.json` |
| `data/churn/` | `<product>.parquet` (5-minute candles) + `meta.json` |
| `data/curves/` | `generic/<product>/…` generic-position history, `snapshots/<product>.parquet`, `prev_settlement.json` |
| `data/correlation_uploads/` | uploaded Excel workbooks |

The OI, ATR and Churn stores write to a temporary file and then replace the real one, so an interrupted write cannot leave a half-written file.

### Notable settings keys

`api_access_token`, `alert_portfolio_pnl_stop`, `staleness_threshold_seconds`, `margin_limit`, `correlation_rolling_window`,
`correlation_lookback`, `pnl_reset_baseline`, `pnl_reset_realized_baseline`, `teams_webhook_url`, `teams_alerts_enabled`,
`curve_params`, `curve_open_time`, `oi_history_days`, `atr_history_days`, `var_mc_params`, `churn_costs`, `trade_ideas`,
`journal_exit_reasons`, `journal_mistakes`, `require_stop_target`.

---

## 9. Configuration (.env)

Copy `.env.example` to `.env`. Highlights (all have defaults):

| Variable | Default | Meaning |
|---|---|---|
| `APP_HOST`, `APP_PORT` | `0.0.0.0`, `8050` | bind address |
| `APP_DEBUG` | `False` | Dash debug mode |
| `AUTH_USERNAME`, `AUTH_PASSWORD` | empty | HTTP Basic login; **empty = no login** (fine only on localhost) |
| `AUTH_SECRET_KEY` | empty | fixed session secret (otherwise regenerated per start) |
| `DB_PATH` | `db/crude_oil_risk.db` | SQLite file |
| `HISTORICAL_DATA_DIR`, `CURVE_DATA_DIR`, `OI_DATA_DIR`, `ATR_DATA_DIR`, `CHURN_DATA_DIR`, `CORRELATION_UPLOADS_DIR` | `data/...` | store locations |
| `ALERT_PORTFOLIO_PNL_STOP`, `ALERT_STRUCTURE_MAX_LOSS` | −50,000 / −10,000 | defaults for the Settings page |
| `LIVE_STALENESS_THRESHOLD_SECONDS` | 10 | a live price older than this is stale |
| `TEAMS_WEBHOOK_URL`, `TEAMS_ALERTS_ENABLED` | empty / False | Teams (also editable in Settings) |
| `DEFAULT_VAR_CONFIDENCE`, `DEFAULT_CORRELATION_ROLLING_WINDOW`, `DEFAULT_CORRELATION_LOOKBACK`, `DEFAULT_MARGIN_LIMIT` | 0.95, 5, 20, 1,000,000 | analysis defaults |

The API token is entered on the Settings page and stored in the database, not in `.env`.

---

## 10. Deployment

### Windows service (NSSM)

```bash
service\nssm_install.bat      # elevated prompt; registers CrudeOilRiskManager (starts ui/app.py, logs to logs/, restarts after 5 s)
nssm start CrudeOilRiskManager
service\nssm_uninstall.bat
```

### Docker

```bash
docker compose up --build
```

Builds a multi-stage image (`python:3.11-slim`, served by `gunicorn` on `ui.app:server`), maps port 8050, mounts `./data` and
`./db` so history and the database persist, loads `.env`, and restarts always.

### Opening the dashboard from another PC

The app binds to `0.0.0.0:8050`, so it listens on every interface; two things stand between it and another PC:

1. **A login.** Set `AUTH_USERNAME` and `AUTH_PASSWORD` (HTTP Basic via `dash-auth`). Without them anyone who can reach the
   address sees all positions and P&L.
2. **Reachability.** Same network: allow inbound TCP 8050 in the firewall and browse to the machine's IP. Different network:
   use [Tailscale](https://tailscale.com/) (private encrypted mesh, no port-forwarding). **Never forward port 8050 to the public
   internet.**

Only **one running copy per API token** should be active: the minute budgets are shared across copies.

---

## 11. Development

```bash
pytest                      # ~1,300 tests, ~2–4 minutes
```

Shared fixtures are in `tests/conftest.py` (temporary database, synthetic Parquet). Tests cover the maths (P&L, positions,
costs, correlation, VaR, kinks, journal, analysis), the stores and services, and the callbacks as plain functions.

**Adding a product** (e.g. another ICE contract): register it in `core/curve_calendar.PRODUCTS` (API code, calendar, expiry rule,
multiplier / tick / tick value), add its translator entry in `adapters/base.PRODUCT_TO_API_CODE`, its churn cost row in
`core/churn_costs.DEFAULT_COSTS`, and check that its candles and settlements come back from the API for the usual
codes (a small read-only probe is enough). Every curve-based page loops over `PRODUCTS`.

**Adding a data adapter:** implement the interface in `adapters/base.py`, place it under `adapters/live/` or
`adapters/historical/`, use a mock under `adapters/mock/` for offline work, and wire it in `ui/app.py`.

**Rules to keep when changing code:** put a rule in one place and import it; add a test for every behaviour change; keep layouts
static and callbacks thin; never put a Dash component library's first import inside a callback; write files via the
temp-file-and-replace helpers.

---

## 12. Assumptions and known limits

- Daily "closes" of CL / BRN / G are **settlements**; the daily candle closes can differ from them by up to ~0.6 points.
- Expiry dates are rule-based approximations (holiday calendars per exchange; one-off closures are not modelled). Contracts
  roll out one business day before the rule date; the page tells you when it used that buffer.
- The API's daily candle for the last day or two is not available yet, so anything "so far today" is built from hourly candles.
- `/fairvalue/` outright prices are not used for journal snapshots; its composites for far months may be derived from simpler
  contracts and are marked as such. A composite is derived only if its components were priced within 30 minutes of each other.
- MFE / MAE from daily closes (and from combining legs' candle closes for multi-leg structures) can miss a spike inside a
  period; they are a floor on how far the trade really went.
- ATR of a structure with no quoted contract is a close-to-close approximation (highs and lows of a combination are unknown).
- OI roll attribution is an estimate; exchange OI cannot say who rolled.
- Correlation and VaR need history: a pair with fewer than 20 shared days shows no correlation; a book whose shortest-history
  symbol is short has a short VaR sample (the result says how many days it used).
- The structure-max-loss threshold and the VaR-confidence default are stored settings that no alert or page currently
  evaluates; the margin limit is displayed on Home but margin usage is not tracked.
- The Curve Kinks outright curves of Brent and CL still flag about one kink a day; alerts default to d-flies, where the rate is
  lower. Gasoil flies flag more often than the other products' (steep recent backwardation and 0.25-tick price steps).
- With few archived trades the journal analysis can only point to where to look; groups under 15 trades are marked "low" and
  the winners-vs-losers table is labelled a pointer below 100 trades.
- Churn candles of d-flies are derived close to close; targets are therefore slightly harder to hit than on a real instrument.
- First in, first out needs each entry's quoted structure price. A structure with real per-leg fills typed at entry, a hand-corrected
  entry price, or a per-leg exit keeps the average-cost figure. On the first start after the move to FIFO, the stored exit P&L and the
  entry price of lots still open are re-allocated once (`core/fifo_migration.py`, one audit line per structure); a structure with a P&L
  typed in the Archive editor is left exactly as it is. A structure's total P&L never changes, only the realized / unrealized split.
- Single-user: there is no per-user data; the optional Basic Auth login protects the whole app.
