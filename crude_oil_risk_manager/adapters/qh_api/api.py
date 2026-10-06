"""One method per QH API endpoint (https://qh-api.corp.hertshtengroup.com/apis/swagger/).

Every method returns the endpoint's parsed JSON untouched - what each endpoint's data MEANS
for the dashboard is decided where it is integrated, not here. Methods only build the request
(list params joined, None dropped, documented size limits checked) and let QHClient apply the
rate limits. Endpoints whose use has not been decided yet are still wrapped here so they are
ready to call.

Rate limits (per token, per endpoint): default 7/min, 420/h, 10,080/day; /ohlc/ 30/min,
1,800/h, 43,200/day; /tas/ 10/min, 600/h, 14,400/day; /common/instruments/,
/fundamental/codes/ and /taq/products/ are public and unlimited. GET and POST on the same
path share one budget. The live price poll and the historical sync call /fairvalue/ and
/ohlc/ through this client too, so all callers share each endpoint's limiter.

Use ENDPOINTS / list_endpoints() to see every endpoint and its limit.
"""

from typing import Any, Iterable

from adapters.qh_api.client import EndpointSpec, QHClient
from adapters.qh_api.rate_limiter import OHLC_LIMIT, TAS_LIMIT

MAX_INSTRUMENTS = 50  # /ohlc/, /vap/, /fairvalue/historical/
MAX_FUNDAMENTAL_QHCODES = 10
MAX_OPTIONS_OI_CONTRACTS = 25
OHLC_INTERVALS = frozenset({"1M", "5M", "1H", "1D"})
VAP_INTERVALS = frozenset({"5M", "1D"})
OHLC_EXTRA_FIELDS = frozenset({"buyvolume", "sellvolume"})
FAIRVALUE_HISTORICAL_FIELDS = frozenset({"open", "high", "low"})

_P = frozenset
_OFFSET = {"limit", "offset"}

ENDPOINTS: dict[str, EndpointSpec] = {
    s.name: s
    for s in [
        EndpointSpec("daily_market_data_summary", "GET", "/common/dailymarketdata-summary/",
                     params=_P({"product", "start", "end", "count"} | _OFFSET), pagination="offset"),
        EndpointSpec("daily_market_data", "GET", "/common/dailymarketdata/",
                     params=_P({"qhcode", "gdcode", "start", "end", "count", "fields"} | _OFFSET), pagination="offset"),
        EndpointSpec("economies", "GET", "/common/economies/"),
        EndpointSpec("economy_premiums", "GET", "/common/economies/premiums/", params=_P({"economies"})),
        EndpointSpec("fixings", "GET", "/common/fixings/",
                     params=_P({"name", "economy", "start", "end", "page", "page_size"}), pagination="page"),
        EndpointSpec("instruments", "GET", "/common/instruments/", limit=None,
                     params=_P({"qh_codes", "tt_instrument_ids"} | _OFFSET), pagination="offset"),
        EndpointSpec("curves", "GET", "/curves/", params=_P({"products", "strategies", "range", "start", "end"})),
        EndpointSpec("fairvalue", "GET", "/fairvalue/", params=_P({"products"})),
        EndpointSpec("fairvalue_post", "POST", "/fairvalue/"),
        EndpointSpec("fairvalue_historical", "GET", "/fairvalue/historical/",
                     params=_P({"instruments", "timeinterval", "fields", "count", "start", "end"})),
        EndpointSpec("fundamental_codes", "GET", "/fundamental/codes/", limit=None,
                     params=_P({"page", "page_size", "qhcode", "asset", "country", "frequency"}), pagination="page"),
        EndpointSpec("fundamental_data", "GET", "/fundamental/data/",
                     params=_P({"qhcode", "count", "start_date", "end_date", "page", "page_size"}), pagination="page"),
        EndpointSpec("generic_chart_data", "POST", "/generic-charts/generic/data/"),
        EndpointSpec("gtc", "GET", "/gtc/", params=_P({"products", "date"} | _OFFSET), pagination="offset"),
        EndpointSpec("ohlc", "GET", "/ohlc/", limit=OHLC_LIMIT,
                     params=_P({"instruments", "hg_instrument_ids", "interval", "count", "start", "end", "extraFields"})),
        EndpointSpec("options_generic_code", "GET", "/options/generic-code/",
                     params=_P({"options_gd_codes", "start_date", "end_date"} | _OFFSET), pagination="offset"),
        EndpointSpec("options_greeks", "GET", "/options/greeks/",
                     params=_P({"timeframe", "contracts", "cme_contracts", "startTime", "endTime", "atm_min", "atm_max",
                                "dte_min", "dte_max", "strike_min", "strike_max", "call_delta_value",
                                "call_delta_range", "put_delta_value", "put_delta_range", "straddle_delta_value",
                                "straddle_delta_range", "spike_removed"} | _OFFSET), pagination="offset"),
        EndpointSpec("options_instruments", "GET", "/options/instruments/",
                     params=_P({"options_qh_codes", "cme_code", "product", "options_type", "active_date", "dte"} | _OFFSET),
                     pagination="offset"),
        EndpointSpec("options_oi", "GET", "/options/oi/",
                     params=_P({"contracts", "cme_contracts", "date_from", "date_to", "strike_min", "strike_max",
                                "dte_min", "dte_max", "atm_min", "atm_max", "count"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_bot_historical", "GET", "/platts/bot/historical/",
                     params=_P({"order_time", "order_time__lt", "order_time__lte", "order_time__gt", "order_time__gte",
                                "market", "product"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_bot_historical_v2", "GET", "/platts/bot/historical/v2/",
                     params=_P({"order_time", "order_time__lt", "order_time__lte", "order_time__gt", "order_time__gte",
                                "market", "product"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_commentary", "GET", "/platts/commentary/",
                     params=_P({"date", "ric_code"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_ladder_historical", "GET", "/platts/ladder/historical/",
                     params=_P({"date", "date__lt", "date__lte", "date__gt", "date__gte", "market", "product"} | _OFFSET),
                     pagination="offset"),
        EndpointSpec("platts_ltp_historical", "GET", "/platts/ltp/historical/",
                     params=_P({"date", "date__lt", "date__lte", "date__gt", "date__gte", "market", "product"} | _OFFSET),
                     pagination="offset"),
        EndpointSpec("platts_ltp_historical_by_id", "GET", "/platts/ltp/historical/{id}/",
                     params=_P({"date", "date__lt", "date__lte", "date__gt", "date__gte", "market", "product"} | _OFFSET)),
        EndpointSpec("platts_market_data", "GET", "/platts/market-data/",
                     params=_P({"time__range", "product"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_market_product", "GET", "/platts/market-product/"),
        EndpointSpec("platts_moc_snapshot", "GET", "/platts/moc/snapshot/",
                     params=_P({"date", "date__range", "market", "product", "strip"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_moc_snapshot_v2", "GET", "/platts/moc/snapshot/v2/",
                     params=_P({"date", "date__range", "market", "product", "strip"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_nai_historical", "GET", "/platts/nai-historical/",
                     params=_P({"timestamp", "timestamp__range", "type"} | _OFFSET), pagination="offset"),
        EndpointSpec("platts_refinery_outage", "GET", "/platts/refinery-outage/",
                     params=_P({"time__range", "refinery_name", "planningstatus", "processunit_name"} | _OFFSET),
                     pagination="offset"),
        EndpointSpec("platts_summary_historical", "GET", "/platts/summary/historical/",
                     params=_P({"date", "date__lt", "date__lte", "date__gt", "date__gte", "market", "product"} | _OFFSET),
                     pagination="offset"),
        EndpointSpec("seasonality", "GET", "/seasonality/",
                     params=_P({"qhcodes", "multipliers", "field", "years", "datatype"} | _OFFSET), pagination="offset"),
        EndpointSpec("taq", "POST", "/taq/"),
        EndpointSpec("taq_products", "GET", "/taq/products/", limit=None),
        EndpointSpec("tas", "POST", "/tas/", limit=TAS_LIMIT),
        EndpointSpec("vap", "GET", "/vap/", params=_P({"instruments", "interval", "count", "end"})),
    ]
}


def list_endpoints() -> list[dict]:
    """Every wrapped endpoint with its method, path and limit (None limit = public)."""
    return [
        {"name": s.name, "method": s.method, "path": s.path,
         "per_minute": s.limit.per_minute if s.limit else None, "paginated": s.pagination}
        for s in ENDPOINTS.values()
    ]


def _check_max(label: str, items: Iterable | str, maximum: int, sep: str = ",") -> None:
    count = len(items.split(sep)) if isinstance(items, str) else len(list(items))
    if count > maximum:
        raise ValueError(f"{label}: at most {maximum} allowed, got {count}")


def _check_choice(label: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}, got {value!r}")


class QHApi:
    """Typed entry points for every QH API endpoint (see module docstring)."""

    def __init__(self, client: QHClient):
        self.client = client

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _get(self, name: str, _options: dict | None = None, **params: Any) -> Any:
        """`_options` are transport options (token, max_attempts), never query parameters."""
        spec = ENDPOINTS[name]
        if spec.params:
            unknown = set(params) - spec.params
            if unknown:
                raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}; allowed: {sorted(spec.params)}")
        return self.client.request(spec, params, **(_options or {}))

    def _get_all(self, name: str, page_size: int, max_pages: int, **params: Any) -> list:
        """Every page of a paginated endpoint (one request per page, capped by max_pages)."""
        spec = ENDPOINTS[name]
        unknown = set(params) - spec.params
        if unknown:
            raise ValueError(f"{name}: unknown parameter(s) {sorted(unknown)}")
        return self.client.request_all(spec, params, page_size=page_size, max_pages=max_pages)

    def _post(self, name: str, body: dict) -> Any:
        return self.client.request(ENDPOINTS[name], json_body=body)

    def get_all(self, name: str, page_size: int = 1000, max_pages: int = 10, **params: Any) -> list:
        """Fetch all pages of any paginated endpoint by registry name, e.g.
        get_all("platts_refinery_outage", refinery_name="X"). Each page costs one request."""
        return self._get_all(name, page_size, max_pages, **params)

    # ------------------------------------------------------------------
    # common
    # ------------------------------------------------------------------

    def daily_market_data_summary(self, **filters: Any) -> Any:
        """GET /common/dailymarketdata-summary/ - filters: product (comma list), start, end, count, limit, offset."""
        return self._get("daily_market_data_summary", **filters)

    def daily_market_data(self, **filters: Any) -> Any:
        """GET /common/dailymarketdata/ - filters: qhcode, gdcode, start, end, count, fields (e.g. oi,close,volume), limit, offset."""
        return self._get("daily_market_data", **filters)

    def economies(self) -> Any:
        """GET /common/economies/"""
        return self._get("economies")

    def economy_premiums(self, economies: str | list[str] = "*") -> Any:
        """GET /common/economies/premiums/ - economies required: comma list or '*' for all."""
        return self._get("economy_premiums", economies=economies)

    def fixings(self, **filters: Any) -> Any:
        """GET /common/fixings/ - filters: name, economy, start, end, page, page_size."""
        return self._get("fixings", **filters)

    def instruments(self, **filters: Any) -> Any:
        """GET /common/instruments/ (public, unlimited) - filters: qh_codes, tt_instrument_ids, limit, offset."""
        return self._get("instruments", **filters)

    # ------------------------------------------------------------------
    # curves / fairvalue
    # ------------------------------------------------------------------

    def curves(self, products: str | list[str] | None = None, strategies: str | list[str] | None = None,
               range: str | None = None, start: str | None = None, end: str | None = None) -> Any:
        """GET /curves/ - products (e.g. SRA,ER), strategies (e.g. 3MS,1YS), range (e.g. 1-10), start/end datetimes."""
        return self._get("curves", products=products, strategies=strategies, range=range, start=start, end=end)

    def fairvalue(self, products: str | list[str] = "*", token: str | None = None,
                  max_attempts: int | None = None) -> Any:
        """GET /fairvalue/ - products required: comma list of product IDs or '*' for all.

        The live price poll (adapters/live/fairvalue.py) spends ~6 of this endpoint's 7 req/min
        through this same method, so other callers queue behind it. `token` overrides the saved
        token for this call; `max_attempts` caps retries.
        """
        return self._get("fairvalue", {"token": token, "max_attempts": max_attempts}, products=products)

    def fairvalue_post(self, products: list[str]) -> Any:
        """POST /fairvalue/ - body {"products": [...]} (product IDs or '*')."""
        return self._post("fairvalue_post", {"products": list(products)})

    def fairvalue_historical(self, instruments: str | list[str], timeinterval: str | None = None,
                             fields: str | list[str] | None = None, count: int | None = None,
                             start: int | None = None, end: int | None = None) -> Any:
        """GET /fairvalue/historical/ - candles per QH code (max 50).

        timeinterval: 1m (default), 5m, 15m, 1h, 4h, 1d (minute/hour multiples up to 1440 min);
        fields: extra of open,high,low (product, time, close always returned); count default 50
        per instrument; start/end are unix seconds.
        """
        _check_max("instruments", instruments, MAX_INSTRUMENTS)
        if fields is not None:
            wanted = fields.split(",") if isinstance(fields, str) else list(fields)
            bad = set(wanted) - FAIRVALUE_HISTORICAL_FIELDS
            if bad:
                raise ValueError(f"fields must be among {sorted(FAIRVALUE_HISTORICAL_FIELDS)}, got {sorted(bad)}")
        return self._get("fairvalue_historical", instruments=instruments, timeinterval=timeinterval,
                         fields=fields, count=count, start=start, end=end)

    # ------------------------------------------------------------------
    # fundamental
    # ------------------------------------------------------------------

    def fundamental_codes(self, **filters: Any) -> Any:
        """GET /fundamental/codes/ (public) - filters: qhcode (partial match), asset, country, frequency
        (comma lists; allowed values are listed in the Swagger), page (1-indexed), page_size (max 5000, default 1000)."""
        if filters.get("page_size") and int(filters["page_size"]) > 5000:
            raise ValueError("page_size: at most 5000 allowed")
        return self._get("fundamental_codes", **filters)

    def fundamental_data(self, qhcodes: str | list[str], count: int | None = None,
                         start_date: str | list[str] | None = None, end_date: str | list[str] | None = None,
                         page: int | None = None, page_size: int | None = None) -> Any:
        """GET /fundamental/data/ - qhcodes: up to 10, SEMICOLON-separated on the wire (codes contain commas/spaces);
        start_date/end_date may be one date or one per code (semicolon-separated); page_size max 1000 (default 100)."""
        codes = qhcodes if isinstance(qhcodes, str) else ";".join(qhcodes)
        _check_max("qhcodes", codes, MAX_FUNDAMENTAL_QHCODES, sep=";")
        if page_size is not None and page_size > 1000:
            raise ValueError("page_size: at most 1000 allowed")
        def semi(v):
            return v if v is None or isinstance(v, str) else ";".join(v)
        return self._get("fundamental_data", qhcode=codes, count=count, start_date=semi(start_date),
                         end_date=semi(end_date), page=page, page_size=page_size)

    # ------------------------------------------------------------------
    # generic charts / gtc
    # ------------------------------------------------------------------

    def generic_chart_data(self, groups: list[dict], end_inclusive: int = 1) -> Any:
        """POST /generic-charts/generic/data/ - groups: list of {product (QH code), interval (5M/1H/1D...),
        resolve (0/1), months ('F,G,H,...'), gap (0/1 adjust on rollover), end (unix, intraday only),
        count (default 500)}; end_inclusive 1 = timestamp <= end."""
        if not groups:
            raise ValueError("groups must not be empty")
        return self._post("generic_chart_data", {"groups": groups, "end_inclusive": end_inclusive})

    def gtc(self, products: str | list[str] = "*", date: str | None = None, **page: Any) -> Any:
        """GET /gtc/ - products required (comma list or '*'), optional date, limit, offset."""
        return self._get("gtc", products=products, date=date, **page)

    # ------------------------------------------------------------------
    # ohlc / vap
    # ------------------------------------------------------------------

    def ohlc(self, interval: str, instruments: str | list[str] | None = None,
             hg_instrument_ids: str | list[str] | None = None, count: int | None = None,
             start: int | None = None, end: int | None = None,
             extra_fields: str | list[str] | None = None, token: str | None = None,
             max_attempts: int | None = None) -> Any:
        """GET /ohlc/ (30/min) - candles for up to 50 QH codes OR HG instrument ids (not both).

        interval: 1M/5M/1H/1D; count default 50 per instrument; start/end unix seconds;
        extra_fields: buyvolume, sellvolume. `token` / `max_attempts`: per-call transport options.
        """
        _check_choice("interval", interval, OHLC_INTERVALS)
        if bool(instruments) == bool(hg_instrument_ids):
            raise ValueError("provide either instruments or hg_instrument_ids, not both/neither")
        _check_max("instruments", instruments or hg_instrument_ids, MAX_INSTRUMENTS)
        if extra_fields is not None:
            wanted = extra_fields.split(",") if isinstance(extra_fields, str) else list(extra_fields)
            bad = set(wanted) - OHLC_EXTRA_FIELDS
            if bad:
                raise ValueError(f"extra_fields must be among {sorted(OHLC_EXTRA_FIELDS)}, got {sorted(bad)}")
        return self._get("ohlc", {"token": token, "max_attempts": max_attempts}, instruments=instruments, hg_instrument_ids=hg_instrument_ids, interval=interval,
                         count=count, start=start, end=end, extraFields=extra_fields)

    def vap(self, instruments: str | list[str], interval: str, count: int | None = None,
            end: int | None = None) -> Any:
        """GET /vap/ - volume-at-price candles for up to 50 QH codes; interval 5M or 1D; end unix seconds."""
        _check_choice("interval", interval, VAP_INTERVALS)
        _check_max("instruments", instruments, MAX_INSTRUMENTS)
        return self._get("vap", instruments=instruments, interval=interval, count=count, end=end)

    # ------------------------------------------------------------------
    # options
    # ------------------------------------------------------------------

    def options_generic_code(self, options_gd_codes: str | list[str], start_date: str, end_date: str | None = None,
                             **page: Any) -> Any:
        """GET /options/generic-code/ - options_gd_codes (e.g. ODESA1 daily/weekly, OMTY1 monthly/quarterly),
        start_date required (ISO e.g. 2025-10-16T09:30:00), end_date optional."""
        return self._get("options_generic_code", options_gd_codes=options_gd_codes, start_date=start_date,
                         end_date=end_date, **page)

    def options_greeks(self, **filters: Any) -> Any:
        """GET /options/greeks/ - filters: timeframe (1m,5m,15m,1h,2h,4h,1d; ES minute+hourly, TY/TU/ZB/FV hourly only),
        contracts, cme_contracts, startTime/endTime (YYYY-MM-DD HH:MM:SS), atm/dte/strike _min/_max,
        call/put/straddle _delta_value and _delta_range, spike_removed, limit, offset."""
        return self._get("options_greeks", **filters)

    def options_instruments(self, **filters: Any) -> Any:
        """GET /options/instruments/ - filters: options_qh_codes, cme_code, product (e.g. TY,ES),
        options_type (Monthly,Weekly), active_date, dte, limit, offset."""
        return self._get("options_instruments", **filters)

    def options_oi(self, **filters: Any) -> Any:
        """GET /options/oi/ - daily options open interest. filters: contracts / cme_contracts (max 25 each),
        date_from, date_to, strike/dte/atm _min/_max, count (max rows per contract), limit, offset."""
        for key in ("contracts", "cme_contracts"):
            if filters.get(key):
                _check_max(key, filters[key], MAX_OPTIONS_OI_CONTRACTS)
        return self._get("options_oi", **filters)

    # ------------------------------------------------------------------
    # platts
    # ------------------------------------------------------------------

    def platts_bot_historical(self, **filters: Any) -> Any:
        """GET /platts/bot/historical/ - filters: order_time (+ __lt/__lte/__gt/__gte), market, product, limit, offset."""
        return self._get("platts_bot_historical", **filters)

    def platts_bot_historical_v2(self, **filters: Any) -> Any:
        """GET /platts/bot/historical/v2/ (ClickHouse) - same filters as platts_bot_historical."""
        return self._get("platts_bot_historical_v2", **filters)

    def platts_commentary(self, **filters: Any) -> Any:
        """GET /platts/commentary/ - filters: date (YYYY-MM-DD), ric_code, limit, offset."""
        return self._get("platts_commentary", **filters)

    def platts_ladder_historical(self, **filters: Any) -> Any:
        """GET /platts/ladder/historical/ - filters: date (+ __lt/__lte/__gt/__gte), market, product, limit, offset."""
        return self._get("platts_ladder_historical", **filters)

    def platts_ltp_historical(self, **filters: Any) -> Any:
        """GET /platts/ltp/historical/ - filters: date (+ __lt/__lte/__gt/__gte), market, product, limit, offset."""
        return self._get("platts_ltp_historical", **filters)

    def platts_ltp_historical_by_id(self, record_id: str | int, **filters: Any) -> Any:
        """GET /platts/ltp/historical/{id}/ - same filters as platts_ltp_historical."""
        spec = ENDPOINTS["platts_ltp_historical_by_id"]
        unknown = set(filters) - spec.params
        if unknown:
            raise ValueError(f"platts_ltp_historical_by_id: unknown parameter(s) {sorted(unknown)}")
        return self.client.request(spec, filters, path_args={"id": record_id})

    def platts_market_data(self, **filters: Any) -> Any:
        """GET /platts/market-data/ - filters: time__range ('start,end' ISO), product, limit, offset."""
        return self._get("platts_market_data", **filters)

    def platts_market_product(self) -> Any:
        """GET /platts/market-product/ - mapping of markets to their products."""
        return self._get("platts_market_product")

    def platts_moc_snapshot(self, **filters: Any) -> Any:
        """GET /platts/moc/snapshot/ - filters: date, date__range ('YYYY-MM-DD,YYYY-MM-DD'), market, product, strip, limit, offset."""
        return self._get("platts_moc_snapshot", **filters)

    def platts_moc_snapshot_v2(self, **filters: Any) -> Any:
        """GET /platts/moc/snapshot/v2/ (ClickHouse) - same filters as platts_moc_snapshot."""
        return self._get("platts_moc_snapshot_v2", **filters)

    def platts_nai_historical(self, **filters: Any) -> Any:
        """GET /platts/nai-historical/ - market news. filters: timestamp, timestamp__range ('start,end' ISO), type, limit, offset."""
        return self._get("platts_nai_historical", **filters)

    def platts_refinery_outage(self, **filters: Any) -> Any:
        """GET /platts/refinery-outage/ - filters: time__range ('start,end' ISO), refinery_name, planningstatus,
        processunit_name, limit, offset."""
        return self._get("platts_refinery_outage", **filters)

    def platts_summary_historical(self, **filters: Any) -> Any:
        """GET /platts/summary/historical/ - filters: date (+ __lt/__lte/__gt/__gte), market, product, limit, offset."""
        return self._get("platts_summary_historical", **filters)

    # ------------------------------------------------------------------
    # seasonality / taq / tas
    # ------------------------------------------------------------------

    def seasonality(self, **filters: Any) -> Any:
        """GET /seasonality/ - filters: qhcodes, multipliers, field (close/oi/volume), years,
        datatype (settlement default, or close), limit, offset."""
        return self._get("seasonality", **filters)

    def taq(self, products: list[dict]) -> Any:
        """POST /taq/ - trade and quote data. products: [{"id": product, "dates": ["YYYY-MM-DD", ...],
        "start": "HH:MM:SS", "end": "HH:MM:SS"}]. Trade side codes (U,B,S,M,P,T,E,D,X,C,F,G) and
        trade type codes (R,O,L,B,I,S) are documented in the Swagger."""
        for product in products:
            if not product.get("id") or not product.get("dates"):
                raise ValueError("each TAQ product needs 'id' and a non-empty 'dates' list")
        return self._post("taq", {"products": products})

    def taq_products(self) -> Any:
        """GET /taq/products/ (public) - available TAQ products."""
        return self._get("taq_products")

    def tas(self, products: list[dict]) -> Any:
        """POST /tas/ (10/min) - products: [{"id": product, "dates": ["YYYY-MM-DD", ...],
        "start": "HH:MM:SS", "end": "HH:MM:SS"}]."""
        for product in products:
            if not product.get("dates"):
                raise ValueError("each TAS product needs a non-empty 'dates' list")
        return self._post("tas", {"products": products})
