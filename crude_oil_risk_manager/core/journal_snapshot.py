"""The market at the moment of a trade, for the trade journal (see core.journal for what is recorded).

One snapshot = three /ohlc/ requests (one-minute, hourly and daily candles of the front eight outright contracts of Brent,
WTI and gasoil) plus the stored ATR data. Everything is read from candles ("last traded"), never from /fairvalue/, whose
outright prices are not reliable. The API's daily candle of the last day or two is not there yet, so the day so far is built
from the hourly candles:

  prices   the last traded price at or before the trade of each product's FRONT contract: the last one-minute close, else the
           last hourly close (within three days), else the latest daily close
  moves    Bullish / Bearish / Stalemate for the front outright of the structure's own (first leg's) product: the day so far
           (hourly candles from 00:00 UTC to the trade, so a past trade is judged on what was known then; the daily candle if
           there are no hourly ones), and the last one-minute candle at or before the trade (blank when the minute history is gone)
  curves   for each product, the shape (core.journal.curve_shape) of its first six values of the structure's curve type, built
           from the outright prices (spread = M1 - M2, fly = M1 - 2 M2 + M3 ...), so a back month that did not trade still has a value
  atr      the average true range of the whole structure, per lot, in the structure's own points (structure_atr), with
           atr_basis: "quoted" (a contract's own ATR) or "estimated ±30%/65%" (k x the volatility of the legs' combined price)

The same code serves a trade entered now and an old one (backfill): for a past trade the daily candle of its date and whatever
one-minute candles the API still keeps are used. Nothing here raises: a part that cannot be worked out is None.
"""

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from adapters.base import SymbolTranslator
from core.atr_calc import atr_frame
from core.contract_specs import spec_for_product, spec_for_symbol
from core.curve_calendar import PRODUCTS, generic_contracts, product_of_symbol
from core.journal import (
    CURVE_POINTS, PRODUCT_LABELS, classify_move, curve_shape, curve_values, leg_family,
)
from core.models import Structure
from core.structure_utils import get_structure_type_from_symbol

logger = logging.getLogger(__name__)

CONTRACTS_READ = CURVE_POINTS + 2  # a fly curve of six points needs eight outrights
DAILY_LOOKBACK_DAYS = 10  # a contract that did not trade on the day takes its latest earlier close
HOURLY_LOOKBACK_HOURS = 240  # ten days: three for the last traded price, and the baseline of volatility and volume
RECENT_HOURS = 24  # "now" for volatility and volume: the last 24 hourly candles
MIN_RECENT, MIN_BASE = 12, 48  # hourly candles needed before a ratio is worked out
QUOTE_WINDOW_SECONDS = 120  # the bid-ask is read from the last quote within this long before the trade
LIVE_WITHIN_SECONDS = 600  # a spread and a Curve Kinks score exist only for a trade being entered or exited now
TAQ_MIN_SPARE = 3  # /taq/ requests that must be free (it is shared with Curve Compare) before the bid-ask is fetched
BARRELS_PER_TONNE = 7.45  # gasoil crack = gasoil $/tonne / 7.45 - Brent $/bbl
from zoneinfo import ZoneInfo

_LONDON = ZoneInfo("Europe/London")  # the TAQ feed's clock
MINUTE_WINDOW_MINUTES = 45
_STANDARD_RATIOS = {2: [-1, 1], 3: [-2, 1, 1]}


def _ts(moment: datetime) -> int:
    return int(moment.astimezone(timezone.utc).timestamp())


def _rows(payload) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("results") or []
    return [r for r in payload if isinstance(r, dict) and "product" in r] if isinstance(payload, list) else []


def candles_by_symbol(payload) -> dict[str, pd.DataFrame]:
    """/ohlc/ rows as {symbol: frame indexed by UTC time with open, high, low, close}, oldest first."""
    rows = _rows(payload)
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    frame["ts"] = pd.to_datetime(frame["time"], unit="ms", utc=True)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame = frame.dropna(subset=["close"]).drop_duplicates(["product", "ts"], keep="last")
    return {str(symbol): part.set_index("ts").sort_index()[["open", "high", "low", "close", "volume"]] for symbol, part in frame.groupby("product")}


def activity_ratios(bars: pd.DataFrame | None, at: pd.Timestamp) -> tuple[float | None, float | None]:
    """(volatility ratio, volume ratio) of the front contract at the trade: the last RECENT_HOURS hourly candles against the
    ten days before them. Volatility is the standard deviation of the hourly change in the close; volume is the volume per
    RECENT_HOURS candles. 1.0 is a normal market, 2.0 twice as fast (or heavy) as usual. None without enough candles."""
    if bars is None:
        return None, None
    bars = bars[bars.index <= at]
    recent, base = bars.iloc[-RECENT_HOURS:], bars.iloc[:-RECENT_HOURS]
    if len(recent) < MIN_RECENT or len(base) < MIN_BASE:
        return None, None
    vol_recent, vol_base = recent["close"].diff().dropna().std(), base["close"].diff().dropna().std()
    vol_ratio = float(vol_recent / vol_base) if vol_base and vol_base > 0 and vol_recent == vol_recent else None
    volume_base = base["volume"].sum() / len(base) * RECENT_HOURS if base["volume"].notna().any() else 0
    volume_ratio = float(recent["volume"].sum() / volume_base) if volume_base > 0 else None
    return (None if vol_ratio is None else round(vol_ratio, 2)), (None if volume_ratio is None else round(volume_ratio, 2))


def structure_products(structure: Structure) -> list[str]:
    """Curve products (CL, BRN, G) the structure's legs trade, first leg's first."""
    seen: list[str] = []
    for leg in structure.legs:
        product = product_of_symbol(_api(leg.contract.symbol))
        if product and product not in seen:
            seen.append(product)
    return seen


def _api(symbol: str) -> str:
    try:
        return SymbolTranslator.internal_to_api(symbol)
    except ValueError:
        return symbol


# ----------------------------------------------------------------------
# ATR of the whole structure
# ----------------------------------------------------------------------


def _atr_of(part: pd.DataFrame, period: int, skip_idle: bool, as_of: date) -> float | None:
    data = part[part["date"] <= pd.Timestamp(as_of)].set_index("date")[["open", "high", "low", "close", "volume"]]
    if len(data) < period + 2:
        return None
    atr = atr_frame(data, period, skip_idle, percent=False)["atr"].dropna()
    return float(atr.iloc[-1]) if len(atr) else None


SIGMA_WINDOW = 60  # daily changes whose standard deviation scales into an estimated ATR
MIN_SIGMA_DAYS = 30
# ATR / standard deviation of the daily settlement change, measured on the quoted contracts in the ATR store (Brent and WTI):
# about 1.55 for outrights and spreads, about 1.0 for flies; a structure built from legs has no true range of its own.
ATR_PER_SIGMA_SPREAD, ATR_PER_SIGMA_OTHER = 1.55, 1.0
ESTIMATE_ERROR_SPREAD, ESTIMATE_ERROR_OTHER = 0.3, 0.65  # typical error seen against quoted spreads (20%, widened) and flies


def structure_atr(structure: Structure, frame_of, period: int, as_of: date, skip_idle: bool = True) -> float | None:
    """Average true range of the whole structure per lot, in the structure's points; None if the stored data cannot give one.
    (structure_atr_with_basis says where the figure comes from.)"""
    return structure_atr_with_basis(structure, frame_of, period, as_of, skip_idle)[0]


def structure_atr_with_basis(structure: Structure, frame_of, period: int, as_of: date, skip_idle: bool = True) -> tuple[float | None, str | None]:
    """(ATR, basis): basis is "quoted" for a contract's own ATR and "estimated +/-30%" (or 65%) for a figure built from legs.

    `frame_of(product)` is the ATR store's frame (symbol, date, open, high, low, close, volume).
    1. The structure is one quoted contract (an outright, or a spread / fly of consecutive months): that contract's own ATR,
       exactly as the ATR page shows it.
    2. Otherwise it is ESTIMATED as k x sigma, where sigma is the standard deviation of the structure's daily change over the
       last SIGMA_WINDOW days (ratio x close of every leg, summed: the daily closes are synchronous, so the legs' correlation
       is already in it) and k = ATR / sigma measured on quoted contracts. Adding or averaging the legs' ATRs would be wrong
       (a range does not add; variance does) and so would the mean close-to-close move (that is not a true range).
       A fly or d-fly carries a wide error band (see ESTIMATE_ERROR_*): its weights make it very sensitive to the legs'
       correlation, and there is no quoted d-fly to calibrate on.
    """
    legs = structure.legs
    symbols = [_api(leg.contract.symbol) for leg in legs]
    frames: dict[str, pd.DataFrame] = {}

    def frame_for(api_symbol: str) -> pd.DataFrame | None:
        product = product_of_symbol(api_symbol)
        if product is None:
            return None
        if product not in frames:
            frames[product] = frame_of(product)
        part = frames[product]
        return part[part["symbol"] == api_symbol] if part is not None and not part.empty else None

    ratios = sorted(leg.ratio for leg in legs)
    try:
        if len(legs) == 1:
            part = frame_for(symbols[0])
            value = _atr_of(part, period, skip_idle, as_of) if part is not None and len(part) else None
            return (None, None) if value is None else (value * abs(legs[0].ratio), "quoted")
        if len(legs) in _STANDARD_RATIOS and ratios == _STANDARD_RATIOS[len(legs)]:
            composite = SymbolTranslator.structure_to_api_symbol(legs)
            part = frame_for(composite)
            if part is not None and len(part):
                value = _atr_of(part, period, skip_idle, as_of)
                if value is not None:
                    return value, "quoted"
    except (ValueError, NotImplementedError):
        pass
    closes = []
    for leg, symbol in zip(legs, symbols):
        part = frame_for(symbol)
        if part is None or part.empty:
            return None, None
        series = part[part["date"] <= pd.Timestamp(as_of)].set_index("date")["close"]
        closes.append(series * leg.ratio)
    price = pd.concat(closes, axis=1, join="inner").dropna().sum(axis=1).sort_index()
    moves = price.diff().dropna().iloc[-SIGMA_WINDOW:]
    if len(moves) < MIN_SIGMA_DAYS:
        return None, None
    sigma = float(moves.std())
    if not sigma == sigma:
        return None, None
    is_spread = len(legs) == 2
    k, error = (ATR_PER_SIGMA_SPREAD, ESTIMATE_ERROR_SPREAD) if is_spread else (ATR_PER_SIGMA_OTHER, ESTIMATE_ERROR_OTHER)
    return k * sigma, f"estimated ±{error:.0%}"


def structure_price_path(structure: Structure, loader) -> pd.Series | None:
    """The structure's daily price as it is quoted (sum of ratio x leg close; negated when the first leg's ratio is negative,
    as core.structure_view does), on the days every leg has a close. `loader` is the app's price source (settlements first).
    None when a leg has no history."""
    levels = []
    for leg in structure.legs:
        try:
            levels.append(loader.load_close_series(leg.contract.symbol, min_rows=2) * leg.ratio)
        except Exception:  # noqa: BLE001 - a leg without history: no path
            return None
    price = pd.concat(levels, axis=1, join="inner").dropna().sum(axis=1).sort_index()
    return price if structure.legs[0].ratio > 0 else -price


MINUTE_BARS_UP_TO = timedelta(hours=36)  # trades up to this long are marked on one-minute closes,
FIVE_MINUTE_BARS_UP_TO = timedelta(days=7)  # up to this long on five-minute closes, longer ones on hourly closes


def intraday_interval(duration: timedelta) -> str:
    return "1M" if duration <= MINUTE_BARS_UP_TO else "5M" if duration <= FIVE_MINUTE_BARS_UP_TO else "1H"


def intraday_price_path(structure: Structure, api, start: datetime, end: datetime) -> pd.Series | None:
    """The structure's price through [start, end] from the legs' intraday candle closes (ONE /ohlc/ request).

    The candle size follows the length of the trade (intraday_interval). A leg that did not trade in a candle keeps its last
    price, and the path starts once every leg has one. Quoted like structure_price_path (negated if the first ratio is negative).
    None if the API has nothing for a leg."""
    symbols = [_api(leg.contract.symbol) for leg in structure.legs]
    unique = list(dict.fromkeys(symbols))
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    frames = candles_by_symbol(api.ohlc(intraday_interval(end - start), instruments=unique, start=_ts(start - timedelta(hours=1)), end=_ts(end) + 60))
    levels = []
    for leg, symbol in zip(structure.legs, symbols):
        bars = frames.get(symbol)
        if bars is None or bars.empty:
            return None
        levels.append(bars["close"] * leg.ratio)
    price = pd.concat(levels, axis=1, join="outer").sort_index().ffill().dropna().sum(axis=1)
    return price if structure.legs[0].ratio > 0 else -price


# ----------------------------------------------------------------------
# The snapshot
# ----------------------------------------------------------------------


class MarketSnapshotter:
    """Builds the market snapshot of a structure at a moment. `api` is the shared QHApi; `atr_frame_of(product)` and
    `atr_period()` come from the ATR store and its saved parameters (both optional: without them the ATR is left blank)."""

    def __init__(self, api, atr_frame_of=None, atr_period=lambda: 14, curve_z=None, clock=lambda: datetime.now(timezone.utc)):
        self._api = api
        self._atr_frame_of = atr_frame_of
        self._atr_period = atr_period
        self._curve_z = curve_z  # curve_z(structure) -> the structure's live Curve Kinks score, or None
        self._clock = clock

    def _bid_ask_ticks(self, structure: Structure, when: datetime) -> float | None:
        """Bid-ask of the quoted structure in ticks at the trade, from /taq/ (one request); None for a structure that has no
        single quoted contract, when the quote feed has nothing for it, or when /taq/ has no spare requests."""
        legs = structure.legs
        ratios = sorted(leg.ratio for leg in legs)
        try:
            if len(legs) == 1:
                symbol = _api(legs[0].contract.symbol)
            elif len(legs) in _STANDARD_RATIOS and ratios == _STANDARD_RATIOS[len(legs)]:
                symbol = SymbolTranslator.structure_to_api_symbol(legs)
            else:
                return None
            if self._api.spare_requests("taq") < TAQ_MIN_SPARE:
                return None
            local = when.astimezone(_LONDON)
            start = local - timedelta(seconds=QUOTE_WINDOW_SECONDS)
            reply = self._api.taq([{"id": symbol, "dates": [local.date().isoformat()], "start": start.strftime("%H:%M:%S"), "end": local.strftime("%H:%M:%S")}])
            rows = reply.get("data", []) if isinstance(reply, dict) else []
            for row in reversed(rows):
                bid, ask = row.get("best_bid"), row.get("best_ask")
                if bid is not None and ask is not None and ask >= bid:
                    return round((ask - bid) / spec_for_symbol(legs[0].contract.symbol).tick_size, 2)
        except Exception as exc:  # noqa: BLE001 - the bid-ask is a nice-to-have
            logger.info("Journal: bid-ask unavailable (%s)", exc)
        return None

    def _candles(self, interval: str, symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        try:
            return candles_by_symbol(self._api.ohlc(interval, instruments=symbols, start=_ts(start), end=_ts(end)))
        except Exception as exc:  # noqa: BLE001 - a missing part must not stop the journal
            logger.warning("Journal: %s candles unavailable (%s)", interval, exc)
            return {}

    def snapshot(self, structure: Structure, when: datetime) -> dict:
        when = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
        day = when.date()
        products = list(PRODUCT_LABELS)  # BRN, CL, G
        contracts = {p: [c.symbol for c in generic_contracts(p, day, CONTRACTS_READ)] for p in products}
        symbols = [s for p in products for s in contracts[p]]
        minute = self._candles("1M", symbols, when - timedelta(minutes=MINUTE_WINDOW_MINUTES), when + timedelta(minutes=1))
        hourly = self._candles("1H", symbols, when - timedelta(hours=HOURLY_LOOKBACK_HOURS), when + timedelta(hours=1))
        daily = self._candles("1D", symbols, when - timedelta(days=DAILY_LOOKBACK_DAYS), when + timedelta(days=1))
        at = pd.Timestamp(when)

        def last_traded(symbol: str) -> float | None:
            for bars in (minute.get(symbol), hourly.get(symbol)):
                if bars is not None:
                    bars = bars[bars.index <= at]
                    if len(bars):
                        return float(bars["close"].iloc[-1])
            bars = daily.get(symbol)
            if bars is not None:
                bars = bars[bars.index.normalize() <= pd.Timestamp(day, tz="UTC")]
                if len(bars):
                    return float(bars["close"].iloc[-1])
            return None

        family = leg_family([leg.contract.symbol for leg in structure.legs], get_structure_type_from_symbol)
        prices, curves = {}, {}
        for product in products:
            outrights = [last_traded(s) for s in contracts[product]]
            prices[PRODUCT_LABELS[product]] = outrights[0]
            curves[product] = curve_shape(curve_values(outrights, family), min_tol=spec_for_product(product).tick_size)

        traded = structure_products(structure)
        # A CL-BZ box has no curve of its own in the journal; its day's move and ATR basis are read on CL, its first leg.
        own = next((p for p in traded if p in contracts), "CL" if "CLBZ" in traded else products[0])
        front = contracts[own][0]
        tick = spec_for_product(own).tick_size
        move_day = move_min = None
        so_far = hourly.get(front)
        if so_far is not None:
            so_far = so_far[(so_far.index.normalize() == pd.Timestamp(day, tz="UTC")) & (so_far.index <= at)]
        if so_far is not None and len(so_far):
            move_day = classify_move(so_far["open"].iloc[0], so_far["high"].max(), so_far["low"].min(), so_far["close"].iloc[-1], tick)
        elif daily.get(front) is not None:
            whole_day = daily[front][daily[front].index.normalize() == pd.Timestamp(day, tz="UTC")]
            if len(whole_day):
                bar = whole_day.iloc[-1]
                move_day = classify_move(bar["open"], bar["high"], bar["low"], bar["close"], tick)
        minute_bars = minute.get(front)
        if minute_bars is not None:
            minute_bars = minute_bars[minute_bars.index <= at]
            if len(minute_bars):
                bar = minute_bars.iloc[-1]
                move_min = classify_move(bar["open"], bar["high"], bar["low"], bar["close"], tick)

        atr = atr_basis = None
        if self._atr_frame_of is not None:
            try:
                atr, atr_basis = structure_atr_with_basis(structure, self._atr_frame_of, int(self._atr_period()), day)
            except Exception:  # noqa: BLE001
                logger.exception("Journal: ATR failed")
        vol_ratio, volume_ratio = activity_ratios(hourly.get(front), at)
        co, cl, go = prices.get("CO"), prices.get("CL"), prices.get("GO")
        live = abs((self._clock() - when).total_seconds()) <= LIVE_WITHIN_SECONDS
        spread_ticks = self._bid_ask_ticks(structure, when) if live else None
        kink = None
        if live and self._curve_z is not None:
            try:
                kink = self._curve_z(structure)
            except Exception:  # noqa: BLE001
                logger.exception("Journal: Curve Kinks score failed")
        return {
            "at": when.isoformat(), "product": own, "curve_family": family, "prices": prices,
            "move_day": move_day, "move_min": move_min, "curve": curves, "atr": atr, "atr_basis": atr_basis,
            "vol_ratio": vol_ratio, "volume_ratio": volume_ratio, "spread_ticks": spread_ticks,
            "brent_wti": None if co is None or cl is None else round(co - cl, 3),
            "crack": None if go is None or co is None else round(go / BARRELS_PER_TONNE - co, 3),
            "kink_z": None if kink is None else round(kink["z"], 2),
        }
