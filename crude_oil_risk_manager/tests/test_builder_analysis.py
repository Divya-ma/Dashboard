"""New Structure builder analysis: the settlement source, the correlation maths and its +-0.35 flag, the VaR that follows the VaR
page's Monte Carlo, and the beta / position sizing that no longer comes out empty."""

import numpy as np
import pandas as pd
import pytest

from adapters.mock.mock_historical import MockHistoricalAdapter
from core import structure_builder as sb
from core.correlation import classify_correlation, correlate_structures, structure_return_series, structure_series
from core.data_loader import DataLoader
from core.models import Contract, Leg, Structure, StructureStatus, StructureType
from core.settlement_prices import PROXY_NOTE, SettlementPrices, settlement_code
from core.positions import candidate_dollar_weights, dollar_weights
from core.var import (
    DEFAULT_MC_STD,
    KEY_VAR_MC,
    VOL_FIXED,
    beta_vs_portfolio,
    load_mc_settings,
    portfolio_outright_pnl_series,
    save_mc_settings,
)
from db.repository import Repository
from ui.callbacks import structure_builder_callbacks as bc
from ui.container import Container


def utc_index(n, start="2025-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def walk(n, seed, start=75.0, vol=1.0):
    return start + np.cumsum(np.random.default_rng(seed).normal(0, vol, n))


class FakeSettlements:
    """Settlement closes by internal outright symbol (the interface of core.settlement_prices.SettlementPrices)."""

    def __init__(self, series: dict[str, pd.Series], proxied: set[str] = frozenset()):
        self.series, self.proxied = series, set(proxied)

    def closes(self, symbol):
        series = self.series.get(symbol)
        return (series, symbol in self.proxied) if series is not None else (None, False)


def contract(symbol, product="CL", multiplier=1000.0):
    return Contract(product=product, contract_month=12, contract_year=2026, symbol=symbol, multiplier=multiplier, tick_size=0.01, tick_value=10)


def open_structure(name, symbols, ratios, lots=1.0, direction="buy", status=StructureStatus.OPEN):
    legs = [Leg(contract=contract(s), ratio=r, lots=lots, entry_price=1.0, direction=direction) for s, r in zip(symbols, ratios)]
    return Structure(name=name, structure_type=StructureType.CUSTOM, products=["CL"], legs=legs, status=status)


# ----------------------------------------------------------------------
# The settlement source
# ----------------------------------------------------------------------


def test_settlement_codes_map_outrights_and_proxy_bz():
    assert settlement_code("CLZ26") == ("CL", "CLZ26", False)
    assert settlement_code("BRNM27") == ("BRN", "COM27", False)
    assert settlement_code("BZM27") == ("BRN", "COM27", True)  # BZ converges to ICE Brent of the same month
    for symbol in ("CLZ26-F27", "BRNM27-N27-Q27", "WBSZ26", "GF27", "", "XYZ"):
        assert settlement_code(symbol) is None


class FakeStore:
    def __init__(self, frames):
        self.frames, self.version = frames, "v1"

    def frame(self, product):
        return self.frames.get(product, pd.DataFrame(columns=["contract", "date", "oi", "close", "volume"]))


def oi_frame(contract_code, n=30, seed=1):
    dates = pd.bdate_range("2026-01-05", periods=n)
    return pd.DataFrame({"contract": contract_code, "date": dates, "oi": 1.0, "close": walk(n, seed), "volume": 1.0})


def test_settlement_prices_read_the_store_with_a_utc_index_and_follow_its_version():
    store = FakeStore({"BRN": oi_frame("COM27"), "CL": oi_frame("CLZ26", seed=2)})
    prices = SettlementPrices(store)
    series, proxied = prices.closes("BRNM27")
    assert len(series) == 30 and not proxied and series.index.tz is not None and str(series.index.tz) == "UTC"
    assert prices.closes("BZM27")[1] is True and prices.closes("BZM27")[0].equals(series.rename("BZM27"))
    assert prices.closes("CLZ26")[0] is not None
    assert prices.closes("BRNN28") == (None, False) and prices.closes("CLZ26-F27") == (None, False) and prices.closes("WBSZ26") == (None, False)
    store.frames["BRN"] = oi_frame("COM27", n=40)
    assert len(prices.closes("BRNM27")[0]) == 30  # cached until the stored data changes
    store.version = "v2"
    assert len(prices.closes("BRNM27")[0]) == 40


# ----------------------------------------------------------------------
# The correlation maths
# ----------------------------------------------------------------------


def test_a_structures_price_is_combined_before_it_is_differenced():
    index = utc_index(40)
    a, b = pd.Series(walk(40, 1), index=index), pd.Series(walk(40, 2), index=index)
    b_gap = b.drop(index[20])  # leg B has no settlement on day 20
    result = structure_series({"CLZ26": 1.0, "CLF27": -1.0}, object(), FakeSettlements({"CLZ26": a, "CLF27": b_gap})).series
    price = (a - b).drop(index[20])
    assert result.to_numpy() == pytest.approx(price.diff().dropna().to_numpy())  # across the gap both legs move over the same two days
    assert index[20] not in result.index and len(result) == 38


def test_every_leg_is_needed_and_the_missing_ones_are_named(tmp_path):
    index = utc_index(60)
    settlements = FakeSettlements({"CLZ26": pd.Series(walk(60, 1), index=index)})
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())
    built = structure_series({"CLZ26": 1.0, "CLF27": -3.0, "CLG27": 1.0}, loader, settlements)
    assert built.series is None and sorted(built.missing) == ["CLF27", "CLG27"]
    entry = correlate_structures({"CLZ26": 1.0, "CLF27": -3.0}, {"CLZ26": 1.0}, 5, 20, loader, settlements)
    assert entry["correlation"] is None and "CLF27" in entry["error"] and entry["classification"] == "insufficient_data"


def test_settlements_are_used_when_the_loader_has_nothing_and_a_proxy_is_noted(tmp_path):
    index = utc_index(80)
    base = walk(80, 3)
    settlements = FakeSettlements({"BRNM27": pd.Series(base, index=index), "BZM27": pd.Series(base, index=index)}, proxied={"BZM27"})
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())  # no price files at all
    entry = correlate_structures({"BRNM27": 1.0}, {"BZM27": 1.0}, 5, 20, loader, settlements)
    assert entry["correlation"] == pytest.approx(1.0) and entry["classification"] == "highly_correlated" and entry["note"] == PROXY_NOTE
    plain = correlate_structures({"BRNM27": 1.0}, {"BRNM27": 1.0}, 5, 20, loader, settlements)
    assert plain["note"] is None


def test_the_flag_threshold_is_adjustable_and_defaults_to_the_old_one():
    assert classify_correlation(0.5) == "uncorrelated" and classify_correlation(0.5, 0.35) == "highly_correlated"
    assert classify_correlation(-0.36, 0.35) == "negatively_correlated" and classify_correlation(-0.34, 0.35) == "uncorrelated"
    assert classify_correlation(0.35, 0.35) == "highly_correlated" and classify_correlation(None, 0.35) == "insufficient_data"
    assert sb.CORRELATION_FLAG_THRESHOLD == 0.35


def correlated_pair(rho, n=120, seed=5):
    rng = np.random.default_rng(seed)
    shared, own = rng.normal(0, 1, n), rng.normal(0, 1, n)
    first = 75 + np.cumsum(shared)
    second = 75 + np.cumsum(rho * shared + np.sqrt(1 - rho ** 2) * own)
    index = utc_index(n)
    return pd.Series(first, index=index), pd.Series(second, index=index)


def test_the_builder_flags_a_moderate_correlation_as_correlated_and_a_negative_one_as_a_hedge(tmp_path):
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())
    candidate, positive = correlated_pair(0.6)  # about +0.52 as measured: between the two cut-offs
    _, negative = correlated_pair(-0.7)  # about -0.43
    settlements = FakeSettlements({"CLZ26": candidate, "CLF27": positive, "CLG27": negative})
    portfolio = [open_structure("Same way", ["CLF27"], [1]), open_structure("Other way", ["CLG27"], [1])]
    rows = sb.check_portfolio_correlation([{"symbol": "CLZ26", "ratio": 1}], portfolio, 5, 20, loader, settlements)
    by_name = {r["existing_structure"]: r for r in rows}
    assert by_name["Same way"]["correlation"] > 0.35 and by_name["Same way"]["classification"] == "highly_correlated"
    assert by_name["Other way"]["correlation"] < -0.35 and by_name["Other way"]["classification"] == "negatively_correlated"
    # the same numbers under the old 0.7 cut-off were "New Exposure"
    old = sb.check_portfolio_correlation([{"symbol": "CLZ26", "ratio": 1}], portfolio, 5, 20, loader, settlements, threshold=0.7)
    assert {r["classification"] for r in old if r["existing_symbol"] != "PORTFOLIO"} == {"uncorrelated"}


# ----------------------------------------------------------------------
# The VaR that follows the VaR page
# ----------------------------------------------------------------------


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = Repository(str(tmp_path / "builder.db"))
    loader = DataLoader(str(tmp_path / "hist"), MockHistoricalAdapter())
    fresh = Container(repository=repo, data_loader=loader)
    monkeypatch.setattr(bc, "container", fresh)
    return fresh


# ----------------------------------------------------------------------
# Beta and position sizing
# ----------------------------------------------------------------------


def test_beta_has_data_from_settlements_even_when_the_legs_candles_are_sparse(tmp_path, create_synthetic_parquet):
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())
    create_synthetic_parquet(tmp_path, "CLZ26", n_days=3)  # the leg's own candles: three days
    index = utc_index(150)
    prices = pd.Series(walk(150, 4), index=index)
    settlements = FakeSettlements({"CLZ26": prices})
    book = [open_structure("long 2", ["CLZ26"], [1], lots=2.0)]
    candidate = structure_return_series({"CLZ26": 1000.0}, loader, settlements)  # 1 lot of the same outright
    sized = beta_vs_portfolio(candidate, 1.0, book, loader, 30, settlements)
    assert sized["error"] is None and sized["beta"] == pytest.approx(2.0) and sized["r_squared"] == pytest.approx(1.0)
    assert sized["lots_to_match"] == pytest.approx(2.0) and sized["lots_to_hedge"] == pytest.approx(-2.0) and sized["observations"] == 30
    # without settlements the sparse candles leave nothing to regress on (the old behaviour)
    old = beta_vs_portfolio(candidate, 1.0, book, loader, 30)
    assert old["error"] and "available" in old["error"] or "No portfolio" in old["error"]


def test_beta_says_why_when_there_is_nothing_to_size_against(tmp_path):
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())
    candidate = pd.Series(np.arange(60.0), index=utc_index(60))
    only_shells = [open_structure("shell", ["CLZ26"], [1], lots=0.0, status=StructureStatus.SHELL)]
    assert "no open traded positions" in beta_vs_portfolio(candidate, 1.0, only_shells, loader, 30, FakeSettlements({}))["error"]
    book = [open_structure("long", ["BRNZ26"], [1])]
    assert "No portfolio price data" in beta_vs_portfolio(candidate, 1.0, book, loader, 30, FakeSettlements({}))["error"]
    series, skipped, proxied = portfolio_outright_pnl_series(book, loader, FakeSettlements({}))
    assert series is None and skipped == {"BRNZ26": "no price history"} and proxied == []


def test_bz_legs_in_the_book_use_brent_and_the_panel_says_so(tmp_path):
    loader = DataLoader(str(tmp_path), MockHistoricalAdapter())
    index = utc_index(120)
    brent = pd.Series(walk(120, 6), index=index)
    settlements = FakeSettlements({"BRNZ26": brent, "BZZ26": brent}, proxied={"BZZ26"})
    bz = Structure(name="bz", structure_type=StructureType.OUTRIGHT, products=["BZ"], status=StructureStatus.OPEN,
                   legs=[Leg(contract=contract("BZZ26", product="BZ"), ratio=1, lots=1.0, entry_price=1.0)])
    candidate = structure_return_series({"BRNZ26": 1000.0}, loader, settlements)
    sized = beta_vs_portfolio(candidate, 1.0, [bz], loader, 30, settlements)
    assert sized["proxied"] == ["BZZ26"] and sized["beta"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# The button end to end
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# The VaR of the candidate: both methods of the VaR page
# ----------------------------------------------------------------------


def test_candidate_weights_view_the_book_as_the_var_page_does():
    saved = {"CLZ26": contract("CLZ26", multiplier=500.0)}
    legs = [{"symbol": "clz26", "ratio": 1}, {"symbol": "CLF27", "ratio": -2}, {"symbol": "CLG27", "ratio": 1}, {"symbol": "CLG27", "ratio": -1}, {"symbol": "", "ratio": 1}]
    weights = candidate_dollar_weights(legs, saved.get, default_multiplier=250.0)
    assert weights == {"CLZ26": 500.0, "CLF27": -500.0}  # the saved multiplier wins, then the typed one; legs that cancel or are blank drop out
    assert candidate_dollar_weights([{"symbol": "CLF27", "ratio": 1}], lambda s: None) == {"CLF27": 1000.0}
    fly = candidate_dollar_weights([{"symbol": "CLZ26-F27-G27", "ratio": 1}], lambda s: None, decompose=True)
    assert fly == {"CLZ26": 1000.0, "CLF27": -2000.0, "CLG27": 1000.0}  # the same lot split into outrights


def test_the_book_is_built_from_outright_dollar_weights():
    book = [open_structure("fly", ["CLZ26-F27-G27"], [1], lots=2.0), open_structure("short", ["CLZ26"], [1], lots=1.0, direction="sell"),
            open_structure("shell", ["CLH27"], [1], lots=0.0, status=StructureStatus.SHELL)]
    assert dollar_weights(book, decompose=True) == {"CLZ26": 1000.0, "CLF27": -4000.0, "CLG27": 2000.0}  # 2 lots of the fly less 1 lot sold; the shell has no lots


def test_the_monte_carlo_controls_are_shared_with_the_var_page(tmp_path):
    repo = Repository(str(tmp_path / "x.db"))
    default = load_mc_settings(repo)
    assert (default.distribution, default.dist_params, default.n_simulations, default.vol_mode, default.lookback_days) == ("normal", {"mean": 0.0, "std": DEFAULT_MC_STD}, 10_000, "history", 60)
    assert load_mc_settings(None).vol_mode == "history"
    save_mc_settings(repo, "normal", 0.0, 0.2, -1.0, 1.0, 5000, VOL_FIXED, 90)
    saved = load_mc_settings(repo)
    assert (saved.dist_params, saved.n_simulations, saved.vol_mode, saved.lookback_days) == ({"mean": 0.0, "std": 0.2}, 5000, "fixed", 90)
    save_mc_settings(repo, "uniform", 0.0, 0.2, -0.3, 0.4, 2000)
    assert load_mc_settings(repo).dist_params == {"low": -0.3, "high": 0.4}
    repo.set_setting(KEY_VAR_MC, {"distribution": "weird", "std": "abc", "n_simulations": 5, "vol_mode": "x", "lookback_days": 1})
    bad = load_mc_settings(repo)
    assert (bad.distribution, bad.dist_params, bad.n_simulations, bad.vol_mode, bad.lookback_days) == ("normal", {"mean": 0.0, "std": 0.5}, 100, "history", 10)  # bad values fall back, counts are clamped


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = Repository(str(tmp_path / "builder.db"))
    loader = DataLoader(str(tmp_path / "hist"), MockHistoricalAdapter())
    fresh = Container(repository=repo, data_loader=loader)
    monkeypatch.setattr(bc, "container", fresh)
    return fresh


def write_leg(create_synthetic_parquet, tmp_path, symbol, closes):
    create_synthetic_parquet(tmp_path / "hist", symbol, closes=closes)


def test_both_methods_agree_when_the_legs_move_like_a_normal_distribution(env, create_synthetic_parquet, tmp_path):
    rng = np.random.default_rng(3)
    write_leg(create_synthetic_parquet, tmp_path, "CLZ26", 75 + np.cumsum(rng.normal(0, 0.12, 300)))  # about 0.12 points a day
    save_mc_settings(env.repository, "normal", 0.0, 0.5, -1.0, 1.0, 20_000, "history", 250)  # a long window: both methods see 250 days
    var = bc._candidate_var({"CLZ26": 1000.0}, 5, 20)
    history, mc = var["historical"], var["mc"]
    assert history["observations"] == 250 and history["var_95"] > 0
    assert mc["vol_mode"] == "history" and mc["sigma"]["CLZ26"] == pytest.approx(0.12, rel=0.15)
    assert mc["var_95"] == pytest.approx(history["var_95"], rel=0.25)  # the two methods agree up to sampling noise
    assert mc["var_95"] == pytest.approx(1.645 * 1000 * mc["sigma"]["CLZ26"], rel=0.06)  # one leg: 1.645 sigma


def test_the_monte_carlo_no_longer_shakes_every_leg_by_half_a_point(env, create_synthetic_parquet, tmp_path):
    rng = np.random.default_rng(4)
    write_leg(create_synthetic_parquet, tmp_path, "CLZ26-F27-G27", 0.5 + np.cumsum(rng.normal(0, 0.03, 300)))  # a fly moves about 3 ticks a day
    history_mode = bc._candidate_var({"CLZ26-F27-G27": 1000.0}, 5, 20)["mc"]["var_95"]
    save_mc_settings(env.repository, "normal", 0.0, 0.5, -1.0, 1.0, 10_000, VOL_FIXED, 60)
    fixed = bc._candidate_var({"CLZ26-F27-G27": 1000.0}, 5, 20)["mc"]
    assert fixed["vol_mode"] == "fixed" and fixed["var_95"] == pytest.approx(1.645 * 500, rel=0.06)  # the old behaviour: $500 a day per lot
    assert history_mode < fixed["var_95"] / 10  # a fly's own size gives a VaR an order of magnitude lower


def test_the_candidate_var_follows_whatever_the_var_page_was_set_to(env):
    save_mc_settings(env.repository, "normal", 0.0, 0.5, -1.0, 1.0, 10_000, VOL_FIXED, 60)
    base = bc._candidate_var({"CLZ26": 1000.0}, 5, 20)["mc"]["var_95"]
    save_mc_settings(env.repository, "normal", 0.0, 0.1, -1.0, 1.0, 10_000, VOL_FIXED, 60)
    assert bc._candidate_var({"CLZ26": 1000.0}, 5, 20)["mc"]["var_95"] == pytest.approx(base / 5, rel=0.12)
    save_mc_settings(env.repository, "uniform", 0.0, 0.5, -0.2, 0.2, 10_000, VOL_FIXED, 60)
    uniform = bc._candidate_var({"CLZ26": 1000.0}, 5, 20)["mc"]
    assert uniform["distribution"] == "uniform" and uniform["var_95"] == pytest.approx(0.9 * 200, rel=0.1)  # 5th percentile of U(-0.2, 0.2) x $1,000


def test_a_spread_hedges_itself_in_the_var(env, create_synthetic_parquet, tmp_path):
    closes = walk(120, 8)
    for symbol, noise in (("CLZ26", 0.0), ("CLF27", 0.05)):  # two legs that move together
        write_leg(create_synthetic_parquet, tmp_path, symbol, closes + np.random.default_rng(1).normal(0, noise, 120))
    together = bc._candidate_var({"CLZ26": 1000.0, "CLF27": -1000.0}, 5, 20)
    alone = bc._candidate_var({"CLZ26": 1000.0}, 5, 20)
    assert together["mc"]["var_95"] < 0.5 * alone["mc"]["var_95"] and together["historical"]["var_95"] < 0.5 * alone["historical"]["var_95"]


def test_a_leg_without_history_is_reported_not_hidden(env):
    var = bc._candidate_var({"CLZ26": 1000.0, "CLF27": -1000.0}, 5, 20)
    assert "No price data" in var["historical"]["error"]
    assert any("fixed size" in w for w in var["mc"]["warnings"]) and any("uncorrelated" in w for w in var["mc"]["warnings"])


# ----------------------------------------------------------------------
# The button end to end
# ----------------------------------------------------------------------


def test_refresh_correlation_fills_all_three_panels(env, create_synthetic_parquet, tmp_path):
    index = utc_index(150)
    settlements = FakeSettlements({s: pd.Series(walk(150, seed), index=index) for s, seed in (("CLZ26", 1), ("CLF27", 2), ("CLG27", 3))})
    env.settlements = settlements
    for symbol in ("CLZ26-F27", "CLF27-G27"):  # the quoted legs' own candles, for the Monte Carlo and historical VaR
        create_synthetic_parquet(tmp_path / "hist", symbol, n_days=150, seed=len(symbol))
    existing = open_structure("Open spread", ["CLZ26-F27"], [1], lots=3.0)
    env.repository.save_contract(contract("CLZ26-F27"))
    env.repository.save_structure(existing)
    table, rows, var_panel, beta_panel = bc.check_correlation(1, ["CLF27-G27"], [1], "outright", 1000)
    assert [r["existing_structure"] for r in rows][0] == "Open spread" and rows[0]["correlation"] is not None
    assert "Correlated when the correlation is +0.35 or higher" in str(table)
    text = str(var_panel)
    assert "Historical simulation" in text and "Monte Carlo" in text and "VaR 95% (1 lot)" in text and "Same methods and settings as the VaR & Scenarios page" in text
    assert "Lots to Match" in str(beta_panel) and "Lots to Hedge" in str(beta_panel) and "daily settlements" in str(beta_panel)
