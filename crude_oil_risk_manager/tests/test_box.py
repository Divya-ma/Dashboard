"""CL-BZ boxes (outbox WTCLCO, spreadbox CLBZZX): symbols, curve ladder, derived prices and a history built from the legs."""

from datetime import date

import pandas as pd
import pytest

from adapters.base import SymbolTranslator
from core.box import box_legs, is_box_symbol, outbox_components, price_with_components
from core.contract_specs import spec_for_symbol
from core.curve_calendar import DFLY, FAMILIES, FLY, OUTRIGHT, PRODUCTS, SPREAD, build_structures, live_symbols, product_of_symbol
from core.curve_history import CurveHistoryStore, generic_code
from core.derived_prices import components_needed, derivation_options, derive_price
from core.structure_utils import compose_structure_symbol, decompose_to_outrights, parse_symbol

TODAY = date(2026, 10, 29)


def test_internal_symbols_translate_to_the_exchange_codes_and_back():
    assert SymbolTranslator.internal_to_api("CLBZF27") == "WTCLCOF27"
    assert SymbolTranslator.internal_to_api("CLBZF27-G27") == "WTCLCOF27-G27"
    assert SymbolTranslator.internal_to_api("CLBZSF27-G27") == "CLBZZXF27-G27"
    assert SymbolTranslator.api_to_internal("WTCLCOF27") == "CLBZF27"
    assert SymbolTranslator.api_to_internal("CLBZZXF27-G27-H27") == "CLBZSF27-G27-H27"
    # the shorter product codes still win for their own symbols
    assert SymbolTranslator.api_to_internal("WTCLZ26") == "WBSZ26" and SymbolTranslator.api_to_internal("CLZ26") == "CLZ26"
    assert product_of_symbol("WTCLCOF27") == product_of_symbol("CLBZZXF27-G27") == "CLBZ"
    assert product_of_symbol("CLF27") == "CL" and product_of_symbol("COF27") == "BRN"


def test_outbox_same_month_and_cross_month():
    assert decompose_to_outrights("CLBZF27", 2) == {"CLF27": 2.0, "BZF27": -2.0}
    assert decompose_to_outrights("CLBZF27-G27", 1) == {"CLF27": 1.0, "BZG27": -1.0}  # CL Jan27 - BZ Feb27


def test_spreadbox_flybox_and_condor_box_are_the_cl_structure_minus_the_bz_structure():
    assert decompose_to_outrights("CLBZSF27-G27", 1) == {"CLF27": 1.0, "BZF27": -1.0, "CLG27": -1.0, "BZG27": 1.0}
    fly = decompose_to_outrights("CLBZSF27-G27-H27", 1)
    assert fly == {"CLF27": 1.0, "CLG27": -2.0, "CLH27": 1.0, "BZF27": -1.0, "BZG27": 2.0, "BZH27": -1.0}
    assert sum(decompose_to_outrights("CLBZSF27-G27-H27+J27", 1).values()) == 0
    with pytest.raises(ValueError):
        box_legs("CLBZS", ["F27"])
    with pytest.raises(ValueError):
        decompose_to_outrights("CLBZF27-G27", 1, [1, -1])  # a box has fixed ratios


def test_a_box_is_not_confused_with_a_plain_structure():
    assert is_box_symbol("CLBZF27") and is_box_symbol("CLBZSF27-G27")
    assert not is_box_symbol("CLF27") and not is_box_symbol("CLZ26-F27") and not is_box_symbol(None)
    assert parse_symbol("CLBZSF27-G27")[0]["product"] == "CLBZS"
    assert spec_for_symbol("CLBZSF27-G27").tick_value == 10.0 and spec_for_symbol("CLBZF27").multiplier == 1000.0


def test_two_outboxes_compose_into_a_spreadbox_not_a_cross_month_outbox():
    assert compose_structure_symbol(["CLBZF27", "CLBZG27"], [1, -1]) == "CLBZSF27-G27"
    assert compose_structure_symbol(["CLBZF27", "CLBZG27", "CLBZH27"], [1, -2, 1]) == "CLBZSF27-G27-H27"
    assert compose_structure_symbol(["CLF27", "CLG27"], [1, -1]) == "CLF27-G27"


def test_the_curve_ladder_uses_the_outbox_for_outrights_and_the_spreadbox_code_for_the_rest():
    ladder = build_structures("CLBZ", TODAY, 15)
    assert [len(ladder[f]) for f in FAMILIES] == [15, 14, 13, 12]
    first = {family: ladder[family][0] for family in FAMILIES}
    assert first[OUTRIGHT].symbol == "WTCLCOF27" and first[OUTRIGHT].generic_code == "WTCLCO1"
    assert first[SPREAD].symbol == "CLBZZXF27-G27" and first[SPREAD].generic_code == "CLBZZX1-2"
    assert first[FLY].symbol == "CLBZZXF27-G27-H27" and first[FLY].generic_code == "CLBZZX1-2-3"
    assert first[DFLY].components == ("CLBZZXF27-G27-H27", "CLBZZXG27-H27-J27") and first[DFLY].component_weights == (1, -1)
    assert generic_code("CLBZ", SPREAD, 3) == "CLBZZX3-4" and generic_code("CLBZ", OUTRIGHT, 1) == "WTCLCO1"
    assert generic_code("CL", SPREAD, 3) == "CL3-4"  # the plain products are unchanged


def test_the_box_front_month_is_the_one_brent_has_not_expired_yet():
    # BZ Dec26 expires on the last business day of October, before CL Dec26 (about 20 Nov), so Jan27 is the box's front month.
    assert PRODUCTS["CLBZ"].is_box
    assert build_structures("CLBZ", TODAY, 3)[OUTRIGHT][0].label == "Jan27"
    assert build_structures("CL", TODAY, 3)[OUTRIGHT][0].label == "Dec26"


def test_live_symbols_include_the_cl_and_bz_outrights_a_missing_box_is_built_from():
    symbols = live_symbols("CLBZ", TODAY, 15)
    assert "WTCLCOF27" in symbols and "CLBZZXF27-G27" in symbols and "CLF27" in symbols and "BZZF27" in symbols
    assert outbox_components("WTCLCOF27") == ("CLF27", "BZZF27") and outbox_components("CLBZZXF27-G27") is None


def test_price_with_components_prefers_the_boxes_own_price():
    assert price_with_components({"CLF27": 70.0, "BZZF27": 72.5}, "WTCLCOF27") == -2.5
    assert price_with_components({"WTCLCOF27": -2.0, "CLF27": 70.0, "BZZF27": 72.5}, "WTCLCOF27") == -2.0
    assert price_with_components({"CLF27": 70.0}, "WTCLCOF27") is None


def test_a_missing_box_price_is_built_from_the_pieces_the_user_described():
    assert derivation_options("CLBZF27") == [[("CLF27", 1.0), ("BZF27", -1.0)]]
    spreadbox = derivation_options("CLBZSF27-G27")
    assert spreadbox[0] == [("CLBZF27", 1.0), ("CLBZG27", -1.0)]  # spreadbox = outbox - next outbox
    assert spreadbox[1] == [("CLF27-G27", 1.0), ("BZF27-G27", -1.0)]  # CL spread - BZ spread
    assert derivation_options("CLBZSF27-G27-H27")[0] == [("CLBZSF27-G27", 1.0), ("CLBZSG27-H27", -1.0)]  # flybox = spreadbox - spreadbox
    assert "CLBZF27" in components_needed("CLBZSF27-G27")

    prices = {"CLBZF27": (-2.0, 1000.0), "CLBZG27": (-2.4, 1000.0)}
    value, when, how = derive_price("CLBZSF27-G27", prices)
    assert value == pytest.approx(0.4) and when == 1000.0 and "CLBZF27" in how
    only_legs = {"CLF27": (70.0, 1.0), "BZF27": (72.0, 1.0), "CLG27": (69.0, 1.0), "BZG27": (71.6, 1.0)}
    assert derive_price("CLBZSF27-G27", only_legs)[0] == pytest.approx(0.6)  # (70 - 72) - (69 - 71.6)


# ----------------------------------------------------------------------
# History built from the legs
# ----------------------------------------------------------------------


class LegsApi:
    """Answers CL and BZ generic series with the right contract on each day; the box codes themselves are an error."""

    def __init__(self, cl_front: int = 11, bz_front: int = 12):
        self.fronts = {"CL": cl_front, "BZZ": bz_front}  # month index (0-based from Jan 2026): CL front Dec26, BZ front Jan27
        self.calls: list[list[str]] = []

    @staticmethod
    def token(index: int) -> str:
        return "FGHJKMNQUVXZ"[index % 12] + f"{26 + index // 12:02d}"

    @staticmethod
    def price(leg: str, index: int) -> float:
        return (90.0 + 0.5 * index) if leg == "CL" else (92.0 + 0.4 * index)

    def generic_chart_data(self, groups):
        self.calls.append([g["product"] for g in groups])
        out = {}
        for group in groups:
            code = group["product"]
            leg = next((candidate for candidate in ("BZZ", "CL") if code.startswith(candidate) and not code.startswith("CLBZ")), None)
            if leg is None or "-" in code:
                out[f"{code}_1D"] = {"status": "SUCCESS", "df": [], "error": "no such generic code"}
                continue
            position = int(code[len(leg):])
            index = self.fronts[leg] + position - 1
            days = pd.bdate_range(end=pd.Timestamp(2026, 10, 29), periods=group["count"])
            out[f"{code}_1D"] = {"status": "SUCCESS", "df": [
                {"Date": int(day.timestamp() * 1000), "Close": self.price(leg, index), "Product": f"{leg}{self.token(index)}"} for day in days
            ]}
        return out


def test_history_is_built_from_the_legs_when_the_api_has_no_box_series(tmp_path):
    api = LegsApi()
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("CLBZ", years=1, today=TODAY)
    assert report.ok and report.series_updated == 15
    first = store.read_series("CLBZ", "WTCLCO1")
    # position 1 is Jan27 (the later of the two front months): CL Jan27 minus BZ Jan27, matched by month, not by position
    assert first["close"].iloc[-1] == pytest.approx(api.price("CL", 12) - api.price("BZZ", 12))
    assert (first["contract"] == "WTCLCOF27").all()
    spread = store.read_series("CLBZ", "CLBZZX2-3")
    assert (spread["contract"] == "CLBZZXG27-H27").all()
    assert spread["close"].iloc[-1] == pytest.approx((api.price("CL", 13) - api.price("BZZ", 13)) - (api.price("CL", 14) - api.price("BZZ", 14)))
    matrices = {family: store.matrix("CLBZ", family) for family in FAMILIES}
    assert all(m is not None for m in matrices.values())
    assert matrices[FLY].values[-1, 0] == pytest.approx(0.0, abs=1e-9)  # the box is linear in the month here, so its fly is flat
    assert int(matrices[SPREAD].months[-1, 0]) == 1  # the front spreadbox starts in January


def test_the_plain_products_never_use_the_legs_fallback(tmp_path):
    api = LegsApi()
    store = CurveHistoryStore(tmp_path, api)
    report = store.refresh("CL", years=1, today=TODAY)
    assert not report.ok  # the stand-in only knows outrights
    assert not any(code.startswith("BZZ") for call in api.calls for code in call)
