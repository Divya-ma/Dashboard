"""Excel export of the archive: one workbook with every trade on screen, its legs and the summary.

Sheets
  Trades    one row per closed structure with every archive column, real numbers and dates (so Excel can sort, filter and
            pivot them); a figure the journal does not have is left empty
  Legs      one row per leg of every trade (structure, contract, side, ratio, lots)
  Summary   the archive statistics of the exported trades and when the file was made

The rows are core.archive.build_archive_rows' (as kept by the Archive tab after its filters), so the file holds exactly the
trades the table shows.
"""

import io
from datetime import date, datetime

import pandas as pd

from core.archive import SNAPSHOT_FIELDS, summarize

_SNAPSHOT_LABELS = {
    "px_co": "CO price", "px_cl": "CL price", "px_go": "GO price", "move_day": "Day move", "move_min": "1-min move",
    "curve_cl": "CL curve", "curve_brn": "BRN curve", "curve_go": "GO curve", "atr": "ATR",
    "vol_ratio": "Volatility vs normal", "volume_ratio": "Volume vs normal", "spread_ticks": "Bid-ask (ticks)",
    "brent_wti": "Brent-WTI", "crack": "Gasoil crack", "kink_z": "Curve z-score",
}
# (row key, header)
TRADE_COLUMNS = [
    ("name", "Structure Name"), ("type", "Type"), ("legs_summary", "Legs"), ("entry_date", "Entry Date"), ("exit_date", "Exit Date"),
    ("days_held", "Days Held"), ("entry_price", "Entry Price (pts)"), ("exit_price", "Exit Price (pts)"), ("lots", "Lots"), ("open_lots", "Lots Still Open"),
    ("transaction_costs", "Transaction Cost ($)"), ("realized_pnl", "Realized PnL ($, net of TC)"), ("result", "Result"),
    ("rr", "Risk : Reward"), ("ideas_text", "Trade Ideas"), ("payup_entry", "Payups in Entry"), ("payup_exit", "Payups in Exit"),
    ("conviction", "Conviction"), ("exit_reason", "Exit Reason"), ("mistakes_text", "Mistakes"), ("risk_usd", "Risk at Stop ($)"),
    ("r_multiple", "Result in R"), ("mfe_i", "MFE Intraday ($)"), ("mae_i", "MAE Intraday ($)"), ("mfe_d", "MFE Daily ($)"), ("mae_d", "MAE Daily ($)"),
    *[(prefix + column, f"{label} {_SNAPSHOT_LABELS[column]}") for prefix, label in (("e_", "Entry"), ("x_", "Exit")) for column, _k, _s in SNAPSHOT_FIELDS],
    ("notes", "Notes"),
]
_MONEY = '"$"#,##0.00;[Red]-"$"#,##0.00'
_NUMBER_FORMATS = {"Transaction Cost ($)": _MONEY, "Realized PnL ($, net of TC)": _MONEY, "Risk : Reward": "0.00", "Entry Date": "yyyy-mm-dd", "Exit Date": "yyyy-mm-dd"}


def _date(value):
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _lots(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def trades_frame(rows: list[dict]) -> pd.DataFrame:
    """The Trades sheet as a frame: archive rows with Excel-friendly types."""
    records = []
    for row in rows:
        record = {}
        for key, header in TRADE_COLUMNS:
            value = row.get(key)
            if key in ("entry_date", "exit_date"):
                value = _date(value)
            elif key == "lots":
                value = _lots(value)
            record[header] = value
        records.append(record)
    return pd.DataFrame(records, columns=[header for _key, header in TRADE_COLUMNS])


def legs_frame(rows: list[dict]) -> pd.DataFrame:
    records = [
        {"Structure": row["name"], "Contract": leg["symbol"], "Side": "Buy" if leg["side"] == "buy" else "Sell", "Ratio": leg.get("ratio"), "Lots": leg["lots"]}
        for row in rows for leg in row.get("legs", [])
    ]
    return pd.DataFrame(records, columns=["Structure", "Contract", "Side", "Ratio", "Lots"])


def summary_frame(rows: list[dict], made_at: datetime | None = None) -> pd.DataFrame:
    stats = summarize(rows)
    made_at = made_at or datetime.now()
    items = [
        ("Trades", stats["total"]), ("Winners", stats["winners"]), ("Losers", stats["losers"]), ("Win rate (%)", round(stats["win_rate"], 2)),
        ("Total PnL ($, net of TC)", stats["total_pnl"]), ("Average PnL ($)", stats["avg_pnl"]), ("Best trade ($)", stats["best"]),
        ("Worst trade ($)", stats["worst"]), ("Exported", made_at.strftime("%Y-%m-%d %H:%M")),
    ]
    return pd.DataFrame(items, columns=["Item", "Value"])


def _tidy(sheet, frame: pd.DataFrame) -> None:
    from openpyxl.utils import get_column_letter

    sheet.freeze_panes = "A2"
    if len(frame.columns) and len(frame):
        sheet.auto_filter.ref = sheet.dimensions
    for index, header in enumerate(frame.columns, start=1):
        letter = get_column_letter(index)
        longest = max([len(str(header))] + [len(str(v)) for v in frame[header].head(300) if v is not None])
        sheet.column_dimensions[letter].width = min(max(longest + 2, 10), 60)
        number_format = _NUMBER_FORMATS.get(header)
        if number_format:
            for cell in sheet[letter][1:]:
                cell.number_format = number_format


def archive_to_excel(rows: list[dict], made_at: datetime | None = None) -> bytes:
    """The workbook (.xlsx bytes) for these archive rows."""
    trades, legs, summary = trades_frame(rows), legs_frame(rows), summary_frame(rows, made_at)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl", date_format="yyyy-mm-dd") as writer:
        for name, frame in (("Trades", trades), ("Legs", legs), ("Summary", summary)):
            frame.to_excel(writer, sheet_name=name, index=False)
            _tidy(writer.sheets[name], frame)
    return buffer.getvalue()
