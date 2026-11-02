"""The Trigger study section of the Strategy Rollover page: layout, table builders, text blocks, chart and callbacks."""

import pytest
from dash import Dash
from dash.exceptions import PreventUpdate

from core.rollover import RolloverOptions
from core.rollover_triggers import DONE
from tests.test_rollover_triggers import GENERIC, box_store_with_brent, mixed_regime, planted_store, study_service
from ui.callbacks import rollover_callbacks, rollover_study_callbacks as cb
from ui.container import Container
from ui.layouts import rollover_study as view
from ui.layouts import rollover_tab, seasonality_tab

# the page's controls, in the order the callbacks take them: product, curve, generic, expiries, days before
PAGE = ("CL", "outright", 2, 1, 20)
WINDOW = (-10, 10, 10, 36)  # from, to, lookback, expiries studied


@pytest.fixture
def env(monkeypatch):
    fresh = Container(rollover_study=study_service(planted_store(mixed_regime)))
    monkeypatch.setattr(cb, "container", fresh)
    return fresh


@pytest.fixture
def study(env):
    return env.rollover_study.study("CL", GENERIC, -10, 10, 10, 36)


@pytest.fixture
def box_study():
    return study_service(box_store_with_brent()).study("CLBZ", RolloverOptions(), -10, 10, 10, 8)


# ----------------------------------------------------------------------
# Layout and wiring
# ----------------------------------------------------------------------


def test_the_study_is_on_the_rollover_page_below_the_chart_and_no_longer_on_the_seasonality_page():
    text = str(rollover_tab.rollover_layout())
    for component_id in [
        "rollover-study-from", "rollover-study-to", "rollover-study-lookback", "rollover-study-lines", "rollover-study-title",
        "rollover-study-verdict", "rollover-study-lines-table", "rollover-study-board", "rollover-study-trigger", "rollover-study-scatter",
        "rollover-study-values", "rollover-study-rule", "rollover-study-save-rule", "rollover-study-rolls", "rollover-study-notes",
        "rollover-scan-run", "rollover-scan-table", "rollover-forward-table", "rollover-forward-pick", "rollover-forward-delete",
    ]:
        assert component_id in text, component_id
    assert text.index("rollover-graph") < text.index("rollover-study-from")
    seasonality = str(seasonality_tab.seasonality_layout())
    assert "Trigger study" not in seasonality and "season-study" not in seasonality


def test_every_id_the_callbacks_use_is_in_the_layout():
    layout = str(rollover_tab.rollover_layout())
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = rollover_tab.rollover_layout()
    rollover_callbacks.register_rollover_callbacks(app)
    cb.register_rollover_study_callbacks(app)  # a duplicate output or a bad signature would raise here
    for key, spec in app.callback_map.items():
        outputs = key[2:-2].split("...") if key.startswith("..") else [key]
        used = [part.rsplit(".", 1)[0] for part in outputs] + [item["id"] for item in spec["inputs"] + spec["state"]]
        for component_id in used:
            assert component_id in layout, f"{component_id} (used by {key})"


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------


def test_the_expiry_table_lists_the_running_lines_first_then_the_finished_ones(study):
    rows = view.line_rows(study)
    assert rows[0]["label"].endswith("(now)") and rows[0]["exit"] == "not started"
    assert any(r["label"].endswith("(now)") and r["exit"] == "running" and r["move"] is not None for r in rows)
    done = next(r for r in rows if r["exit"] not in ("running", "not started"))
    assert done["entry"] < done["exit"] and done["move"] is not None and done["skipped"] is None  # a generic line has no switch days
    assert [c["id"] for c in view.line_columns(study)] == [c["id"] for c in view.LINE_COLUMNS]
    assert view.line_rows(None) == []


def test_a_clbz_table_shows_what_the_switches_would_have_added_and_the_cl_brent_split(box_study):
    assert [c["id"] for c in view.line_columns(box_study)][-3:] == ["cl", "brent", "residual"]
    row = next(r for r in view.line_rows(box_study) if not r["label"].endswith("(now)"))
    assert row["move"] == pytest.approx(0.0) and row["skipped"] == pytest.approx(0.2)  # plain difference 0.2, of which 0.2 was the switches
    assert row["cl"] is not None and row["brent"] is not None and row["residual"] is not None


def test_the_scoreboard_has_a_row_per_trigger_with_the_best_first(study):
    rows = view.board_rows(study)
    assert len(rows) == len(study.screens) and rows[0]["verdict"] in ("strong", "possible")
    assert rows[0]["p"] is not None and rows[0]["q"] >= rows[0]["p"] and rows[0]["stable"] == "yes"
    assert all(r["why"] and r["group"] for r in rows)
    constant = next(r for r in rows if r["verdict"] == "does not vary")
    assert constant["p"] is None and constant["stable"] == ""


def test_readings_are_laid_out_triggers_down_and_expiries_across(study):
    columns, rows = view.values_view(study)
    assert columns[0]["name"] == "Trigger" and len(columns) > 5
    assert rows[0]["trigger"] == "MOVE OVER THE WINDOW" and len(rows) == len(study.screens) + 1
    assert view.values_view(None) == ([], [])


def test_the_verdict_says_what_kind_of_window_it_is_and_where_the_current_line_stands(study):
    text = str(view.verdict_block(study))
    assert "MIXED REGIMES" in text and "earlier expiries" in text and "hypothesis" in text
    assert "line is running" in text or "has not reached the entry day" in text
    assert str(view.verdict_block(None)) == str(view.html.Div())


def test_the_rule_block_gives_the_rule_its_leave_one_out_score_and_the_current_call(study):
    text = str(view.rule_block(study))
    assert "Rule: above" in text and "leave-one-expiry-out" in text and "was right in" in text
    plain = study_service(planted_store(lambda index: 1.0)).study("CL", GENERIC, -10, 10, 10, 4)
    assert "No trigger passed the screening" in str(view.rule_block(plain))


def test_the_switch_block_explains_and_lists_the_days_that_were_left_out(box_study, study):
    text = str(view.roll_block(box_study))
    assert "first trading day after each CL and Brent expiry" in text and "Jump not counted" in text
    assert f"{len(box_study.skips)} switch days" in text and "CL" in text and "Jump if it had been counted" in text
    assert "No switch days: this line follows one contract" in str(view.roll_block(study))
    assert "never switches" in str(view.roll_block(None))


def test_the_notes_name_the_triggers_the_app_has_no_data_for(study):
    text = str(view.notes_block(study))
    assert "by chance" in text and "Freight" in text and "EIA" in text and "OVX" in text


def test_the_scatter_plots_each_expiry_with_the_current_reading_and_the_rule_cut(study):
    key = study.rule.spec.key
    figure = view.scatter_figure(study, key)
    assert len(figure.data[0].x) == len(study.finished) and figure.data[0].text[0]
    assert "rule cut" in [a.text for a in figure.layout.annotations] and "rank corr." in figure.layout.title.text
    assert "Pick a trigger" in view.scatter_figure(None, None).layout.annotations[0].text
    assert "not part of the study" in view.scatter_figure(study, "nope").layout.annotations[0].text


def test_a_trigger_with_too_few_readings_is_not_plotted(study):
    for obs in study.obs:
        obs.triggers["vol"] = None
    assert "Too few expiries" in view.scatter_figure(study, "vol").layout.annotations[0].text


def test_the_scan_table_and_its_honest_note():
    result = study_service(planted_store(lambda index: 1.0)).scan("CL", GENERIC, 10, 36)
    rows, note = view.scan_view(result)
    assert view.scan_view(None) == ([], "")
    assert note.count("windows scanned") == 1 and "random sign" in note
    assert rows and rows[0]["id"] == 0 and rows[0]["length"] == rows[0]["x_to"] - rows[0]["x_from"] and rows[0]["hit"].endswith("%")
    assert view.scan_view(study_service(planted_store(lambda index: 1.0)).scan("CL", GENERIC, 10, 2))[1].endswith("scanning needs 5.")


def test_forward_rows_and_the_delete_choices(env, study):
    rule = env.rollover_study.save_rule(study)
    table, options = view.forward_view(env.rollover_study.forward_test(), env.rollover_study.rules())
    assert table and table[0]["rule_label"] and options[0]["value"] == rule["id"] and "saved 2026-10-23" in options[0]["label"]


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------


def test_render_study_fills_every_part(env):
    out = cb.render_study(*PAGE, *WINDOW, None)
    title, verdict, columns, lines, board, options, value, value_columns, value_rows, rule, rolls, notes = out
    assert "CL" in title and "from -10 to 10" in title and verdict.children and lines and board and options
    assert value in [o["value"] for o in options] and value_rows and value_columns and rule.children and rolls.children and notes.children


def test_the_chosen_trigger_is_kept_when_the_window_changes_and_replaced_when_it_is_gone(env):
    assert cb.render_study(*PAGE, *WINDOW, "price_CL")[6] == "price_CL"
    replaced = cb.render_study(*PAGE, *WINDOW, "no_such_trigger")[6]
    assert replaced != "no_such_trigger" and replaced is not None


def test_render_study_with_a_bad_entry_says_so(env):
    assert cb.render_study("nothing", *PAGE[1:], *WINDOW, None)[0] == view.PLACEHOLDER
    assert "From must be earlier than To" in cb.render_study(*PAGE, 10, -10, 10, 36, None)[0]
    assert "whole number" in cb.render_study(*PAGE, "abc", 10, 10, 36, None)[0]
    assert "Expiries studied" in cb.render_study(*PAGE, -10, 10, 10, 0, None)[0]
    assert "Generic" in cb.render_study("CL", "outright", 99, 1, 20, *WINDOW, None)[0]


def test_render_study_survives_a_crash_inside_the_study(env, monkeypatch):
    monkeypatch.setattr(env.rollover_study, "study", lambda *a, **k: 1 / 0)
    assert cb.render_study(*PAGE, *WINDOW, None)[0] == "Could not build the study; see the log"


def test_the_scatter_callback_draws_the_chosen_trigger(env):
    figure = cb.render_scatter("price_CL", *PAGE, *WINDOW)
    assert len(figure.data[0].x) > 10
    assert "From must be earlier" in cb.render_scatter("price_CL", *PAGE, 10, -10, 10, 36).layout.annotations[0].text
    assert "Pick a trigger" in cb.render_scatter(None, *PAGE, *WINDOW).layout.annotations[0].text


def test_saving_a_rule_lists_it_and_deleting_removes_it(env):
    saved = cb.save_rule(1, *PAGE, *WINDOW)
    assert saved.startswith("Saved: tracking")
    table, options = cb.render_forward(0, saved, "")
    assert table and len(options) == 1
    rule_id = options[0]["value"]
    assert cb.delete_rule(1, rule_id) == (None, "Deleted")
    assert cb.render_forward(0, saved, "Deleted") == ([], [])
    assert cb.delete_rule(1, rule_id) == (None, "That rule is already gone")
    assert cb.delete_rule(1, None) == (None, "Choose a rule first")


def test_a_study_without_a_rule_cannot_be_saved_and_says_why(monkeypatch):
    monkeypatch.setattr(cb, "container", Container(rollover_study=study_service(planted_store(lambda index: 1.0))))
    assert "no rule to save" in cb.save_rule(1, *PAGE, -10, 10, 10, 4)


def test_the_scan_button_fills_the_table_and_a_clicked_row_sets_the_window(monkeypatch):
    monkeypatch.setattr(cb, "container", Container(rollover_study=study_service(planted_store(lambda index: 1.0))))
    rows, note = cb.run_scan(1, *PAGE, 10, 36)
    assert "windows scanned" in note and rows
    assert cb.apply_scan_row({"row": 0, "row_id": rows[0]["id"]}, rows) == (rows[0]["x_from"], rows[0]["x_to"])
    with pytest.raises(PreventUpdate):
        cb.apply_scan_row({"row": 0, "row_id": 999}, rows)
    with pytest.raises(PreventUpdate):
        cb.apply_scan_row(None, rows)
    assert "whole number" in cb.run_scan(1, *PAGE, "x", 36)[1]
    assert cb.run_scan(1, "nothing", *PAGE[1:], 10, 36)[1] == view.PLACEHOLDER


def test_the_callbacks_do_nothing_harmful_when_the_service_is_missing(monkeypatch):
    monkeypatch.setattr(cb, "container", Container())
    assert cb.render_forward(0, "", "") == ([], [])
    assert cb.run_scan(1, *PAGE, 10, 36)[0] == []
    assert cb.delete_rule(1, "x")[1] == "Choose a rule first"
    assert "not running" in cb.render_study(*PAGE, *WINDOW, None)[0]
