"""The dashboard as a whole: every page builds, every callback registers, and every component id a callback names really exists.

These tests never import ui.app (that opens the real database and starts the background jobs); they build the same pieces with a
temporary database: the page layouts through the router, the page frame (ui.layouts.root) and every callback module (ui.callbacks.registry).
"""

import re
from pathlib import Path

import dash
import pytest

from db.repository import Repository
from ui.callbacks import shell_callbacks as shell
from ui.callbacks.registry import REGISTRATIONS, register_all
from ui.container import container
from ui.layouts.root import build_root_layout
from ui.layouts.shell import NAV_ITEMS

UI = Path(__file__).resolve().parent.parent / "ui"


@pytest.fixture
def wired(tmp_db_path, monkeypatch):
    monkeypatch.setattr(container, "repository", Repository(str(tmp_db_path)))
    return container


def collect_ids(component, found: set) -> set:
    """Every string id in a component tree."""
    if component is None or isinstance(component, (str, int, float, bool)):
        return found
    if isinstance(component, (list, tuple)):
        for item in component:
            collect_ids(item, found)
        return found
    component_id = getattr(component, "id", None)
    if isinstance(component_id, str):
        found.add(component_id)
    for name in getattr(component, "_prop_names", []):
        value = getattr(component, name, None)
        if name != "id" and (hasattr(value, "_prop_names") or isinstance(value, (list, tuple))):
            collect_ids(value, found)
    return found


@pytest.mark.parametrize("path", sorted(shell._ROUTES))
def test_every_page_builds_from_a_fresh_database(wired, path):
    layout, banner = shell.render_page(path, True)
    assert layout is not None and banner == {"display": "none"}
    assert collect_ids(layout, set()), f"{path} built a page with no components"


def test_the_navigation_and_the_router_list_the_same_pages(wired):
    assert {path for path, _ in NAV_ITEMS} <= set(shell._ROUTES)
    unknown, _ = shell.render_page("/no-such-page", True)
    assert "404" in str(unknown.children)
    layout, _ = shell.render_page("/structures/some-id", True)  # a structure's own address is the structures page
    assert layout is not None


def test_every_callback_module_registers_on_one_app_without_a_duplicate_output(wired):
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_all(app)  # Dash itself refuses a duplicate output that is not marked allow_duplicate
    assert len(REGISTRATIONS) >= 19 and len(app.callback_map) > 150


def test_the_page_frame_holds_the_shared_stores_timers_and_the_login_link(wired):
    ids = collect_ids(build_root_layout(True, True), set())
    assert {"url", "store-live-prices", "interval-live-poll", "interval-pnl-refresh", "page-content", "sidebar-signout"} <= ids
    assert "sidebar-signout" not in collect_ids(build_root_layout(True), set())


def _used_ids(app) -> dict[str, str]:
    used = {}
    for key, spec in app.callback_map.items():
        for kind in ("inputs", "state"):
            for item in spec.get(kind, []):
                if isinstance(item["id"], str) and not item["id"].startswith("{"):  # "{...}" is a pattern-matching id
                    used.setdefault(item["id"], key)
        outputs = spec["output"] if isinstance(spec["output"], list) else [spec["output"]]
        for output in outputs:
            if isinstance(output.component_id, str) and not output.component_id.startswith("{"):
                used.setdefault(output.component_id, key)
    return used


def _declared_in_source() -> str:
    """The text of every ui module that builds components (a component made inside a callback, like the structure-detail modal, is not
    in any page's first layout but its id is written out next to its construction)."""
    return "\n".join(path.read_text(encoding="utf-8") for path in UI.rglob("*.py") if "registry" not in path.name)


def _fstring_templates(source: str) -> list:
    """f"trade-{kind}-payups" -> a pattern matching any "trade-<word>-payups" (components made in a loop or by a helper)."""
    found = []
    for template in re.findall(r"""f["']([^"'\n]*\{[^"'\n]*)["']""", source):
        if " " in template or re.search(r"\{[^}]*:[^}]*\}", template):  # text or a number format, not an id
            continue
        pieces = re.split(r"\{[^}]*\}", template)
        words = "".join(pieces)
        if "-" not in words or len(words) < 6:  # an id template keeps some words of its own: "trade-{kind}-payups"
            continue
        found.append(re.compile(r"[\w-]+".join(re.escape(piece) for piece in pieces)))
    return found


def _unknown_ids(app) -> list[str]:
    present = collect_ids(build_root_layout(True, True), set())
    for path in shell._ROUTES:
        collect_ids(shell.render_page(path, True)[0], present)
    source = _declared_in_source()
    layouts = "\n".join(path.read_text(encoding="utf-8") for path in (UI / "layouts").rglob("*.py"))
    templates = _fstring_templates(source)
    missing = []
    for component_id, callback in sorted(_used_ids(app).items()):
        if component_id in present:
            continue
        declared = re.search(r"""\bid\s*=\s*f?["']%s["']|["']id["']\s*:\s*["']%s["']""" % (re.escape(component_id), re.escape(component_id)), source)
        written_in_a_layout = f'"{component_id}"' in layouts or f"'{component_id}'" in layouts  # e.g. _live_label(live, "trade-live-price-label")
        if not declared and not written_in_a_layout and not any(t.fullmatch(component_id) for t in templates):  # or built from an f-string such as f"trade-{kind}-payups"
            missing.append(f"{component_id}  (used by {callback[:70]})")
    return missing


def test_every_component_id_a_callback_names_exists_somewhere(wired):
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_all(app)
    assert _unknown_ids(app) == []


def test_a_callback_that_names_a_component_nobody_builds_is_caught(wired):
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    register_all(app)
    app.callback(dash.Output("settings-no-such-field", "children"), dash.Input("settings-save-rebates", "n_clicks"))(lambda n: n)
    unknown = _unknown_ids(app)
    assert len(unknown) == 1 and unknown[0].startswith("settings-no-such-field")
