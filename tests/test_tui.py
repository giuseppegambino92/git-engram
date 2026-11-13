import asyncio

import pytest

pytest.importorskip("textual")

from git_engram.tui.app import EngramApp  # noqa: E402


def screen_text(app) -> str:
    """Visible text of the current screen (parsed from Textual's SVG export)."""
    import html
    import re

    svg = app.export_screenshot()
    return html.unescape(" ".join(re.findall(r">([^<>]+)</text>", svg))).replace("\xa0", " ")


def test_dashboard_search_graph_and_graveyard(shopfront):
    async def scenario():
        app = EngramApp(shopfront.engine)
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.click("#search-input")
            await pilot.press(*"ERR_CART_STALE")
            await pilot.press("enter")
            await pilot.pause()
            results = app.query_one("#results")
            assert results.row_count > 0
            assert app.last_response.query_class == "exact"
            assert "ms" in str(app.query_one("#search-status").render())
            await pilot.press("down")

            await pilot.press("f2")
            await pilot.pause()
            assert app.query_one("TabbedContent").active == "graph"
            assert app.query_one("#search-input").value == "ERR_CART_STALE"  # key did not leak into the box
            hot = app.query_one("#hotspots")
            assert hot.row_count > 0
            tree = app.query_one("#coupling-tree")
            app._show_coupling("backend/api/schemas.py")
            labels = [str(n.label) for n in tree.root.children]
            assert any("mobile/src/parsers/orderParser.ts" in lbl and "hidden" in lbl for lbl in labels)

            await pilot.press("f3")
            await pilot.pause()
            assert app.query_one("TabbedContent").active == "graveyard"
            fails = app.query_one("#failures")
            assert fails.row_count == len(shopfront.failure_ids)
            app._show_failure(shopfront.failure_ids["F2-idempotency"])
            await pilot.pause()
            assert "attempted diff" in screen_text(app) and "uuid4" in screen_text(app)

            await pilot.press("f4")
            await pilot.pause()
            assert app.query_one("TabbedContent").active == "overview"
            assert "load-bearing hotspots" in screen_text(app)
            await pilot.press("ctrl+q")

    asyncio.run(scenario())
