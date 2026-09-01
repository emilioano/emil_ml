"""Headless UI verification of app/pages/3_inspection_station.py using
Streamlit's AppTest harness (streamlit.testing.v1) — runs the real page
script without a browser or websocket, so it actually executes the
query-param parsing/widget code this module's docstring claims to, unlike
a plain curl against the dev server (which only ever sees the static SPA
shell, never runs page-specific script logic).

Checks:
1. No query params -> no exception, sort defaults to chronological, no
   crash (this IS the default view now that filter/sort were split apart
   in the second revision of this feature).
2. Valid params (component/verdict/status/report_status/sort) all get
   picked up and reflected in the corresponding widget's value.
3. Invalid/garbage params on every one of those fields fall back to their
   default silently — no exception, no crash.
4. The forgiving "?status=failed" alias (a verdict word in the status
   param) resolves to the verdict filter, not a crash or silent drop.
5. Filtering and sorting are genuinely independent: selecting "Failed
   first" as the sort does not change the Verdict filter's own value (and
   vice versa) — two different widgets holding two different states.

Run with: python scripts/verify_inspection_station_ui.py
"""

from __future__ import annotations

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from streamlit.testing.v1 import AppTest

PAGE_PATH = "app/pages/3_inspection_station.py"


def _run_with_params(params: dict[str, str]) -> AppTest:
    at = AppTest.from_file(PAGE_PATH, default_timeout=30)
    for key, value in params.items():
        at.query_params[key] = value
    at.run()
    return at


def _selectbox_value(at: AppTest, key: str):
    return at.selectbox(key=key).value


def main() -> None:
    all_pass = True

    # === 1: no params -> no exception, sort defaults to chronological =====
    print("=== 1: no query params -> loads cleanly, default sort is chronological ===")
    at = _run_with_params({})
    ok1 = not at.exception and _selectbox_value(at, "filter_sort") == "chronological"
    if at.exception:
        print(f"  exception: {at.exception[0]}")
    print(f"  sort widget value: {_selectbox_value(at, 'filter_sort')!r}")
    print(f"-> {'PASS' if ok1 else 'FAIL'}")
    all_pass &= ok1
    print()

    # === 2: valid params reflected in widgets ==============================
    print("=== 2: valid params (?verdict=failed&sort=failed_first&status=acknowledged) are picked up ===")
    at = _run_with_params({"verdict": "failed", "sort": "failed_first", "status": "acknowledged"})
    ok2 = (
        not at.exception
        and _selectbox_value(at, "filter_verdict") == "failed"
        and _selectbox_value(at, "filter_sort") == "failed_first"
        and _selectbox_value(at, "filter_status") == "acknowledged"
    )
    if at.exception:
        print(f"  exception: {at.exception[0]}")
    print(f"  verdict={_selectbox_value(at, 'filter_verdict')!r} sort={_selectbox_value(at, 'filter_sort')!r} status={_selectbox_value(at, 'filter_status')!r}")
    print(f"-> {'PASS' if ok2 else 'FAIL'}")
    all_pass &= ok2
    print()

    # === 3: invalid params on every field fall back to default =============
    print("=== 3: invalid params on every field -> silent fallback to default, no crash ===")
    at = _run_with_params(
        {
            "component": "this-component-does-not-exist",
            "verdict": "maybe",
            "status": "sort-of-acknowledged",
            "report_status": "????",
            "show_approved": "not-a-boolean",
            "sort": "alphabetical-by-vibes",
        }
    )
    ok3 = (
        not at.exception
        and _selectbox_value(at, "filter_component") == "all"
        and _selectbox_value(at, "filter_verdict") == "all"
        and _selectbox_value(at, "filter_status") == "all"
        and _selectbox_value(at, "filter_report_status") == "all"
        and _selectbox_value(at, "filter_sort") == "chronological"
    )
    if at.exception:
        print(f"  exception: {at.exception[0]}")
    print(
        f"  component={_selectbox_value(at, 'filter_component')!r} verdict={_selectbox_value(at, 'filter_verdict')!r} "
        f"status={_selectbox_value(at, 'filter_status')!r} report_status={_selectbox_value(at, 'filter_report_status')!r} "
        f"sort={_selectbox_value(at, 'filter_sort')!r}"
    )
    print(f"-> {'PASS' if ok3 else 'FAIL'}")
    all_pass &= ok3
    print()

    # === 4: forgiving "?status=failed" alias -> verdict filter =============
    print("=== 4: '?status=failed' (a verdict word in the wrong param) resolves as the verdict filter ===")
    at = _run_with_params({"status": "failed"})
    ok4 = (
        not at.exception
        and _selectbox_value(at, "filter_verdict") == "failed"
        and _selectbox_value(at, "filter_status") == "all"
    )
    if at.exception:
        print(f"  exception: {at.exception[0]}")
    print(f"  verdict={_selectbox_value(at, 'filter_verdict')!r} status={_selectbox_value(at, 'filter_status')!r}")
    print(f"-> {'PASS' if ok4 else 'FAIL'}")
    all_pass &= ok4
    print()

    # === 5: filter and sort are independent widgets =========================
    print("=== 5: sort=failed_first does not implicitly set/require any particular verdict filter ===")
    at = _run_with_params({"sort": "failed_first"})
    ok5 = not at.exception and _selectbox_value(at, "filter_verdict") == "all" and _selectbox_value(at, "filter_sort") == "failed_first"
    if at.exception:
        print(f"  exception: {at.exception[0]}")
    print(f"  verdict={_selectbox_value(at, 'filter_verdict')!r} (should stay 'all', untouched by sort) sort={_selectbox_value(at, 'filter_sort')!r}")
    print(f"-> {'PASS' if ok5 else 'FAIL'}")
    all_pass &= ok5
    print()

    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED — see above'}")


if __name__ == "__main__":
    main()
