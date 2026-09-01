"""Console entry point: `emil-ml` launches the Streamlit app.

A thin wrapper around `streamlit run app/streamlit_app.py` so the app can be
started with a single installed command after `pip install -e .`.
"""

from __future__ import annotations

import sys

from emil_ml.config.settings import PROJECT_ROOT

APP_PATH = PROJECT_ROOT / "app" / "streamlit_app.py"


def main() -> None:
    from streamlit.web import cli as stcli

    if not APP_PATH.exists():
        raise FileNotFoundError(f"Streamlit entry point not found at {APP_PATH}")

    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
