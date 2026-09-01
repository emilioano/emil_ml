"""EMIL Lab Streamlit entry point. Pure view layer — all logic lives in emil_ml."""

from __future__ import annotations

import streamlit as st

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry

configure_logging()

st.set_page_config(page_title="EMIL Lab", page_icon="🔍", layout="wide")

EMIL_LAB_LOGO = r"""
╔═════════════════════════════════════════════════════════════════════════╗
║                                                                	      ║
║   ███████╗   ███╗   ███╗   ██╗   ██╗         ██╗      █████╗ ██████╗    ║
║   ██╔════╝   ████╗ ████║   ██║   ██║         ██║     ██╔══██╗██╔══██╗   ║
║   █████╗     ██╔████╔██║   ██║   ██║         ██║     ███████║██████╔╝   ║
║   ██╔══╝     ██║╚██╔╝██║   ██║   ██║         ██║     ██╔══██║██╔══██╗   ║
║   ███████╗██╗██║ ╚═╝ ██║██╗██║██╗███████╗    ███████╗██║  ██║██████╔╝   ║
║   ╚══════╝╚═╝╚═╝     ╚═╝╚═╝╚═╝╚═╝╚══════╝    ╚══════╝╚═╝  ╚═╝╚═════╝    ║
║   Enhanced Machine Inspection & Learning Lab                            ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
"""

st.markdown(f"<pre>{EMIL_LAB_LOGO}</pre>", unsafe_allow_html=True)
st.caption("Modular industrial anomaly-detection / inspection tool")

registry = ComponentRegistry()
components = registry.list_all()

col1, col2, col3 = st.columns(3)
col1.metric("Components", len(components))
col2.metric("Ready", sum(1 for c in components if c.status == "ready"))
col3.metric("In training / created", sum(1 for c in components if c.status in ("created", "training")))

st.markdown(
    """
Use the sidebar to navigate:
- **Inspect** — run an input through a trained component's model
- **Onboard** — register a new component, upload training files, and train it
"""
)

if components:
    st.subheader("Components")
    st.dataframe(
        [
            {
                "name": c.name,
                "display_name": c.display_name,
                "status": c.status,
                "threshold": c.anomaly_threshold,
                "image_size": c.image_size,
            }
            for c in components
        ],
        width="stretch",
    )
