"""Diagnostics page: assess class separability before relying on a model.

Pure analysis tool — extracts CNN embeddings from a component's training
images, projects them to 2D, and visualizes/quantifies how well approved and
failed images separate. Trains and detects nothing; doesn't touch
pipeline/registry_factory/watcher; never writes anything back to the
component. See core/diagnostics/embeddings.py and
core/diagnostics/projection.py for the actual logic.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from emil_ml.config.logging_config import configure_logging
from emil_ml.config.registry import ComponentRegistry
from emil_ml.core.diagnostics import embeddings as diag_embeddings
from emil_ml.core.diagnostics import projection as diag_projection
from emil_ml.utils.paths import for_component

configure_logging()

st.set_page_config(page_title="EMIL Lab — Diagnostics", page_icon="🔬", layout="wide")
st.title("Diagnostics")
st.caption(
    "Visualize how separable approved and failed images are in whole-image CNN feature "
    "space, before you invest time training and tuning a real model. This is a diagnostic "
    "tool — it doesn't train or detect anything, and the projection below is not reused at "
    "inference time (UMAP especially isn't a stable transform to apply to one new image)."
)

registry = ComponentRegistry()
components = registry.list_all()

if not components:
    st.info("No components yet. Onboard one on the **Onboard** page first.")
    st.stop()

names = {c.display_name: c.name for c in components}
selected_display = st.selectbox("Component", list(names.keys()), key="diagnostics_component_select")
component = registry.get(names[selected_display])
paths = for_component(component.name)

col_method, col_n_neighbors, col_min_dist = st.columns(3)
with col_method:
    method_label = st.radio(
        "Projection method",
        ["UMAP (non-linear)", "PCA (linear, fast, deterministic)"],
        key="diagnostics_method",
        help=(
            "UMAP can reveal non-linear structure PCA can't, but its layout depends on the "
            "parameters below and can vary between runs. PCA is a fast, deterministic "
            "baseline — always worth comparing against, not just trusting UMAP alone."
        ),
    )
method = "umap" if method_label.startswith("UMAP") else "pca"

n_neighbors = diag_projection.DEFAULT_UMAP_N_NEIGHBORS
min_dist = diag_projection.DEFAULT_UMAP_MIN_DIST
if method == "umap":
    with col_n_neighbors:
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=2,
            max_value=50,
            value=diag_projection.DEFAULT_UMAP_N_NEIGHBORS,
            help="How many nearby points define 'local' structure. Lower = more local detail (noisier with few images); higher = more global structure.",
        )
    with col_min_dist:
        min_dist = st.slider(
            "min_dist",
            min_value=0.0,
            max_value=0.99,
            value=diag_projection.DEFAULT_UMAP_MIN_DIST,
            step=0.05,
            help="How tightly points are allowed to pack together. Lower values cluster more aggressively and can visually exaggerate separation.",
        )

if st.button("Run diagnostics", type="primary"):
    with st.spinner("Extracting CNN embeddings from training images..."):
        labeled = diag_embeddings.extract_embeddings(paths)

    if len(labeled.labels) < 2:
        st.warning(
            "Not enough training images to run diagnostics — need at least 2 (approved + "
            "failed combined). Upload training images for this component on the **Onboard** "
            "page first."
        )
        st.stop()

    if len(labeled.labels) < 10:
        st.caption(
            f"Only {len(labeled.labels)} images available — treat the numbers below as a "
            "rough signal, not a reliable measurement."
        )

    with st.spinner(f"Projecting to 2D with {method.upper()}..."):
        try:
            result = diag_projection.project(
                labeled.embeddings,
                labeled.labels,
                method=method,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
        except Exception as exc:
            st.error(f"Projection failed: {exc}")
            st.stop()

    df = pd.DataFrame(
        {
            "x": result.points[:, 0],
            "y": result.points[:, 1],
            "class": labeled.labels,
            "filename": labeled.filenames,
        }
    )

    chart = (
        alt.Chart(df)
        .mark_circle(size=90, opacity=0.75)
        .encode(
            x=alt.X("x:Q", title=None),
            y=alt.Y("y:Q", title=None),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(domain=["approved", "failed"], range=["#2ca02c", "#d62728"]),
                title="Class",
            ),
            tooltip=["filename:N", "class:N"],
        )
        .properties(height=450)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")

    interpretation = diag_projection.interpret(result)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Silhouette score", f"{result.silhouette:.3f}" if result.silhouette is not None else "n/a")
    col_b.metric("Mean within-class distance", f"{result.mean_intra_class_distance:.3f}")
    col_c.metric(
        "Mean between-class distance",
        f"{result.mean_inter_class_distance:.3f}" if result.mean_inter_class_distance is not None else "n/a",
    )

    st.markdown(f"**{interpretation.headline}**")
    st.write(interpretation.explanation)
    if interpretation.recommendation:
        st.info(interpretation.recommendation)

    with st.expander("How to read this"):
        st.markdown(
            "- **Clearly separated clouds** → the signal exists in whole-image features; a "
            "classifier should be able to learn it. If it still struggles in practice, the "
            "likely cause is training setup or data volume, not that the signal is missing.\n"
            "- **Mixed/overlapping clouds** → whole-image embeddings don't distinguish the "
            "classes, which usually means the defect is small and localized relative to the "
            "whole image. A localizing method (YOLO) or patch-based approach tends to do "
            "better here than whole-image classification.\n\n"
            "Silhouette score ranges roughly -1 to 1 (higher = better separated; near 0 = "
            "overlapping). The within/between-class distances are in the same rough spirit — "
            "between clearly bigger than within suggests separation. Both come from a "
            "general-purpose ImageNet backbone, not a model trained on this component, so "
            "treat them as a rough, honest signal — not a guarantee either way."
        )
