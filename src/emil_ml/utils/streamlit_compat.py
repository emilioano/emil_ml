"""Compatibility shims for third-party Streamlit components on newer Streamlit.

``streamlit-drawable-canvas`` (last released for Streamlit <1.24) calls the
removed ``streamlit.elements.image.image_to_url(image, width, clamp,
channels, output_format, image_id)`` helper. Modern Streamlit still has the
same functionality, just moved to ``streamlit.elements.lib.image_utils`` with
a ``LayoutConfig`` object instead of a raw ``width`` int as the second
argument. This restores the old call signature so unmaintained components
built against it keep working.
"""

from __future__ import annotations


def patch_image_to_url() -> None:
    import streamlit.elements.image as st_image_module

    if hasattr(st_image_module, "image_to_url"):
        return  # already present (older/newer Streamlit that still has it)

    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        return _image_to_url(
            image, LayoutConfig(width=width), clamp, channels, output_format, image_id
        )

    st_image_module.image_to_url = image_to_url_compat
