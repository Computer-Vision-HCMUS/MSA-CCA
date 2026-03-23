"""
Geometry package for Canonical Correlation Analysis (CCA).

Exposes geometric descriptions, canonical angles, and plotting utilities.
"""

from .geometry import (
    get_geometry_description,
    get_canonical_angles_degrees,
    plot_correlation_angle,
    plot_geometry_schematic,
    plot_cca_variable_spaces_canonical,
    plot_first_pair_scatter_with_angle,
)

__all__ = [
    "get_geometry_description",
    "get_canonical_angles_degrees",
    "plot_correlation_angle",
    "plot_geometry_schematic",
    "plot_cca_variable_spaces_canonical",
    "plot_first_pair_scatter_with_angle",
]
