"""
CCA Geometry Visualization (Textbook Style)

- Two skewed planes (X and Y)
- Basis vectors (x1,x2) (y1,y2)
- Canonical directions Vx, Vy
- Angle φ with cosφ = r
- Distance e between vectors
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.colors import to_rgba
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from core import CCA


# =========================
# BASIC UTILITIES
# =========================

def get_canonical_angles_degrees(cca):
    """Convert canonical correlations to geometric angles in degrees."""
    r = np.asarray(cca.canonical_correlations)
    r = np.clip(r, -1.0, 1.0)
    return np.degrees(np.arccos(r))


def get_geometry_description(cca, language="vi"):
    """Return a human-readable interpretation of CCA as angle minimization."""
    r = cca.canonical_correlations
    angles = get_canonical_angles_degrees(cca)

    lines = []
    if language == "vi":
        lines.append("## Ý nghĩa hình học của CCA\n")
        for i, (ri, ai) in enumerate(zip(r, angles)):
            lines.append(f"- CC{i+1}: r = {ri:.4f} → góc ≈ {ai:.2f}°")
        lines.append("\n👉 r = cos(φ) ⇒ CCA = tối thiểu hóa góc giữa 2 vector canonical")
    else:
        lines.append("## Geometric meaning of CCA\n")
        for i, (ri, ai) in enumerate(zip(r, angles)):
            lines.append(f"- CC{i+1}: r = {ri:.4f} → angle ≈ {ai:.2f}°")
        lines.append("\n👉 r = cos(φ) ⇒ CCA minimizes angle between canonical vectors")

    return "\n".join(lines)


# =========================
# CORE: TEXTBOOK STYLE PLOT
# =========================

def plot_cca_geometry(cca, component=0, figsize=(12, 6)):
    """
    Draw geometric meaning of CCA like textbook diagram
    """

    r = float(np.clip(cca.canonical_correlations[component], -1, 1))
    phi = np.arccos(r)
    e_len = np.sqrt(2 - 2 * r)

    fig, ax = plt.subplots(figsize=figsize)

    # -------------------------
    # Helper: draw vector
    # -------------------------
    def draw_vec(base, vec, color, label):
        """Draw a vector arrow with a label in the current axis."""
        ax.arrow(
            base[0], base[1],
            vec[0], vec[1],
            head_width=0.12,
            length_includes_head=True,
            color=color,
            linewidth=2
        )
        ax.text(
            base[0] + vec[0]*1.1,
            base[1] + vec[1]*1.1,
            label,
            fontsize=11,
            color=color,
            fontweight="bold"
        )

    # =========================
    # LEFT: PLANE Y
    # =========================
    origin_y = np.array([-3, 0])

    y1 = np.array([1.0, 1.3])
    y2 = np.array([-1.0, 1.1])
    vy = np.array([0.3, 1.8])

    plane_y = np.array([
        origin_y + [-1.5, -1],
        origin_y + [1.5, -1],
        origin_y + [2.0, 2],
        origin_y + [-1.0, 2]
    ])

    ax.plot(*plane_y.T, color="black")

    draw_vec(origin_y, y1, "green", "y₁")
    draw_vec(origin_y, y2, "green", "y₂")
    draw_vec(origin_y, vy, "red", "Vᵧ")

    ax.text(origin_y[0], origin_y[1]+2.5, "plane Y", fontsize=12)

    # =========================
    # RIGHT: PLANE X
    # =========================
    origin_x = np.array([3, 0])

    x1 = np.array([1.0, 1.1])
    x2 = np.array([-0.8, 1.3])
    vx = np.array([0.7, 1.7])

    plane_x = np.array([
        origin_x + [-1.5, -1],
        origin_x + [1.5, -1],
        origin_x + [2.0, 2],
        origin_x + [-1.0, 2]
    ])

    ax.plot(*plane_x.T, color="black")

    draw_vec(origin_x, x1, "blue", "x₁")
    draw_vec(origin_x, x2, "blue", "x₂")
    draw_vec(origin_x, vx, "red", "Vₓ")

    ax.text(origin_x[0], origin_x[1]+2.5, "plane X", fontsize=12)

    # =========================
    # CONNECTION e
    # =========================
    p1 = origin_y + vy
    p2 = origin_x + vx

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "r--", linewidth=1.5)
    ax.text((p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + 0.2, f"e ≈ {e_len:.2f}", color="red")

    # =========================
    # RIGHT SIDE: ANGLE φ
    # =========================
    center = np.array([7, 0])

    u = np.array([0, 2])
    v = np.array([2*np.sin(phi), 2*np.cos(phi)])

    draw_vec(center, u, "red", "Vᵧ")
    draw_vec(center, v, "red", "Vₓ")

    # arc
    t = np.linspace(0, phi, 40)
    ax.plot(
        center[0] + 0.7*np.sin(t),
        center[1] + 0.7*np.cos(t),
        "black"
    )

    ax.text(center[0]+0.6, center[1]+0.6, "φ", fontsize=13)

    # =========================
    # FINAL STYLE
    # =========================
    ax.set_xlim(-5, 10)
    ax.set_ylim(-2, 4)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        f"CCA Geometry (Component {component+1})\n"
        f"r = {r:.3f} = cos(φ)",
        fontsize=13
    )

    plt.tight_layout()
    return fig


# =========================
# OPTIONAL: BAR + ANGLE PLOT
# =========================

def plot_correlation_angle(cca):
    """Plot canonical correlations and their equivalent principal angles."""
    r = cca.canonical_correlations
    angles = get_canonical_angles_degrees(cca)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(1, len(r)+1)

    ax1.bar(x, r)
    ax1.set_title("Correlation (r = cos φ)")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("r")

    ax2.bar(x, angles)
    ax2.set_title("Angle φ (degrees)")
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Degrees")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Textbook-style schematic (conceptual; uses only r = ρ from fitted CCA)
# ---------------------------------------------------------------------------


def plot_cca_variable_spaces_canonical(
    cca: "CCA",
    component: int = 0,
    ix: Tuple[int, int] = (0, 1),
    iy: Tuple[int, int] = (0, 1),
    figsize: Tuple[float, float] = (14, 5.2),
    language: str = "vi",
) -> plt.Figure:
    """
    Conceptual textbook diagram (not a data projection).

    Two parallel-looking skewed planes (Y green, X blue), short basis arrows,
    nearly parallel canonical arrows :math:`V_y`, :math:`V_x`, a horizontal
    segment ``e`` between their tips, and a symmetric angle panel with
    :math:`\\varphi = \\arccos(r)`.

    Only ``cca.canonical_correlations[component]`` is used from the fit (for
    :math:`r` and :math:`\\varphi`). ``ix``, ``iy``, ``language`` are ignored
    for drawing but kept for API compatibility.
    """
    _ = (ix, iy, language)  # kept for API compatibility; schematic only uses r

    k = int(component)
    if k < 0 or k >= cca.n_components:
        raise ValueError(f"component must be in [0, {cca.n_components - 1}]")

    r = float(np.clip(cca.canonical_correlations[k], -1.0, 1.0))
    phi = float(np.arccos(r))

    # Shared skewed planes. The layout is intentionally textbook-like.
    w = np.array([2.4, -0.26], dtype=float)
    h = np.array([0.42, 2.55], dtype=float)
    w_u = w / (np.linalg.norm(w) + 1e-15)
    h_u = h / (np.linalg.norm(h) + 1e-15)

    tip_y1 = 0.95 * w_u + 0.35 * h_u
    tip_y2 = -0.42 * w_u + 1.55 * h_u
    tip_x1 = 1.05 * w_u + 0.48 * h_u
    tip_x2 = -0.08 * w_u + 1.45 * h_u

    # Canonical direction: close to vertical inside each plane.
    u_canon = 0.18 * w_u + 0.98 * h_u
    u_canon = u_canon / (np.linalg.norm(u_canon) + 1e-15)
    L_canon = 1.95

    O_y = np.array([1.55, 1.05], dtype=float)
    O_x = O_y + np.array([5.35, 0.0])  # translate; same orientation

    corners_y = np.array([O_y, O_y + w, O_y + w + h, O_y + h])
    corners_x = np.array([O_x, O_x + w, O_x + w + h, O_x + h])

    vy_tip = O_y + L_canon * u_canon
    vx_tip = O_x + L_canon * u_canon
    # Horizontal segment between tips (same y by construction)
    e_mid = (vy_tip + vx_tip) / 2.0

    # --- Right panel: symmetric V_y / V_x / φ (mini diagram)
    O_tri = np.array([12.25, 1.7], dtype=float)
    L_tri = 2.25
    dir_vy = np.array([-np.sin(phi / 2.0), np.cos(phi / 2.0)], dtype=float)
    dir_vx = np.array([np.sin(phi / 2.0), np.cos(phi / 2.0)], dtype=float)
    T_vy = O_tri + L_tri * dir_vy
    T_vx = O_tri + L_tri * dir_vx

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ap_base = dict(arrowstyle="->", mutation_scale=18, shrinkA=0, shrinkB=0)

    # Plane Y
    ax.add_patch(
        Polygon(corners_y, closed=True, facecolor="white", edgecolor="black", linewidth=2.8, zorder=1)
    )
    cy = np.mean(corners_y, axis=0)
    ax.text(
        cy[0],
        corners_y[:, 1].max() + 0.18,
        r"plane $y$",
        fontsize=23,
        fontstyle="italic",
        fontfamily="serif",
        ha="center",
        va="bottom",
    )

    ax.annotate("", xy=O_y + tip_y1, xytext=O_y, arrowprops={**ap_base, "color": "#2ca84a", "lw": 3.2}, zorder=3)
    ax.annotate("", xy=O_y + tip_y2, xytext=O_y, arrowprops={**ap_base, "color": "#2ca84a", "lw": 3.2}, zorder=3)
    ax.annotate("", xy=vy_tip, xytext=O_y, arrowprops={**ap_base, "color": "#d7191c", "lw": 3.8}, zorder=4)
    ax.text(O_y[0] + tip_y1[0] + 0.10, O_y[1] + tip_y1[1] + 0.06, r"$y_1$", fontsize=17, color="#2ca84a", fontweight="bold")
    ax.text(O_y[0] + tip_y2[0] - 0.10, O_y[1] + tip_y2[1] + 0.05, r"$y_2$", fontsize=17, color="#2ca84a", fontweight="bold", ha="right")
    ax.text(vy_tip[0] - 0.08, vy_tip[1] + 0.18, r"$v_y$", fontsize=23, color="#d7191c", fontweight="bold", ha="right")

    # Plane X
    ax.add_patch(
        Polygon(corners_x, closed=True, facecolor="white", edgecolor="black", linewidth=2.8, zorder=1)
    )
    cx = np.mean(corners_x, axis=0)
    ax.text(
        cx[0],
        corners_x[:, 1].max() + 0.18,
        r"plane $x$",
        fontsize=23,
        fontstyle="italic",
        fontfamily="serif",
        ha="center",
        va="bottom",
    )

    ax.annotate("", xy=O_x + tip_x1, xytext=O_x, arrowprops={**ap_base, "color": "#2c4bd6", "lw": 3.2}, zorder=3)
    ax.annotate("", xy=O_x + tip_x2, xytext=O_x, arrowprops={**ap_base, "color": "#2c4bd6", "lw": 3.2}, zorder=3)
    ax.annotate("", xy=vx_tip, xytext=O_x, arrowprops={**ap_base, "color": "#d7191c", "lw": 3.8}, zorder=4)
    ax.text(O_x[0] + tip_x1[0] + 0.08, O_x[1] + tip_x1[1] - 0.04, r"$x_1$", fontsize=17, color="#2c4bd6", fontweight="bold")
    ax.text(O_x[0] + tip_x2[0] - 0.06, O_x[1] + tip_x2[1] + 0.02, r"$x_2$", fontsize=17, color="#2c4bd6", fontweight="bold", ha="right")
    ax.text(vx_tip[0] + 0.08, vx_tip[1] + 0.18, r"$v_x$", fontsize=23, color="#d7191c", fontweight="bold", ha="left")

    ax.plot(
        [vy_tip[0], vx_tip[0]],
        [vy_tip[1], vx_tip[1]],
        color="#d7191c",
        linewidth=3.0,
        zorder=5,
        solid_capstyle="round",
    )
    ax.text(e_mid[0], e_mid[1] + 0.12, r"$e$", fontsize=19, color="#d7191c", fontweight="bold", ha="center", va="bottom")

    # Mini angle diagram
    ap_sym = dict(arrowstyle="->", color="#d7191c", lw=3.8, mutation_scale=18, shrinkA=0, shrinkB=0)
    ax.annotate("", xy=T_vy, xytext=O_tri, arrowprops=ap_sym, zorder=6)
    ax.annotate("", xy=T_vx, xytext=O_tri, arrowprops=ap_sym, zorder=6)
    ax.plot([T_vy[0], T_vx[0]], [T_vy[1], T_vx[1]], color="#d7191c", linewidth=3.0, zorder=5, solid_capstyle="round")

    arc_r = 0.72
    arc_angles = np.linspace(np.pi / 2.0 - phi / 2.0, np.pi / 2.0 + phi / 2.0, 40)
    ax.plot(
        O_tri[0] + arc_r * np.cos(arc_angles),
        O_tri[1] + arc_r * np.sin(arc_angles),
        "k-",
        lw=1.6,
        zorder=5,
    )
    ax.text(O_tri[0], O_tri[1] + 0.68, r"$\phi$", fontsize=22, color="black", fontweight="bold", ha="center", va="bottom")
    ax.text(T_vy[0] - 0.12, T_vy[1] + 0.12, r"$v_y$", fontsize=22, color="#d7191c", fontweight="bold", ha="right")
    ax.text(T_vx[0] + 0.12, T_vx[1] + 0.12, r"$v_x$", fontsize=22, color="#d7191c", fontweight="bold", ha="left")
    ax.text((T_vy[0] + T_vx[0]) / 2, max(T_vy[1], T_vx[1]) + 0.06, r"$e$", fontsize=18, color="#d7191c", fontweight="bold", ha="center")

    ax.text(
        7.2,
        0.35,
        rf"CC{k + 1}: r = {r:.3f}, \phi = arccos(r) \approx {np.degrees(phi):.1f}°",
        fontsize=11,
        color="#333",
        ha="center",
    )

    ax.set_xlim(0.1, 15.0)
    ax.set_ylim(0.0, 5.8)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    plt.tight_layout()
    return fig


def plot_cca_variable_spaces_canonical(
    cca: "CCA",
    component: int = 0,
    ix: Tuple[int, int] = (0, 1),
    iy: Tuple[int, int] = (0, 1),
    figsize: Tuple[float, float] = (14, 5.2),
    language: str = "vi",
) -> plt.Figure:
    """
    Canvas-like textbook diagram matching `MSA-theory/.cursor/skills/geo.html`.

    Notes:
    - Conceptual axes are drawn in a fixed 2D canvas coordinate system (x: 0..900, y: 0..500).
    - canonical directions v_x, v_y are derived from weights using selected 2D coordinates
      (components ix=(ix0,ix1) and iy=(iy0,iy1)).
    - Optionally overlays dataset points projected onto the visual planes (controlled
      by attributes set in `demo_app.py`):
        - cca._geo_show_points (bool, default True)
        - cca._geo_pt_size (int, default 5)
        - cca._geo_pt_alpha (int 0..100, default 75)
    """
    _ = language  # only affects text elsewhere

    CANVAS_W, CANVAS_H = 900, 500
    sc = 55.0
    scale2 = 0.45
    pt_size = int(getattr(cca, "_geo_pt_size", 5))
    pt_alpha_pct = float(getattr(cca, "_geo_pt_alpha", 75))
    pt_alpha = float(np.clip(pt_alpha_pct / 100.0, 0.0, 1.0))
    show_points = bool(getattr(cca, "_geo_show_points", True))

    k = int(component)
    if k < 0 or k >= cca.n_components:
        raise ValueError(f"component must be in [0, {cca.n_components - 1}]")

    r = float(np.clip(cca.canonical_correlations[k], -1.0, 1.0))
    phi = float(np.arccos(r))
    e_len = float(np.sqrt(max(0.0, 2.0 - 2.0 * r)))

    # ---- Canonical direction (2D) from weights using selected indices
    def _norm2(v: np.ndarray) -> np.ndarray:
        """Normalize a 2D vector and return a safe fallback for zero norm."""
        v = np.asarray(v, dtype=float).ravel()
        n = float(np.linalg.norm(v))
        if n < 1e-15:
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    ix0, ix1 = int(ix[0]), int(ix[1])
    iy0, iy1 = int(iy[0]), int(iy[1])
    ix0 = max(0, min(ix0, cca.x_weights.shape[0] - 1))
    ix1 = max(0, min(ix1, cca.x_weights.shape[0] - 1))
    iy0 = max(0, min(iy0, cca.y_weights.shape[0] - 1))
    iy1 = max(0, min(iy1, cca.y_weights.shape[0] - 1))

    vx_dir = _norm2(np.array([cca.x_weights[ix0, k], cca.x_weights[ix1, k]], dtype=float))
    vy_dir = _norm2(np.array([cca.y_weights[iy0, k], cca.y_weights[iy1, k]], dtype=float))

    vx = vx_dir * 1.6
    vy = vy_dir * 1.6

    # ---- Basis directions (fixed, like geo.html)
    y1 = np.array([0.85, 0.55], dtype=float)
    y2 = np.array([-0.35, 0.95], dtype=float)
    x1 = np.array([0.9, 0.45], dtype=float)
    x2 = np.array([-0.25, 0.9], dtype=float)

    # ---- Canvas setup
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(CANVAS_H, 0)  # y-down like canvas
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    red = "#d7191c"
    green = "#2ca84a"
    blue = "#2c4bd6"

    def vec_tip(Oc: np.ndarray, v: np.ndarray) -> np.ndarray:
        # In geo.html: yPt(v) = [Ox + v.x*sc, Oy - v.y*sc]
        """Map a local direction vector to a canvas-space endpoint."""
        return np.array([Oc[0] + v[0] * sc, Oc[1] - v[1] * sc], dtype=float)

    def draw_arrow(base: np.ndarray, tip: np.ndarray, color: str, lw: float, ms: float = 12) -> None:
        """Render a styled arrow from base to tip in canvas coordinates."""
        ax.annotate(
            "",
            xy=(float(tip[0]), float(tip[1])),
            xytext=(float(base[0]), float(base[1])),
            arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=ms),
            zorder=4,
        )

    def draw_plane(cx: float, cy: float, label: str) -> None:
        # drawPlane() in geo.html
        """Draw a skewed conceptual variable plane and its caption."""
        pts = np.array(
            [
                [cx - 60, cy + 80],
                [cx + 80, cy + 80],
                [cx + 100, cy - 80],
                [cx - 40, cy - 80],
            ],
            dtype=float,
        )
        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor=(1.0, 1.0, 1.0, 0.85),
                edgecolor="#1a1a1a",
                linewidth=2.5,
                zorder=1,
            )
        )
        ax.text(
            cx + 20,
            cy - 92,
            label,
            fontsize=18,
            fontstyle="italic",
            fontfamily="serif",
            ha="center",
            va="bottom",
            color="#1a1a1a",
        )

    # ---- Coordinates (matching geo.html)
    OY = np.array([210.0, 300.0], dtype=float)
    OX = np.array([530.0, 300.0], dtype=float)
    OY_c = np.array([OY[0] - 10, OY[1] + 40], dtype=float)
    OX_c = np.array([OX[0] - 5, OX[1] + 40], dtype=float)
    O_tri = np.array([800.0, 295.0], dtype=float)
    L_tri = 90.0

    # ===================== ZONE Y (left plane)
    draw_plane(OY[0], OY[1], "plane y")

    y1_tip = vec_tip(OY_c, y1)
    y2_tip = vec_tip(OY_c, y2)
    vy_tip = vec_tip(OY_c, vy)

    draw_arrow(OY_c, y1_tip, green, lw=2.5, ms=14)
    draw_arrow(OY_c, y2_tip, green, lw=2.5, ms=14)
    draw_arrow(OY_c, vy_tip, red, lw=3.2, ms=16)

    ax.text(y1_tip[0] + 12, y1_tip[1] - 5, r"$y_1$", fontsize=14, fontweight="bold", color=green)
    ax.text(y2_tip[0] - 12, y2_tip[1] - 5, r"$y_2$", fontsize=14, fontweight="bold", color=green, ha="right")
    ax.text(vy_tip[0] - 14, vy_tip[1] - 8, r"$v_y$", fontsize=17, fontweight="bold", color=red)

    if show_points:
        y_scores_k = np.asarray(cca.y_scores[:, k], dtype=float)
        orthoY = _norm2(np.array([-vy_dir[1], vy_dir[0]], dtype=float))
        for i, s in enumerate(y_scores_k):
            s1 = s * 0.28
            s2 = (i % 3 - 1) * 0.18 * scale2
            px = OY_c[0] + (vy_dir[0] * s1 + orthoY[0] * s2) * sc
            py = OY_c[1] - (vy_dir[1] * s1 + orthoY[1] * s2) * sc

            face = to_rgba("#e97f0a", pt_alpha)
            edge = to_rgba("#c06000", 0.8)
            ax.add_patch(Circle((px, py), radius=pt_size, facecolor=face, edgecolor=edge, linewidth=0.8, zorder=2))
            ax.text(px + pt_size + 1, py + 3, f"p{i+1}", fontsize=8, color="#7a4000", ha="left", va="center")

    # ===================== ZONE X (middle plane)
    draw_plane(OX[0], OX[1], "plane x")

    x1_tip = vec_tip(OX_c, x1)
    x2_tip = vec_tip(OX_c, x2)
    vx_tip = vec_tip(OX_c, vx)

    draw_arrow(OX_c, x1_tip, blue, lw=2.5, ms=14)
    draw_arrow(OX_c, x2_tip, blue, lw=2.5, ms=14)
    draw_arrow(OX_c, vx_tip, red, lw=3.2, ms=16)

    ax.text(x1_tip[0] + 12, x1_tip[1] - 5, r"$x_1$", fontsize=14, fontweight="bold", color=blue)
    ax.text(x2_tip[0] - 10, x2_tip[1] - 5, r"$x_2$", fontsize=14, fontweight="bold", color=blue, ha="right")
    ax.text(vx_tip[0] + 14, vx_tip[1] - 8, r"$v_x$", fontsize=17, fontweight="bold", color=red)

    if show_points:
        x_scores_k = np.asarray(cca.x_scores[:, k], dtype=float)
        orthoX = _norm2(np.array([-vx_dir[1], vx_dir[0]], dtype=float))
        for i, s in enumerate(x_scores_k):
            s1 = s * 0.28
            s2 = (i % 3 - 1) * 0.18 * scale2
            px = OX_c[0] + (vx_dir[0] * s1 + orthoX[0] * s2) * sc
            py = OX_c[1] - (vx_dir[1] * s1 + orthoX[1] * s2) * sc

            face = to_rgba("#8b2fc9", pt_alpha)
            edge = to_rgba("#5a1a99", 0.8)
            ax.add_patch(Circle((px, py), radius=pt_size, facecolor=face, edgecolor=edge, linewidth=0.8, zorder=2))
            ax.text(px + pt_size + 1, py + 3, f"p{i+1}", fontsize=8, color="#3d1060", ha="left", va="center")

    # ===================== CONNECTION e (dashed line)
    e_mid = (vy_tip + vx_tip) / 2.0
    ax.plot(
        [vy_tip[0], vx_tip[0]],
        [vy_tip[1], vx_tip[1]],
        color=red,
        linewidth=2,
        linestyle=(0, (6, 4)),
        zorder=3,
    )
    ax.text(e_mid[0], e_mid[1] - 10, "e", fontsize=15, fontweight="bold", color=red, ha="center", va="center")

    # ===================== ZONE C (triangle)
    dvy = np.array([-np.sin(phi / 2.0), -np.cos(phi / 2.0)], dtype=float)
    dvx = np.array([np.sin(phi / 2.0), -np.cos(phi / 2.0)], dtype=float)
    T_vy = O_tri + dvy * L_tri
    T_vx = O_tri + dvx * L_tri

    draw_arrow(O_tri, T_vy, red, lw=3.0, ms=16)
    draw_arrow(O_tri, T_vx, red, lw=3.0, ms=16)
    ax.plot([T_vy[0], T_vx[0]], [T_vy[1], T_vx[1]], color=red, linewidth=2, zorder=3)

    ax.text(T_vy[0] - 12, T_vy[1] - 6, r"$v_y$", fontsize=15, fontweight="bold", color=red, ha="right", va="center")
    ax.text(T_vx[0] + 14, T_vx[1] - 6, r"$v_x$", fontsize=15, fontweight="bold", color=red, ha="left", va="center")
    ax.text((T_vy[0] + T_vx[0]) / 2.0, min(T_vy[1], T_vx[1]) - 6, "e", fontsize=14, fontweight="bold", color=red, ha="center", va="center")

    # arc for phi
    arcR = 28.0
    a0 = float(np.arctan2(dvy[1], dvy[0]))
    a1 = float(np.arctan2(dvx[1], dvx[0]))
    if a1 < a0:
        a1 += 2 * np.pi
    arc_t = np.linspace(a0, a1, 60)
    ax.plot(
        O_tri[0] + arcR * np.cos(arc_t),
        O_tri[1] + arcR * np.sin(arc_t),
        color="#1a1a1a",
        linewidth=1.5,
        linestyle="--",
        zorder=3,
    )
    ax.text(O_tri[0], O_tri[1] - arcR - 3, r"$\varphi$", fontsize=14, color="#1a1a1a", fontweight="bold", ha="center", va="center")

    # bottom stats (like geo.html)
    ax.text(
        CANVAS_W / 2.0,
        CANVAS_H - 14,
        f"CC{k+1}: r = {r:.4f},  φ = arccos(r) ≈ {np.degrees(phi):.1f}°,  ‖e‖ = {e_len:.3f}",
        fontsize=11,
        color="#555",
        ha="center",
        va="bottom",
        zorder=10,
    )

    plt.tight_layout()
    return fig


def plot_geometry_schematic(figsize: Tuple[float, float] = (8, 5)) -> plt.Figure:
    """Conceptual diagram: r = cos(θ) for two unit directions (no real CCA data)."""
    fig, ax = plt.subplots(figsize=figsize)

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.5)

    r_ex = 0.8
    angle_ex = np.arccos(r_ex)
    ax.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.05, fc="steelblue", ec="steelblue", linewidth=2)
    ax.arrow(
        0,
        0,
        np.cos(angle_ex),
        np.sin(angle_ex),
        head_width=0.05,
        head_length=0.05,
        fc="coral",
        ec="coral",
        linewidth=2,
    )
    ax.text(1.15, 0, "U", fontsize=12, fontweight="bold", color="steelblue")
    ax.text(
        np.cos(angle_ex) * 1.15,
        np.sin(angle_ex) * 1.15,
        "V",
        fontsize=12,
        fontweight="bold",
        color="coral",
    )

    arc_theta = np.linspace(0, angle_ex, 30)
    ax.plot(0.3 * np.cos(arc_theta), 0.3 * np.sin(arc_theta), "green", linewidth=2)
    ax.text(
        0.38 * np.cos(angle_ex / 2),
        0.38 * np.sin(angle_ex / 2),
        "θ",
        fontsize=14,
        color="green",
        fontweight="bold",
    )
    ax.text(0.5, -0.25, f"r = cos(θ) = {r_ex}", fontsize=11, ha="center")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title("Geometric meaning: correlation = cos(angle)\nCanonical variates U and V")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_first_pair_scatter_with_angle(cca: "CCA", figsize: Tuple[float, float] = (6, 5)) -> plt.Figure:
    """Scatter U₁ vs V₁ with diagonal reference; angle relates to r."""
    fig, ax = plt.subplots(figsize=figsize)

    u1 = cca.x_scores[:, 0]
    v1 = cca.y_scores[:, 0]
    r1 = cca.canonical_correlations[0]
    angle_deg = get_canonical_angles_degrees(cca)[0]

    ax.scatter(u1, v1, alpha=0.5, s=15, color="steelblue")
    lim_lo = min(u1.min(), v1.min())
    lim_hi = max(u1.max(), v1.max())
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", alpha=0.7, linewidth=2, label="U = V (r=1)")
    ax.set_xlabel("U₁ (X1 - CC1)")
    ax.set_ylabel("V₁ (X2 - CC1)")
    ax.set_title(f"CC1: U₁ vs V₁\nr = {r1:.4f} → angle θ ≈ {angle_deg:.1f}°")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig
