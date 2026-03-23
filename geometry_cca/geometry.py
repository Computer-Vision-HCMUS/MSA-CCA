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
from matplotlib.patches import Polygon
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from core import CCA


# =========================
# BASIC UTILITIES
# =========================

def get_canonical_angles_degrees(cca):
    r = np.asarray(cca.canonical_correlations)
    r = np.clip(r, -1.0, 1.0)
    return np.degrees(np.arccos(r))


def get_geometry_description(cca, language="vi"):
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
