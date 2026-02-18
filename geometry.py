"""
Ý nghĩa hình học của Canonical Correlation Analysis (CCA).

Mô tả các khái niệm hình học: không gian biến, hướng canonical, góc giữa
các biến canonical, và mối liên hệ tương quan–góc.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from core import CCA


def get_geometry_description(cca: "CCA", language: str = "vi") -> str:
    """
    Trả về mô tả ý nghĩa hình học của CCA dựa trên kết quả đã fit.

    Args:
        cca: Đối tượng CCA đã fit (có x_scores, y_scores, canonical_correlations, ...).
        language: "vi" (tiếng Việt) hoặc "en" (tiếng Anh).

    Returns:
        Chuỗi mô tả ý nghĩa hình học.
    """
    r = cca.canonical_correlations
    n_comp = cca.n_components
    # Góc (radian) tương ứng: correlation = cos(angle) khi variates đã chuẩn hóa
    angles_rad = np.arccos(np.clip(r, -1, 1))
    angles_deg = np.degrees(angles_rad)

    if language == "vi":
        lines = [
            "## Ý nghĩa hình học của CCA",
            "",
            "### 1. Hai không gian biến",
            "- **X1** và **X2** tương ứng với hai không gian con (subspace) trong không gian mẫu ℝⁿ.",
            "- Mỗi mẫu là một điểm; mỗi biến là một trục. CCA làm việc với các tổ hợp tuyến tính của các biến.",
            "",
            "### 2. Hướng canonical (Canonical directions)",
            "- CCA tìm các **cặp hướng** (w₁ trong không gian X1, w₂ trong không gian X2) sao cho hai tổ hợp tuyến tính "
            "U = X1·w₁ và V = X2·w₂ có **tương quan cực đại**.",
            "- Các cặp tiếp theo được chọn lần lượt, **trực giao** với các cặp trước (trong không gian đã chuẩn hóa).",
            "",
            "### 3. Biến canonical (Canonical variates)",
            "- **U (x_scores)** và **V (y_scores)** là hình chiếu của dữ liệu lên các hướng canonical.",
            "- Chúng là tọa độ của các điểm trong hệ trục mới (hệ trục canonical).",
            "",
            "### 4. Tương quan và góc",
            "- Với dữ liệu đã chuẩn hóa, **hệ số tương quan r = cos(θ)** với θ là góc giữa hai vector U và V (theo từng cặp CC).",
            "- r càng gần 1 thì góc càng gần 0° (hai hướng gần trùng); r càng gần 0 thì góc càng gần 90°.",
            "",
            "### 5. Góc tương ứng với từng thành phần canonical",
            "",
        ]
        for i in range(n_comp):
            lines.append(f"- **CC{i+1}**: r = {r[i]:.4f} → góc θ ≈ {angles_deg[i]:.1f}° (cos θ = r).")
        lines.extend([
            "",
            "### 6. Tóm tắt hình học",
            "- CCA **xoay** hai không gian biến để tìm các trục (canonical) mà khi chiếu dữ liệu lên đó, "
            "hai bên **đồng biến** tối đa (tương quan cao).",
            "- Số chiều của không gian canonical bằng số thành phần (components); mỗi thành phần tương ứng một cặp trục và một góc (một r).",
        ])
    else:
        lines = [
            "## Geometric meaning of CCA",
            "",
            "### 1. Two variable spaces",
            "- **X1** and **X2** correspond to two subspaces in sample space ℝⁿ.",
            "- Each sample is a point; each variable is an axis. CCA works with linear combinations of variables.",
            "",
            "### 2. Canonical directions",
            "- CCA finds **pairs of directions** (w₁ in X1-space, w₂ in X2-space) such that the linear combinations "
            "U = X1·w₁ and V = X2·w₂ have **maximum correlation**.",
            "- Subsequent pairs are chosen to be **orthogonal** to previous pairs (in the standardized space).",
            "",
            "### 3. Canonical variates",
            "- **U (x_scores)** and **V (y_scores)** are the projections of the data onto the canonical directions.",
            "- They are the coordinates of the points in the new (canonical) coordinate system.",
            "",
            "### 4. Correlation and angle",
            "- For standardized data, **correlation r = cos(θ)** where θ is the angle between the two vectors U and V (for each CC pair).",
            "- r close to 1 ⇒ angle close to 0°; r close to 0 ⇒ angle close to 90°.",
            "",
            "### 5. Angles for each canonical component",
            "",
        ]
        for i in range(n_comp):
            lines.append(f"- **CC{i+1}**: r = {r[i]:.4f} → angle θ ≈ {angles_deg[i]:.1f}° (cos θ = r).")
        lines.extend([
            "",
            "### 6. Geometric summary",
            "- CCA **rotates** the two variable spaces to find axes (canonical) such that when data are projected onto them, "
            "the two sides **covary** maximally (high correlation).",
            "- The dimension of the canonical space equals the number of components; each component corresponds to one pair of axes and one angle (one r).",
        ])

    return "\n".join(lines)


def get_canonical_angles_degrees(cca: "CCA") -> np.ndarray:
    """
    Trả về góc (độ) tương ứng với mỗi canonical correlation.
    Công thức: r = cos(θ) ⇒ θ = arccos(r).

    Args:
        cca: Đối tượng CCA đã fit.

    Returns:
        Mảng góc (độ), shape (n_components,).
    """
    r = np.asarray(cca.canonical_correlations)
    r = np.clip(r, -1.0, 1.0)
    return np.degrees(np.arccos(r))


def plot_correlation_angle(cca: "CCA", figsize: Tuple[float, float] = (10, 4)) -> plt.Figure:
    """
    Vẽ hai đồ thị: (1) Canonical correlations; (2) Góc (độ) tương ứng với từng CC.
    Thể hiện mối quan hệ r = cos(θ).

    Args:
        cca: Đối tượng CCA đã fit.
        figsize: Kích thước figure.

    Returns:
        matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    r = cca.canonical_correlations
    angles_deg = get_canonical_angles_degrees(cca)
    n = len(r)
    x = np.arange(1, n + 1)

    ax1.bar(x, r, color="steelblue", alpha=0.7)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Canonical correlation r")
    ax1.set_title("Canonical correlations\n(r = cos θ)")
    ax1.set_xticks(x)
    ax1.grid(True, alpha=0.3)

    ax2.bar(x, angles_deg, color="coral", alpha=0.7)
    ax2.set_xlabel("Component")
    ax2.set_ylabel("Angle θ (degrees)")
    ax2.set_title("Angle between U and V\n(θ = arccos(r))")
    ax2.set_xticks(x)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_geometry_schematic(figsize: Tuple[float, float] = (8, 5)) -> plt.Figure:
    """
    Vẽ sơ đồ minh họa ý nghĩa hình học: hai không gian, hai hướng canonical,
    và góc θ (tương quan r = cos θ). Không dùng dữ liệu thật, chỉ minh họa khái niệm.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Vòng tròn đơn vị (góc)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.5)

    # Ví dụ: r = 0.8 => angle ~ 37°
    r_ex = 0.8
    angle_ex = np.arccos(r_ex)
    ax.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.05, fc="steelblue", ec="steelblue", linewidth=2, label="U (X1)")
    ax.arrow(0, 0, np.cos(angle_ex), np.sin(angle_ex), head_width=0.05, head_length=0.05, fc="coral", ec="coral", linewidth=2, label="V (X2)")
    ax.text(1.15, 0, "U", fontsize=12, fontweight="bold", color="steelblue")
    ax.text(np.cos(angle_ex) * 1.15, np.sin(angle_ex) * 1.15, "V", fontsize=12, fontweight="bold", color="coral")

    # Cung góc
    arc_theta = np.linspace(0, angle_ex, 30)
    ax.plot(0.3 * np.cos(arc_theta), 0.3 * np.sin(arc_theta), "green", linewidth=2)
    ax.text(0.38 * np.cos(angle_ex / 2), 0.38 * np.sin(angle_ex / 2), "θ", fontsize=14, color="green", fontweight="bold")
    ax.text(0.5, -0.25, f"r = cos(θ) = {r_ex}", fontsize=11, ha="center")

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_title("Geometric meaning: correlation = cos(angle)\nCanonical variates U and V")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_first_pair_scatter_with_angle(cca: "CCA", figsize: Tuple[float, float] = (6, 5)) -> plt.Figure:
    """
    Vẽ scatter U1 vs V1 và đường chéo (góc 45°). Gợi ý: điểm càng gần đường chéo
    thì tương quan càng cao; độ lệch so với đường chéo liên quan góc θ.

    Args:
        cca: Đối tượng CCA đã fit.
        figsize: Kích thước figure.

    Returns:
        matplotlib Figure.
    """
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
