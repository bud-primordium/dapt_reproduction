"""
utils.py

该模块包含DAPT计算的辅助工具函数。

主要功能：
1. 不忠诚度计算
2. 结果可视化
3. 数据处理工具

Author: Gilbert Young
Date: 2025-06-07
Version: 2.1 - 修复版本
- 绘图修正：修复plot_infidelity_comparison函数
  将对数坐标轴(semilogy)改为线性坐标轴(plot)
- 样式增强：更新标记样式以匹配原文
  使用空心标记['o', 's', 'D']，调整标记密度
- 视觉优化：改进颜色方案和网格显示
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def calculate_infidelity(psi_exact, psi_approx):
    """
    根据复现笔记(Eq. 148)计算不忠诚度

    I(s) = 1 - |⟨Ψ_exact(s)|Ψ_approx(s)⟩|²

    参数：
    - psi_exact: 精确解波函数，形状为(N,) 复数向量
    - psi_approx: DAPT近似解波函数，形状为(N,) 复数向量

    返回：
    - infidelity: 不忠诚度值 (实数)
    """
    # 确保波函数已归一化
    psi_exact_norm = psi_exact / np.linalg.norm(psi_exact)
    psi_approx_norm = psi_approx / np.linalg.norm(psi_approx)

    # 计算内积
    overlap = np.dot(psi_exact_norm.conj(), psi_approx_norm)

    # 计算不忠诚度
    fidelity = abs(overlap) ** 2
    infidelity = 1.0 - fidelity

    return infidelity


def calculate_infidelity_series(psi_exact_series, psi_approx_series):
    """
    计算时间序列的不忠诚度

    参数：
    - psi_exact_series: 精确解时间序列，形状为(T, N)
    - psi_approx_series: 近似解时间序列，形状为(T, N)

    返回：
    - infidelity_series: 不忠诚度时间序列，形状为(T,)
    """
    num_times = psi_exact_series.shape[0]
    infidelity_series = np.zeros(num_times)

    for i in range(num_times):
        infidelity_series[i] = calculate_infidelity(
            psi_exact_series[i], psi_approx_series[i]
        )

    return infidelity_series


def plot_infidelity_comparison(
    s_span,
    infidelity_data,
    title="DAPT不忠诚度对比",
    epsilon_value=None,
    save_path=None,
):
    """
    【已修正】绘制不忠诚度对比图 (使用线性坐标轴)

    参数：
    - s_span: 时间数组
    - infidelity_data: 不忠诚度数据字典 {'Zeroth order': I_0, 'First order': I_1, ...}
    - title: 图表标题
    - epsilon_value: ε参数值，用于显示在图中
    - save_path: 保存路径，如果为None则不保存

    返回：
    - fig, ax: matplotlib图形对象
    """
    # 设置图表样式
    setup_matplotlib_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # 【修正】定义颜色和标记样式以匹配原文
    colors = ["black", "blue", "green", "orange", "purple"]
    markers = ["o", "s", "D", "^", "v"]  # 圆圈, 方块, 菱形, 三角形, 倒三角

    # 绘制各阶不忠诚度曲线
    for i, (label, infidelity) in enumerate(infidelity_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        # 【核心修正】使用 ax.plot 替代 ax.semilogy
        ax.plot(
            s_span,
            infidelity,
            color=color,
            marker=marker,
            linestyle="-",
            markerfacecolor="none",  # 制作空心标记
            markeredgecolor=color,
            linewidth=1.5,
            markersize=6,
            markevery=max(1, len(s_span) // 25),  # 调整标记密度
            label=label,
        )

    # 设置坐标轴
    ax.set_xlabel("重标定时间 s", fontsize=14)
    ax.set_ylabel("不忠诚度 (Infidelity)", fontsize=14)
    ax.set_title(title, fontsize=16)

    # 设置网格
    ax.grid(True, alpha=0.4)
    ax.set_xlim(s_span[0], s_span[-1])
    ax.set_ylim(bottom=0)  # 确保Y轴从0开始

    # 添加图例
    ax.legend(loc="best", fontsize=12)

    # 如果提供了ε值，添加到图中
    if epsilon_value is not None:
        ax.text(
            0.05,
            0.9,
            f"ε ≈ {epsilon_value:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图片已保存至: {save_path}")

    return fig, ax


def calculate_epsilon_parameter(params):
    """
    计算DAPT的关键参数ε(s)

    ε(s) = √2 ℏ v / E(s)

    参数：
    - params: 物理参数字典

    返回：
    - epsilon_func: ε(s)函数
    - epsilon_min: 最小ε值（当E(s)最大时）
    """
    hbar = params.get("hbar", 1.0)
    v = params.get("v", 1.0)
    E0 = params["E0"]
    lam = params["lambda"]

    def epsilon_func(s):
        E_s = E0 + lam * (s - 0.5) ** 2
        return np.sqrt(2) * hbar * v / E_s

    # 对于时变能隙情况，最小ε在s=0.5时达到
    if lam != 0:
        epsilon_min = np.sqrt(2) * hbar * v / E0
    else:
        # 恒定能隙情况
        epsilon_min = np.sqrt(2) * hbar * v / E0

    return epsilon_func, epsilon_min


def format_scientific_notation(value, precision=2):
    """
    格式化科学计数法显示
    """
    if value == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)

    if abs(exponent) <= 3:
        return f"{value:.{precision}f}"
    else:
        return f"{mantissa:.{precision}f}×10^{{{exponent}}}"


def save_results_to_file(results_dict, filename):
    """
    将计算结果保存到文件

    参数：
    - results_dict: 结果字典
    - filename: 保存的文件名
    """
    import pickle

    with open(filename, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"结果已保存至: {filename}")


def load_results_from_file(filename):
    """
    从文件加载计算结果

    参数：
    - filename: 文件名

    返回：
    - results_dict: 结果字典
    """
    import pickle

    with open(filename, "rb") as f:
        results_dict = pickle.load(f)

    print(f"结果已从 {filename} 加载")
    return results_dict


def setup_matplotlib_style():
    """
    设置matplotlib样式以匹配论文图表风格
    """
    # 设置中文字体支持
    rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    # 设置图表样式
    rcParams["figure.figsize"] = (10, 6)
    rcParams["font.size"] = 12
    rcParams["axes.linewidth"] = 1.2
    rcParams["lines.linewidth"] = 2
    rcParams["grid.alpha"] = 0.3
