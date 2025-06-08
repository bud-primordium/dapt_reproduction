"""
hamiltonian.py

该模块负责定义DAPT数值示例中的四能级系统哈密顿量及其本征体系计算。

主要功能：
1. 根据复现笔记中的(Eq. 134)构造时变哈密顿量
2. 使用解析公式计算瞬时本征值和本征矢量(Eq. 137-140)
3. 自动保证本征矢量的连续性（通过解析公式）

Author: Gilbert Young
Date: 2025-06-07
"""

import numpy as np
import warnings


def get_hamiltonian(s, params):
    """
    根据复现笔记(Eq. 134)构造四能级系统的瞬时哈密顿量

    哈密顿量形式：
    H(s) = (1/√2) * [[0, H1(s)],
                     [H1†(s), 0]]

    其中 H1(s) = [[-E(s), e^(-iθ(s))E(s)],
                  [e^(iθ(s))E(s), E(s)]]

    参数：
    - s: 重标定时间参数 (0 ≤ s ≤ 1)
    - params: 包含物理参数的字典
        * 'E0': 基本能隙
        * 'lambda': 能隙变化开关
        * 'theta0': 初始相位
        * 'w': 相位变化率

    返回：
    - H: 4×4复数哈密顿量矩阵
    """
    # 提取参数
    E0 = params["E0"]
    lam = params["lambda"]
    theta0 = params["theta0"]
    w = params["w"]

    # 计算时变能量和相位 (Eq. 135 & 136)
    E_s = E0 + lam * (s - 0.5) ** 2  # E(s)
    theta_s = theta0 + w * s**2  # θ(s)

    # 构造2×2子矩阵H1(s)
    H1 = np.array(
        [[-E_s, np.exp(-1j * theta_s) * E_s], [np.exp(1j * theta_s) * E_s, E_s]],
        dtype=complex,
    )

    # 构造完整的4×4哈密顿量
    zeros_2x2 = np.zeros((2, 2), dtype=complex)
    H = (1.0 / np.sqrt(2)) * np.block([[zeros_2x2, H1], [H1.conj().T, zeros_2x2]])

    return H


def get_eigensystem(s, params, prev_eigenvectors=None):
    """
    使用解析公式直接计算瞬时本征值和本征矢量

    根据复现笔记(Eq. 137-140)的解析公式：
    - 基态能量: E_GS(s) = -E(s) (二重简并)
    - 激发态能量: E_EX(s) = +E(s) (二重简并)
    - 本征矢量由解析公式给出，自动满足连续性

    参数：
    - s: 重标定时间参数
    - params: 物理参数字典
    - prev_eigenvectors: 兼容性参数，被忽略（解析公式自动保证连续性）

    返回：
    - eigenvalues: 排序后的本征值数组 (4,)，顺序为 [-E(s), -E(s), +E(s), +E(s)]
    - eigenvectors: 本征矢量矩阵 (4, 4)，每列为一个本征矢，按照子空间排序
    """
    # 提取参数并计算时变函数
    E0 = params["E0"]
    lam = params["lambda"]
    theta0 = params["theta0"]
    w = params["w"]

    # 计算时变能量和相位
    E_s = E0 + lam * (s - 0.5) ** 2  # E(s)
    theta_s = theta0 + w * s**2  # θ(s)

    # 本征值：两个简并能级
    eigenvalues = np.array([-E_s, -E_s, +E_s, +E_s])

    # 根据解析公式构造本征矢量 (Eq. 137-140)
    # 标准基矢顺序：{|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}

    # 基态子空间 H_0 的基矢
    # |0^0(s)⟩ = (1/2)[e^(-iθ(s))|↑↑⟩ + |↑↓⟩ - √2|↓↓⟩]
    state_00 = np.array(
        [
            np.exp(-1j * theta_s) / 2,  # |↑↑⟩的系数
            1.0 / 2,  # |↑↓⟩的系数
            0.0,  # |↓↑⟩的系数
            -np.sqrt(2) / 2,  # |↓↓⟩的系数
        ],
        dtype=complex,
    )

    # |0^1(s)⟩ = (1/2)[|↑↑⟩ - e^(iθ(s))|↑↓⟩ + √2|↓↑⟩]
    state_01 = np.array(
        [
            1.0 / 2,  # |↑↑⟩的系数
            -np.exp(1j * theta_s) / 2,  # |↑↓⟩的系数
            np.sqrt(2) / 2,  # |↓↑⟩的系数
            0.0,  # |↓↓⟩的系数
        ],
        dtype=complex,
    )

    # 激发态子空间 H_1 的基矢
    # |1^0(s)⟩ = (1/2)[e^(-iθ(s))|↑↑⟩ + |↑↓⟩ + √2|↓↓⟩]
    state_10 = np.array(
        [
            np.exp(-1j * theta_s) / 2,  # |↑↑⟩的系数
            1.0 / 2,  # |↑↓⟩的系数
            0.0,  # |↓↑⟩的系数
            np.sqrt(2) / 2,  # |↓↓⟩的系数
        ],
        dtype=complex,
    )

    # |1^1(s)⟩ = (1/2)[|↑↑⟩ - e^(iθ(s))|↑↓⟩ - √2|↓↑⟩]
    state_11 = np.array(
        [
            1.0 / 2,  # |↑↑⟩的系数
            -np.exp(1j * theta_s) / 2,  # |↑↓⟩的系数
            -np.sqrt(2) / 2,  # |↓↑⟩的系数
            0.0,  # |↓↓⟩的系数
        ],
        dtype=complex,
    )

    # 构造本征矢量矩阵：按子空间顺序排列
    # 列顺序：[|0^0⟩, |0^1⟩, |1^0⟩, |1^1⟩]
    eigenvectors = np.column_stack([state_00, state_01, state_10, state_11])

    return eigenvalues, eigenvectors


def get_eigenvector_derivatives(s, params):
    """
    解析计算本征矢量对时间s的导数

    由于本征矢量有解析表达式，我们可以直接对其求导，
    这比数值微分更精确，用于计算M矩阵

    参数：
    - s: 重标定时间参数
    - params: 物理参数字典

    返回：
    - eigenvector_derivatives: 本征矢量导数矩阵 (4, 4)，每列为一个本征矢的导数
    """
    # 提取参数
    E0 = params["E0"]
    lam = params["lambda"]
    theta0 = params["theta0"]
    w = params["w"]

    # 计算时变函数的导数
    # dE/ds = 2λ(s - 1/2)
    dE_ds = 2 * lam * (s - 0.5)

    # dθ/ds = 2ws
    dtheta_ds = 2 * w * s

    # 当前时刻的相位
    theta_s = theta0 + w * s**2

    # 计算各本征矢量的导数

    # d|0^0⟩/ds
    dstate_00_ds = np.array(
        [
            -1j * dtheta_ds * np.exp(-1j * theta_s) / 2,  # d/ds[e^(-iθ)/2]
            0.0,  # d/ds[1/2]
            0.0,  # d/ds[0]
            0.0,  # d/ds[-√2/2]
        ],
        dtype=complex,
    )

    # d|0^1⟩/ds
    dstate_01_ds = np.array(
        [
            0.0,  # d/ds[1/2]
            -1j
            * dtheta_ds
            * (np.exp(1j * theta_s))
            / 2,  # Corrected: removed the minus sign
            0.0,  # d/ds[√2/2]
            0.0,  # d/ds[0]
        ],
        dtype=complex,
    )

    # d|1^0⟩/ds
    dstate_10_ds = np.array(
        [
            -1j * dtheta_ds * np.exp(-1j * theta_s) / 2,  # d/ds[e^(-iθ)/2]
            0.0,  # d/ds[1/2]
            0.0,  # d/ds[0]
            0.0,  # d/ds[√2/2]
        ],
        dtype=complex,
    )

    # d|1^1⟩/ds
    dstate_11_ds = np.array(
        [
            0.0,  # d/ds[1/2]
            -1j
            * dtheta_ds
            * (np.exp(1j * theta_s))
            / 2,  # Corrected: removed the minus sign
            0.0,  # d/ds[-√2/2]
            0.0,  # d/ds[0]
        ],
        dtype=complex,
    )

    # 构造导数矩阵
    eigenvector_derivatives = np.column_stack(
        [dstate_00_ds, dstate_01_ds, dstate_10_ds, dstate_11_ds]
    )

    return eigenvector_derivatives


def calculate_energy_gap(s, params):
    """
    计算瞬时能隙

    参数：
    - s: 重标定时间参数
    - params: 物理参数字典

    返回：
    - gap: 基态子空间与激发态子空间之间的能隙
    """
    # 直接使用解析公式计算能隙
    E0 = params["E0"]
    lam = params["lambda"]
    E_s = E0 + lam * (s - 0.5) ** 2

    # 能隙 = E_EX - E_GS = E(s) - (-E(s)) = 2E(s)
    gap = 2 * E_s

    return gap


def get_initial_state_in_standard_basis(params):
    """
    根据复现笔记(Eq. 137)，计算初始态|0^0(0)⟩在标准基{|↑↑⟩, |↑↓⟩, |↓↑⟩, |↓↓⟩}下的表示

    |0^0(0)⟩ = (1/2)[e^(-iθ₀)|↑↑⟩ + |↑↓⟩ - √2|↓↓⟩]

    参数：
    - params: 物理参数字典

    返回：
    - initial_state: 4×1复数向量，表示初始态在标准基下的系数
    """
    theta0 = params["theta0"]

    # 根据(Eq. 137)构造初始态
    initial_state = np.array(
        [
            np.exp(-1j * theta0) / 2,  # |↑↑⟩的系数
            1.0 / 2,  # |↑↓⟩的系数
            0.0,  # |↓↑⟩的系数
            -np.sqrt(2) / 2,  # |↓↓⟩的系数
        ],
        dtype=complex,
    )

    return initial_state


def verify_analytical_eigensystem(s, params, tolerance=1e-12):
    """
    验证解析本征系统的正确性

    检查：
    1. H|ψ⟩ = E|ψ⟩ 是否成立
    2. 本征矢量是否正交归一
    3. 简并度是否正确

    参数：
    - s: 重标定时间参数
    - params: 物理参数字典
    - tolerance: 数值误差容忍度

    返回：
    - verification_result: 验证结果的字典
    """
    H = get_hamiltonian(s, params)
    eigenvalues, eigenvectors = get_eigensystem(s, params)

    results = {
        "eigenvalue_equation_errors": [],
        "orthonormality_check": True,
        "degeneracy_check": True,
        "max_eigenvalue_error": 0.0,
        "max_orthogonality_error": 0.0,
    }

    # 检查本征值方程 H|ψ⟩ = E|ψ⟩
    for i in range(4):
        computed_result = H @ eigenvectors[:, i]
        expected_result = eigenvalues[i] * eigenvectors[:, i]
        error = np.linalg.norm(computed_result - expected_result)
        results["eigenvalue_equation_errors"].append(error)
        results["max_eigenvalue_error"] = max(results["max_eigenvalue_error"], error)

    # 检查正交归一性
    orthogonality_matrix = eigenvectors.conj().T @ eigenvectors
    identity_matrix = np.eye(4)
    orthogonality_error = np.linalg.norm(orthogonality_matrix - identity_matrix)
    results["max_orthogonality_error"] = orthogonality_error
    results["orthonormality_check"] = orthogonality_error < tolerance

    # 检查简并度
    unique_eigenvalues = np.unique(np.round(eigenvalues, 10))
    expected_degeneracy = len(unique_eigenvalues) == 2  # 应该有两个不同的能级
    results["degeneracy_check"] = expected_degeneracy

    results["is_valid"] = (
        results["max_eigenvalue_error"] < tolerance
        and results["orthonormality_check"]
        and results["degeneracy_check"]
    )

    return results
