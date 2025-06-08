#!/usr/bin/env python3
"""
exact_solver.py

精确薛定谔方程求解器

提供精确数值求解含时薛定谔方程的功能，用于DAPT方法的验证。

Author: Gilbert Young
Date: 2025-06-07
"""

import numpy as np
import time
from scipy.integrate import solve_ivp

from .hamiltonian import get_hamiltonian


def solve_schrodinger_exact(s_span, params, initial_state_vector):
    """
    数值求解完整的含时薛定谔方程

    薛定谔方程：i ℏ v |∂ψ/∂s⟩ = H(s)|ψ⟩
    转换为：|∂ψ/∂s⟩ = -i H(s)|ψ⟩ / (ℏ v)

    参数：
    - s_span: 重标定时间范围数组
    - params: 物理参数字典，必须包含'hbar'和'v'
    - initial_state_vector: 初始态向量 (4×1复数向量)

    返回：
    - exact_solution: 精确解的时间演化，形状为(len(s_span), 4)
    """
    print(f"🎯 开始精确薛定谔方程求解...")
    start_time = time.time()
    print(f"   时间点数量: {len(s_span)}")
    print(f"   时间范围: {s_span[0]:.3f} → {s_span[-1]:.3f}")

    # 提取物理参数
    hbar = params.get("hbar", 1.0)  # 默认ℏ=1
    v = params.get("v", 1.0)  # 默认v=1

    def schrodinger_ode_system(s, y):
        """
        薛定谔方程系统
        y是一个8维实数向量，表示4×1复数向量|ψ⟩的实部和虚部
        """
        # 重构复数向量|ψ⟩
        psi_real = y[:4]
        psi_imag = y[4:]
        psi = psi_real + 1j * psi_imag

        # 获取当前时刻的哈密顿量
        H = get_hamiltonian(s, params)

        # 计算导数：d|ψ⟩/ds = -i H(s)|ψ⟩ / (ℏ v)
        dpsi_ds = -1j * H @ psi / (hbar * v)

        # 分离实部和虚部
        dpsi_real = np.real(dpsi_ds)
        dpsi_imag = np.imag(dpsi_ds)

        return np.concatenate([dpsi_real, dpsi_imag])

    # 设置初始条件
    psi_0_real = np.real(initial_state_vector)
    psi_0_imag = np.imag(initial_state_vector)
    y0 = np.concatenate([psi_0_real, psi_0_imag])

    # 求解ODE
    print("   正在求解含时薛定谔方程...")
    ode_start = time.time()
    solution = solve_ivp(
        schrodinger_ode_system,
        (s_span[0], s_span[-1]),
        y0,
        t_eval=s_span,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    if not solution.success:
        raise RuntimeError(f"薛定谔方程求解失败: {solution.message}")

    ode_time = time.time() - ode_start
    print(f"   ✅ ODE求解完成 (耗时: {ode_time:.2f}s)")

    # 重构复数向量解
    print("   重构复数波函数...")
    exact_solution = np.zeros((len(s_span), 4), dtype=complex)

    for i in range(len(s_span)):
        psi_real = solution.y[:4, i]
        psi_imag = solution.y[4:, i]
        exact_solution[i] = psi_real + 1j * psi_imag

    total_time = time.time() - start_time
    print(f"   ✅ 精确解计算完成！总耗时: {total_time:.2f}s")

    return exact_solution


def calculate_exact_energy_expectation(exact_solution, s_span, params):
    """
    计算精确解的能量期望值随时间的变化

    参数：
    - exact_solution: 精确解的时间演化 (len(s_span), 4)
    - s_span: 时间点数组
    - params: 物理参数字典

    返回：
    - energy_expectation: 能量期望值数组
    """
    energy_expectation = np.zeros(len(s_span))

    for i, s in enumerate(s_span):
        psi = exact_solution[i]
        H = get_hamiltonian(s, params)

        # 计算能量期望值 ⟨ψ|H|ψ⟩
        energy_expectation[i] = np.real(np.dot(psi.conj(), H @ psi))

    return energy_expectation


def verify_exact_solution_properties(exact_solution, s_span, params, tolerance=1e-10):
    """
    验证精确解的基本物理性质

    参数：
    - exact_solution: 精确解的时间演化
    - s_span: 时间点数组
    - params: 物理参数字典
    - tolerance: 数值容忍度

    返回：
    - verification_results: 验证结果字典
    """
    num_times = len(s_span)

    # 检查归一化
    norms = np.array([np.linalg.norm(exact_solution[i]) ** 2 for i in range(num_times)])
    norm_variations = np.abs(norms - 1.0)
    max_norm_error = np.max(norm_variations)

    # 检查能量守恒（对于绝热情况）
    energy_expectation = calculate_exact_energy_expectation(
        exact_solution, s_span, params
    )
    energy_variation = np.max(energy_expectation) - np.min(energy_expectation)

    # 检查数值稳定性
    contains_nan = np.any(np.isnan(exact_solution))
    contains_inf = np.any(np.isinf(exact_solution))

    return {
        "normalization": {
            "max_error": max_norm_error,
            "is_conserved": max_norm_error < tolerance,
        },
        "energy": {
            "expectation_values": energy_expectation,
            "variation": energy_variation,
            "initial": energy_expectation[0],
            "final": energy_expectation[-1],
        },
        "numerical_stability": {
            "contains_nan": contains_nan,
            "contains_inf": contains_inf,
            "is_stable": not (contains_nan or contains_inf),
        },
        "overall_valid": (
            max_norm_error < tolerance and not contains_nan and not contains_inf
        ),
    }
