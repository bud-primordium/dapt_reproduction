"""
core.py

该模块包含DAPT (简并绝热微扰理论) 的核心计算逻辑。

主要功能：
1. 计算耦合矩阵M
2. 求解Wilczek-Zee相矩阵
3. DAPT递推算法实现
4. 精确薛定谔方程求解

Author: Gilbert Young
Date: 2025-06-07
Version: 2.5 - 【二阶修正】修复对角项ODE求解中的索引和矩阵乘法顺序错误
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, CubicSpline
from .hamiltonian import get_hamiltonian, get_eigensystem, get_eigenvector_derivatives
import time
import sys
import warnings

# 尝试导入tqdm，如果没有则使用简单的进度指示
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # 简单的进度条替代
    def tqdm(iterable, desc="", total=None):
        """简单的进度指示器，当tqdm不可用时使用"""
        if total is None:
            total = len(iterable) if hasattr(iterable, "__len__") else None

        print(f"{desc}...")
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 10) == 0:
                progress = (i + 1) / total * 100
                print(f"   进度: {progress:.1f}% ({i+1}/{total})")
            yield item
        print(f"   {desc} 完成。")


def calculate_M_matrix(s, ds, params, get_eigensystem_func=None):
    """
    解析计算DAPT耦合矩阵M^{nm}(s)

    M^{nm}(s) = ⟨n(s)|∂_s m(s)⟩
    使用解析导数公式替代数值微分，提高精度

    参数：
    - s: 当前时间点
    - ds: 兼容性参数（新版中已不再使用）
    - params: 物理参数字典
    - get_eigensystem_func: 兼容性参数（新版中已不再使用）

    返回：
    - M_matrix: 字典，键为(n,m)元组，值为对应的耦合矩阵
    """
    # 获取当前时刻的本征体系和导数
    _, eigenvectors_curr = get_eigensystem(s, params)
    eigenvectors_dot = get_eigenvector_derivatives(s, params)

    # 计算M矩阵元素：M^{nm} = ⟨n|∂_s m⟩
    M_matrix = {}

    # 按照简并子空间分组：基态子空间(0,1)，激发态子空间(2,3)
    subspaces = [(0, 2), (2, 4)]  # [start, end) for each subspace

    for n_subspace_idx, (n_start, n_end) in enumerate(subspaces):
        for m_subspace_idx, (m_start, m_end) in enumerate(subspaces):
            # 计算子空间间的耦合矩阵
            n_dim = n_end - n_start
            m_dim = m_end - m_start

            M_nm = np.zeros((n_dim, m_dim), dtype=complex)

            for i in range(n_dim):
                for j in range(m_dim):
                    n_idx = n_start + i
                    m_idx = m_start + j

                    # M^{nm}_{ij} = ⟨n_i|∂_s m_j⟩
                    M_nm[i, j] = np.dot(
                        eigenvectors_curr[:, n_idx].conj(), eigenvectors_dot[:, m_idx]
                    )

            M_matrix[(n_subspace_idx, m_subspace_idx)] = M_nm

    return M_matrix


def solve_wz_phase(s_span, M_nn_func, U_n_0):
    """
    求解Wilczek-Zee (WZ) 相矩阵U^n(s)

    微分方程：dU^n/ds = -U^n(s) M^{nn}(s)

    参数：
    - s_span: 时间范围数组
    - M_nn_func: 返回M^{nn}(s)的函数，接受参数s，返回2×2矩阵
    - U_n_0: 初始WZ矩阵 (2×2复数矩阵)

    返回：
    - U_n_solution: WZ矩阵的时间演化，形状为(len(s_span), 2, 2)
    """
    print(f"      求解WZ相矩阵微分方程...")
    start_time = time.time()

    # 将2×2复数矩阵微分方程转换为8维实数向量微分方程
    def wz_ode_system(s, y):
        """
        WZ微分方程系统
        y是一个8维实数向量，表示2×2复数矩阵U^n的实部和虚部
        """
        # 重构复数矩阵U^n
        U_real = y[:4].reshape(2, 2)
        U_imag = y[4:].reshape(2, 2)
        U_n = U_real + 1j * U_imag

        # 获取当前时刻的M^{nn}矩阵
        M_nn = M_nn_func(s)

        # 【理论关键修正】根据Debug笔记Eq.29: dU^n/ds = -U^n(s) M^{nn}(s)
        dU_n_ds = -U_n @ M_nn

        # 分离实部和虚部
        dU_real = np.real(dU_n_ds).flatten()
        dU_imag = np.imag(dU_n_ds).flatten()

        return np.concatenate([dU_real, dU_imag])

    # 设置初始条件
    U_n_0_real = np.real(U_n_0).flatten()
    U_n_0_imag = np.imag(U_n_0).flatten()
    y0 = np.concatenate([U_n_0_real, U_n_0_imag])

    # 求解ODE
    solution = solve_ivp(
        wz_ode_system,
        (s_span[0], s_span[-1]),
        y0,
        t_eval=s_span,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not solution.success:
        raise RuntimeError(f"WZ相矩阵求解失败: {solution.message}")

    # 重构WZ矩阵解
    U_n_solution = np.zeros((len(s_span), 2, 2), dtype=complex)

    if len(s_span) == 1:
        # 特殊处理单时间点的情况
        y_array = np.array(solution.y).flatten()
        U_real = y_array[:4].reshape(2, 2)
        U_imag = y_array[4:].reshape(2, 2)
        U_n_solution[0] = U_real + 1j * U_imag
    else:
        # 多时间点的情况
        y_array = np.array(solution.y)
        for i in range(len(s_span)):
            U_real = y_array[:4, i].reshape(2, 2)
            U_imag = y_array[4:, i].reshape(2, 2)
            U_n_solution[i] = U_real + 1j * U_imag

    elapsed_time = time.time() - start_time
    print(f"      WZ相矩阵计算完成 (耗时: {elapsed_time:.2f}s)")

    return U_n_solution


def dapt_recursive_step(
    s_span, B_coeffs_p, M_matrix_func, U_matrices, Delta_func, params, order_p
):
    """
    DAPT递推关系的单步实现

    【重大理论修正】根据用户分析，修正论文Eq.(25)中的递推关系错误：
    正确的递推关系：dB_{mn}^{(p)}/ds + Σ_k B_{nk}^{(p)} M^{km} = ...

    参数：
    - s_span: 时间网格
    - B_coeffs_p: 第p阶系数矩阵字典，键为(m,n)，值为时间序列
    - M_matrix_func: M矩阵计算函数
    - U_matrices: WZ相矩阵字典，键为子空间索引
    - Delta_func: 能隙函数
    - params: 物理参数
    - order_p: 当前阶数p

    返回：
    - B_coeffs_p_plus_1: 第p+1阶系数矩阵字典
    """
    hbar = params.get("hbar", 1.0)
    dt = s_span[1] - s_span[0]  # 假设等间距网格

    B_coeffs_p_plus_1 = {}

    # 首先计算所有非对角项 (m ≠ n)
    # 【理论修正】正确的递推关系：B_{mn}^{(p+1)} = (iℏ/Δ_{nm})[Ḃ_{mn}^{(p)} + Σ_k B_{nk}^{(p)} M^{km}]

    for m in range(2):  # 子空间索引
        for n in range(2):
            if m != n:  # 非对角项
                # 计算Ḃ_{mn}^{(p)}（时间导数）
                B_mn_p = B_coeffs_p[(m, n)]
                # 尝试使用更高阶的边界条件进行数值微分
                B_mn_p_dot = np.gradient(B_mn_p, dt, axis=0, edge_order=2)

                # 【核心修正】使用向量化计算 Σ_k B_{nk}^{(p)} M^{km}
                summation_term = np.zeros_like(B_mn_p, dtype=complex)
                for k in range(2):  # 遍历中间子空间 k
                    # 【最终修正】获取 B_{nk}^{(p)} (索引: n, k)
                    B_nk_p = B_coeffs_p[(n, k)]
                    # 【最终修正】获取 M^{km} 矩阵 (索引: k, m)
                    M_km_series = np.array([M_matrix_func(s)[(k, m)] for s in s_span])
                    # 【最终修正】正确的矩阵乘法: B_nk @ M^km
                    batch_product = np.einsum("tik,tkj->tij", B_nk_p, M_km_series)
                    summation_term += batch_product

                # 计算能隙Δ_{nm}
                Delta_nm = np.zeros(len(s_span))
                for i, s in enumerate(s_span):
                    Delta_nm[i] = Delta_func(s, m, n)

                # 应用递推公式
                B_mn_p_plus_1 = np.zeros_like(B_mn_p, dtype=complex)
                for i in range(len(s_span)):
                    if abs(Delta_nm[i]) > 1e-12:  # 避免除零
                        B_mn_p_plus_1[i] = (1j * hbar / Delta_nm[i]) * (
                            B_mn_p_dot[i] - summation_term[i]
                        )
                    else:
                        B_mn_p_plus_1[i] = np.zeros_like(B_mn_p[i])

                B_coeffs_p_plus_1[(m, n)] = B_mn_p_plus_1

    # 然后计算对角项 (m = n)
    # 根据修正后的递推关系求解微分方程
    for n in range(2):
        # 对角项的微分方程：dB_{nn}^{(p+1)}/ds = -Σ_{k≠n} B_{nk}^{(p+1)} M^{kn} - B_{nn}^{(p+1)} M^{nn}

        # 计算初始条件：B_{nn}^{(p+1)}(0) = -Σ_{m≠n} B_{mn}^{(p+1)}(0)
        initial_condition = np.zeros((2, 2), dtype=complex)
        for m in range(2):
            if m != n:
                initial_condition -= B_coeffs_p_plus_1[(m, n)][0]

        # 求解微分方程
        B_nn_solution = _solve_diagonal_ode(
            s_span,
            n,
            initial_condition,
            B_coeffs_p_plus_1,
            M_matrix_func,
            U_matrices[n],
        )

        B_coeffs_p_plus_1[(n, n)] = B_nn_solution

    if order_p == 0:  # 只在计算第一阶修正时打印
        print("\n--- DEBUG: B^(1) Coefficients ---")
        B_10_p1 = B_coeffs_p_plus_1[(1, 0)]
        B_01_p1 = B_coeffs_p_plus_1[(0, 1)]

        # 检查 B_10 是否接近于零
        norm_B10 = np.linalg.norm(B_10_p1)
        print(f"Norm of B_10^(1): {norm_B10}")
        if np.allclose(norm_B10, 0):
            print("   [诊断确认] B_10^(1) 几乎为零，这是问题的根源！")

        # 检查 B_01 是否非零
        norm_B01 = np.linalg.norm(B_01_p1)
        print(f"Norm of B_01^(1): {norm_B01}")
        if not np.allclose(norm_B01, 0):
            print("   [诊断确认] B_01^(1) 非零，但对波函数修正贡献错误。")
        print("---------------------------------\n")
    return B_coeffs_p_plus_1


def _solve_diagonal_ode(
    s_span,
    subspace_n,
    initial_condition,
    B_coeffs_off_diag,
    M_matrix_func,
    U_n_matrices,
):
    """
    求解对角项的微分方程

    【理论修正】dB_{nn}^{(p+1)}/ds + Σ_k B_{nk}^{(p+1)} M^{kn} = 0

    使用高精度ODE求解器scipy.integrate.solve_ivp
    """

    # 【精度优化】为所有非对角项创建三次样条插值对象
    spline_interpolators = {}
    for key in B_coeffs_off_diag:
        m, n = key
        if m != n:  # 只为非对角项创建插值器
            # 为B_mn^{(p+1)}的每个矩阵元素创建样条插值
            B_mn_data = B_coeffs_off_diag[key]  # shape: (N_steps, 2, 2)
            spline_interpolators[key] = {}

            for i in range(2):
                for j in range(2):
                    # 提取实部和虚部分别插值
                    real_data = np.real(B_mn_data[:, i, j])
                    imag_data = np.imag(B_mn_data[:, i, j])

                    spline_interpolators[key][(i, j)] = {
                        "real": CubicSpline(s_span, real_data, bc_type="natural"),
                        "imag": CubicSpline(s_span, imag_data, bc_type="natural"),
                    }

    def diagonal_ode_system(s, y):
        """
        【理论修正版】对角项微分方程系统
        y是一个8维实数向量，表示2×2复数矩阵B_nn的实部和虚部
        """
        # 重构复数矩阵B_nn
        B_real = y[:4].reshape(2, 2)
        B_imag = y[4:].reshape(2, 2)
        B_nn = B_real + 1j * B_imag

        # 【理论修正】计算右侧项：-Σ_{k≠n} B_{nk}^{(p+1)} M^{kn}
        rhs = np.zeros((2, 2), dtype=complex)
        M_matrices = M_matrix_func(s)

        for k in range(2):  # 遍历中间子空间 k
            if k != subspace_n:  # k != n
                # 获取 B_{nk}^{(p+1)}
                B_nk_current = np.zeros((2, 2), dtype=complex)
                for i in range(2):
                    for j in range(2):
                        # 注意键是 (subspace_n, k)，对应 B_{nk}
                        spline_real = spline_interpolators[(subspace_n, k)][(i, j)][
                            "real"
                        ]
                        spline_imag = spline_interpolators[(subspace_n, k)][(i, j)][
                            "imag"
                        ]
                        s_clamped = np.clip(s, s_span[0], s_span[-1])
                        B_nk_current[i, j] = spline_real(s_clamped) + 1j * spline_imag(
                            s_clamped
                        )
                # 获取 M^{kn}
                M_kn = M_matrices[(k, subspace_n)]

                # 正确的矩阵乘法顺序
                rhs -= B_nk_current @ M_kn

        # 微分方程：dB/ds = rhs - B @ M^{nn}
        M_nn = M_matrices[(subspace_n, subspace_n)]
        dB_ds = rhs - B_nn @ M_nn

        # 将复数矩阵导数转换为实数向量
        dB_real = np.real(dB_ds).flatten()
        dB_imag = np.imag(dB_ds).flatten()

        return np.concatenate([dB_real, dB_imag])

    # 将初始复数矩阵转换为实数向量
    B_initial_real = np.real(initial_condition).flatten()
    B_initial_imag = np.imag(initial_condition).flatten()
    y_initial = np.concatenate([B_initial_real, B_initial_imag])

    # 使用solve_ivp求解
    sol = solve_ivp(
        diagonal_ode_system,
        [s_span[0], s_span[-1]],
        y_initial,
        t_eval=s_span,
        method="RK45",  # 使用4阶Runge-Kutta方法
        rtol=1e-8,  # 相对误差容限
        atol=1e-10,  # 绝对误差容限
    )

    if not sol.success:
        warnings.warn(f"对角项ODE求解失败: {sol.message}")
        # 降级到前向欧拉法作为备选
        return _solve_diagonal_ode_euler_fallback(
            s_span, subspace_n, initial_condition, B_coeffs_off_diag, M_matrix_func
        )

    # 将结果转换回复数矩阵形式
    B_nn_solution = np.zeros((len(s_span), 2, 2), dtype=complex)

    if len(s_span) == 1:
        # 特殊处理单时间点的情况
        y_array = np.array(sol.y).flatten()
        y_real = y_array[:4].reshape(2, 2)
        y_imag = y_array[4:].reshape(2, 2)
        B_nn_solution[0] = y_real + 1j * y_imag
    else:
        # 多时间点的情况
        y_array = np.array(sol.y)
        for i in range(len(s_span)):
            y_real = y_array[:4, i].reshape(2, 2)
            y_imag = y_array[4:, i].reshape(2, 2)
            B_nn_solution[i] = y_real + 1j * y_imag

    return B_nn_solution


def _solve_diagonal_ode_euler_fallback(
    s_span, subspace_n, initial_condition, B_coeffs_off_diag, M_matrix_func
):
    """
    对角项微分方程的前向欧拉法备选求解器
    只在solve_ivp失败时使用
    """
    dt = s_span[1] - s_span[0]
    B_nn_solution = np.zeros((len(s_span), 2, 2), dtype=complex)
    B_nn_solution[0] = initial_condition

    # 使用简单的前向欧拉方法求解
    for i in range(len(s_span) - 1):
        s = s_span[i]

        # 【理论修正】计算右侧项：-Σ_{k≠n} B_{nk}^{(p+1)} M^{kn}
        rhs = np.zeros((2, 2), dtype=complex)
        M_matrices = M_matrix_func(s)

        for k in range(2):  # 遍历中间子空间 k
            if k != subspace_n:  # k != n
                # 【理论修正】获取 B_{nk}^{(p+1)} 和 M^{kn}
                B_nk = B_coeffs_off_diag[(subspace_n, k)][i]  # 对应 B_{nk}^{(p+1)}
                M_kn = M_matrices[(k, subspace_n)]  # 对应 M^{k, n}

                # 【理论修正】正确的矩阵乘法顺序: B @ M
                rhs -= B_nk @ M_kn

        # 微分方程：dB/ds = rhs - B @ M^{nn}
        M_nn = M_matrices[(subspace_n, subspace_n)]
        dB_ds = rhs - B_nn_solution[i] @ M_nn

        # 前向欧拉步
        B_nn_solution[i + 1] = B_nn_solution[i] + dt * dB_ds

    return B_nn_solution


def run_dapt_calculation(s_span, order, params):
    """
    运行完整的DAPT计算

    参数：
    - s_span: 重标定时间范围数组
    - order: DAPT计算的最高阶数
    - params: 物理参数字典

    返回：
    - dapt_results: 字典，包含各阶DAPT近似解和中间计算结果
    """
    start_time = time.time()
    print(f"开始DAPT计算，最高阶数: {order}")
    print(f"   时间点数量: {len(s_span)}")
    print(f"   时间范围: {s_span[0]:.3f} → {s_span[-1]:.3f}")

    # 预计算M矩阵和WZ相矩阵
    print("\n第1步：预计算M矩阵和WZ相矩阵...")
    step1_start = time.time()
    ds = 1e-4  # M矩阵计算的数值微分步长（注释说明：新版中ds参数已不再使用）

    # 预计算M矩阵，减少重复计算
    print("   预计算所有时间点的M矩阵...")
    M_matrix_cache = {}
    for i, s in enumerate(tqdm(s_span, desc="计算M矩阵", total=len(s_span))):
        M_matrix_cache[s] = calculate_M_matrix(s, ds, params)

    def M_matrix_func(s):
        # 如果s在缓存中，直接返回
        if s in M_matrix_cache:
            return M_matrix_cache[s]

        # 否则使用最近邻或线性插值
        s_values = list(M_matrix_cache.keys())
        s_values.sort()

        # 找到最近的时间点
        if s <= s_values[0]:
            return M_matrix_cache[s_values[0]]
        elif s >= s_values[-1]:
            return M_matrix_cache[s_values[-1]]
        else:
            # 找到s左右的两个点，使用最近邻
            for i in range(len(s_values) - 1):
                if s_values[i] <= s <= s_values[i + 1]:
                    # 选择更近的点
                    if abs(s - s_values[i]) <= abs(s - s_values[i + 1]):
                        return M_matrix_cache[s_values[i]]
                    else:
                        return M_matrix_cache[s_values[i + 1]]

        # 如果都没找到，重新计算
        return calculate_M_matrix(s, ds, params)

    # 计算WZ相矩阵
    U_matrices = {}
    print("   计算WZ相矩阵...")
    for n in range(2):  # 两个简并子空间
        print(f"   计算第 {n} 个子空间的WZ相矩阵...")

        def M_nn_func(s):
            return M_matrix_func(s)[(n, n)]

        # 初始WZ矩阵为单位矩阵
        U_n_0 = np.eye(2, dtype=complex)
        U_matrices[n] = solve_wz_phase(s_span, M_nn_func, U_n_0)

    from scipy.integrate import cumulative_trapezoid

    # 【最终精度修复】预计算高精度动态相位并创建插值器
    print("   预计算高精度动态相位...")
    omega_interpolators = {}
    for n in range(2):  # 遍历子空间
        # 获取整个时间跨度上的能量
        energies_n = np.array([get_eigensystem(s, params)[0][2 * n] for s in s_span])
        # 使用累积梯形积分计算 omega_n(s)
        omega_n_series = cumulative_trapezoid(energies_n, s_span, initial=0)
        # 创建高精度插值函数
        omega_interpolators[n] = CubicSpline(s_span, omega_n_series)

    step1_time = time.time() - step1_start
    print(f"   第1步完成 (耗时: {step1_time:.2f}s)")

    def Delta_func(s, m, n):
        """
        【最终理论修正】计算子空间间的能隙 Δ_{nm} = E_n - E_m

        CLARIFICATION: 根据调试笔记，Δ_nm = E_n(s) - E_m(s).
        例如，从基态(m=0)到激发态(n=1)的能隙为 Δ_10 = E_1 - E_0 > 0.
        """
        eigenvalues, _ = get_eigensystem(s, params)

        # 子空间m的能量是 eigenvalues[2*m] (或 2*m + 1，因为简并)
        # 子空间n的能量是 eigenvalues[2*n] (或 2*n + 1，因为简并)
        E_m = eigenvalues[2 * m]
        E_n = eigenvalues[2 * n]

        return E_n - E_m

    # 初始化零阶系数：B^{(0)}_{mn} = b_n(0) U^n(s) δ_{mn}
    print("\n第2步：初始化零阶系数...")
    step2_start = time.time()
    B_coeffs = {}

    # 从基态|0^0(0)⟩开始，所以b_0(0) = 1, b_1(0) = 0
    for m in range(2):
        for n in range(2):
            if m == n == 0:
                # B^{(0)}_{00} = U^0(s)
                B_coeffs[(m, n)] = U_matrices[0]
            else:
                # 其他项为零
                B_coeffs[(m, n)] = np.zeros((len(s_span), 2, 2), dtype=complex)

    # 存储各阶结果
    all_B_coeffs = {0: B_coeffs}
    step2_time = time.time() - step2_start
    print(f"   第2步完成 (耗时: {step2_time:.2f}s)")

    # 递推计算高阶修正项
    if order > 0:
        print(f"\n第3步：递推计算高阶修正项 (0阶 → {order}阶)...")
        step3_start = time.time()
        for p in range(order):
            print(f"   计算第{p+1}阶修正项...")
            substep_start = time.time()
            B_coeffs_next = dapt_recursive_step(
                s_span, B_coeffs, M_matrix_func, U_matrices, Delta_func, params, p
            )
            all_B_coeffs[p + 1] = B_coeffs_next
            B_coeffs = B_coeffs_next
            substep_time = time.time() - substep_start
            print(f"   第{p+1}阶完成 (耗时: {substep_time:.2f}s)")
        step3_time = time.time() - step3_start
        print(f"   第3步完成 (耗时: {step3_time:.2f}s)")

    # --------------------------------------------------------------------------
    # 【最终修正】第4步：构造各阶DAPT近似解
    # --------------------------------------------------------------------------
    print(f"\n第4步：构造各阶DAPT近似解 (0阶 → {order}阶)...")
    step4_start = time.time()
    dapt_solutions = {}
    c_init = np.array([1.0, 0.0], dtype=complex)
    # 【关键】预计算源子空间(n=0)的相位因子，所有修正项共用
    omega_0_series = omega_interpolators[0](s_span)
    source_phase_factor_series = np.exp(-1j * omega_0_series / params.get("v", 1.0))
    psi_p_series = {}
    for p in range(order + 1):
        v_power = params.get("v", 1.0) ** p
        psi_p = np.zeros((len(s_span), 4), dtype=complex)
        for i, s in enumerate(s_span):
            _, eigenvectors = get_eigensystem(s, params)
            psi_p_i = np.zeros(4, dtype=complex)
            # 遍历所有 *目标* 子空间 n_target
            for n_target in range(2):
                B_n0_p = all_B_coeffs[p][(n_target, 0)][i]
                coeffs_in_n = B_n0_p @ c_init
                for alpha in range(2):
                    global_idx = 2 * n_target + alpha
                    base_state = eigenvectors[:, global_idx]
                    psi_p_i += coeffs_in_n[alpha] * base_state
            # 【关键】所有项乘以公共的源相位因子
            psi_p[i] = v_power * source_phase_factor_series[i] * psi_p_i
        psi_p_series[p] = psi_p
    # 累加构造各阶近似解
    for k in tqdm(range(order + 1), desc="构造各阶解", total=order + 1):
        psi_k = np.zeros((len(s_span), 4), dtype=complex)
        for p in range(k + 1):
            psi_k += psi_p_series[p]

        # 归一化以进行比较
        for i in range(len(s_span)):
            norm = np.linalg.norm(psi_k[i])
            if norm > 1e-12:
                psi_k[i] /= norm
        dapt_solutions[k] = psi_k
    step4_time = time.time() - step4_start
    total_time = time.time() - start_time
    print(f"   第4步完成 (耗时: {step4_time:.2f}s)")
    print(f"\nDAPT计算完成！总耗时: {total_time:.2f}s")
    print(f"   平均每阶耗时: {total_time/(order+1):.2f}s")

    return {
        "solutions": dapt_solutions,
        "B_coeffs": all_B_coeffs,
        "U_matrices": U_matrices,
        "M_matrix_func": M_matrix_func,
        "timing": {
            "total": total_time,
            "step1": step1_time,
            "step2": step2_time,
            "step3": step3_time if order > 0 else 0.0,
            "step4": step4_time,
        },
    }
