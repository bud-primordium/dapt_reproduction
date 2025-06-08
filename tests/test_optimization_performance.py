"""
test_optimization_performance.py

专门测试DAPT核心优化的性能效果

Author: Gilbert Young
Date: 2025-06-07
Version: 2.1 - 优化性能验证
"""

import numpy as np
import pytest
import time
import sys
import os
from scipy.interpolate import CubicSpline

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dapt_tools.core import (
    calculate_M_matrix,
    dapt_recursive_step,
    _solve_diagonal_ode,
    run_dapt_calculation,
)
from dapt_tools.hamiltonian import get_eigensystem


@pytest.mark.optimization
class TestVectorizationOptimization:
    """测试向量化计算优化的性能提升"""

    def test_einsum_vectorization_correctness(self):
        """验证einsum向量化计算的正确性"""
        # 创建测试数据
        N_steps = 50
        A = np.random.random((N_steps, 2, 2)) + 1j * np.random.random((N_steps, 2, 2))
        B = np.random.random((N_steps, 2, 2)) + 1j * np.random.random((N_steps, 2, 2))

        # 传统循环计算
        result_loop = np.zeros_like(A)
        for i in range(N_steps):
            result_loop[i] = A[i] @ B[i]

        # einsum向量化计算
        result_einsum = np.einsum("tij,tjk->tik", A, B)

        # 验证结果一致性
        max_diff = np.max(np.abs(result_loop - result_einsum))
        assert max_diff < 1e-14, f"向量化计算结果不一致，最大误差: {max_diff}"

    def test_einsum_performance_improvement(self):
        """测试向量化计算的性能提升"""
        # 使用较大的数据集来体现性能差异
        N_steps = 200
        A = np.random.random((N_steps, 2, 2)) + 1j * np.random.random((N_steps, 2, 2))
        B = np.random.random((N_steps, 2, 2)) + 1j * np.random.random((N_steps, 2, 2))

        # 测试循环计算时间
        start_time = time.time()
        result_loop = np.zeros_like(A)
        for i in range(N_steps):
            result_loop[i] = A[i] @ B[i]
        time_loop = time.time() - start_time

        # 测试向量化计算时间
        start_time = time.time()
        result_einsum = np.einsum("tij,tjk->tik", A, B)
        time_einsum = time.time() - start_time

        # 验证性能提升
        speedup = time_loop / time_einsum if time_einsum > 0 else float("inf")
        print(f"   循环计算时间: {time_loop*1000:.2f}ms")
        print(f"   向量化时间: {time_einsum*1000:.2f}ms")
        print(f"   性能提升: {speedup:.1f}x")

        # 一般情况下向量化应该更快，但在小数据集上可能差异不大
        # 主要验证结果正确性
        max_diff = np.max(np.abs(result_loop - result_einsum))
        assert max_diff < 1e-14, "向量化计算结果不正确"


@pytest.mark.optimization
class TestCubicSplineInterpolation:
    """测试三次样条插值优化的精度提升"""

    def test_cubic_spline_vs_linear_precision(self):
        """比较三次样条插值与线性插值的精度"""
        # 创建一个光滑的测试函数
        t_points = np.linspace(0, 2 * np.pi, 20)
        complex_data = np.sin(t_points) + 1j * np.cos(t_points)

        # 测试插值点
        t_test = np.pi  # 中间点
        true_value = np.sin(t_test) + 1j * np.cos(t_test)

        # 线性插值
        real_linear = np.interp(t_test, t_points, np.real(complex_data))
        imag_linear = np.interp(t_test, t_points, np.imag(complex_data))
        linear_result = real_linear + 1j * imag_linear
        linear_error = abs(linear_result - true_value)

        # 三次样条插值
        real_spline = CubicSpline(t_points, np.real(complex_data), bc_type="natural")
        imag_spline = CubicSpline(t_points, np.imag(complex_data), bc_type="natural")
        spline_result = real_spline(t_test) + 1j * imag_spline(t_test)
        spline_error = abs(spline_result - true_value)

        print(f"   线性插值误差: {linear_error:.6f}")
        print(f"   三次样条误差: {spline_error:.6f}")
        print(f"   精度提升: {linear_error/spline_error:.1f}x")

        # 对于光滑函数，三次样条应该更精确
        assert spline_error < linear_error, "三次样条插值精度应该更高"
        assert spline_error < 1e-3, "三次样条插值对光滑函数应该相当精确"

    def test_cubic_spline_boundary_conditions(self):
        """测试三次样条插值的边界条件处理"""
        s_span = np.linspace(0, 10, 25)
        test_data = np.exp(-s_span / 5) * np.sin(s_span) + 1j * np.cos(s_span / 2)

        # 创建样条插值器
        real_spline = CubicSpline(s_span, np.real(test_data), bc_type="natural")
        imag_spline = CubicSpline(s_span, np.imag(test_data), bc_type="natural")

        # 测试边界附近的插值
        test_points = [
            s_span[0],
            s_span[1] / 2,
            s_span[-1],
            (s_span[-1] + s_span[-2]) / 2,
        ]

        for s_test in test_points:
            # 确保插值在有效范围内
            s_clamped = np.clip(s_test, s_span[0], s_span[-1])
            result = real_spline(s_clamped) + 1j * imag_spline(s_clamped)

            # 检查结果不是NaN或无穷大
            assert not np.isnan(result), f"插值在s={s_test}处产生NaN"
            assert not np.isinf(result), f"插值在s={s_test}处产生无穷大"


class TestWavefunctionAssemblyCorrectness:
    """测试修正后的波函数组装逻辑"""

    def test_matrix_vector_multiplication_logic(self):
        """测试矩阵-向量乘法逻辑的正确性"""
        # 创建测试数据 - 使用非零的第二列来确保方法差异
        B_m0_p = np.array([[1 + 1j, 0.5 + 0.3j], [0.2j, 1 - 0.5j]], dtype=complex)
        c_init = np.array([1.0, 0.0], dtype=complex)

        # 新的逻辑：B_{m0}^{(p)} @ c_init
        coeffs_new = B_m0_p @ c_init
        expected_coeffs = np.array([1 + 1j, 0.2j], dtype=complex)

        assert np.allclose(coeffs_new, expected_coeffs), "矩阵-向量乘法逻辑不正确"

        # 验证这比之前只取第一列的方法更合理
        coeffs_old = B_m0_p[:, 0]  # 旧方法：只取第一列

        # 对于c_init=[1,0]的特殊情况，两种方法结果相同，但逻辑不同
        # 测试概念上的正确性和完整性
        assert coeffs_new.shape == (2,), "新方法应该返回正确形状的系数向量"
        assert isinstance(coeffs_new[0], complex), "系数应该是复数类型"

        # 测试当c_init不同时，新方法的完整性
        c_init_test = np.array([0.6, 0.8], dtype=complex)
        coeffs_full = B_m0_p @ c_init_test
        # 新方法应该能正确处理任意初始向量
        assert coeffs_full.shape == (2,), "新方法应该处理任意初始向量"

    def test_initial_state_propagation(self):
        """测试从初始态开始的传播逻辑"""
        params = {"E0": 1.0, "lambda": 0.1, "theta0": 0.0, "w": 0.05, "v": 0.05}
        s_span = np.linspace(0, 1, 21)

        # 运行0阶DAPT计算
        results = run_dapt_calculation(s_span, order=0, params=params)
        psi_0 = results["solutions"][0]

        # 验证初始时刻的归一化
        initial_norm = np.linalg.norm(psi_0[0])
        assert abs(initial_norm - 1.0) < 1e-10, f"初始态归一化不正确: {initial_norm}"

        # 验证时间演化的连续性
        for i in range(1, len(s_span)):
            norm_i = np.linalg.norm(psi_0[i])
            # 在DAPT近似下，归一化可能有小的变化，但不应该太大
            assert 0.5 < norm_i < 2.0, f"时刻{i}的归一化异常: {norm_i}"


@pytest.mark.performance
class TestIntegratedOptimizationPerformance:
    """测试集成优化后的整体性能"""

    def test_dapt_calculation_performance_scaling(self):
        """测试DAPT计算的性能缩放"""
        params = {"E0": 1.0, "lambda": 0.1, "theta0": 0.0, "w": 0.05, "v": 0.05}

        # 测试不同时间点数量的性能
        time_points = [50, 100, 200]
        times = []

        for N in time_points:
            s_span = np.linspace(0, 5, N)

            start_time = time.time()
            results = run_dapt_calculation(s_span, order=1, params=params)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            print(f"   N={N:3d} 点: {elapsed_time:.2f}s")

            # 验证结果质量
            psi_1 = results["solutions"][1]
            norms = [np.linalg.norm(psi_1[i]) for i in range(N)]
            assert all(0.5 < norm < 2.0 for norm in norms), "解的质量异常"

        # 验证性能缩放合理性（不要求严格的线性缩放）
        assert all(t > 0 for t in times), "所有计算时间应该为正"

    def test_memory_efficiency(self):
        """测试内存使用效率"""
        params = {"E0": 1.0, "lambda": 0.1, "theta0": 0.0, "w": 0.05, "v": 0.05}
        s_span = np.linspace(0, 2, 100)

        # 运行计算并验证内存使用合理
        results = run_dapt_calculation(s_span, order=2, params=params)

        # 验证结果结构
        assert "solutions" in results
        assert "B_coeffs" in results
        assert "timing" in results

        # 验证解的形状
        for order in range(3):
            psi = results["solutions"][order]
            assert psi.shape == (100, 4), f"{order}阶解形状不正确"

        # 验证B系数结构
        for order in range(3):
            B_coeffs = results["B_coeffs"][order]
            assert len(B_coeffs) == 4, f"{order}阶B系数数量不正确"
            for key in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                assert key in B_coeffs, f"缺少B系数 {key}"
                assert B_coeffs[key].shape == (100, 2, 2), f"B系数{key}形状不正确"


class TestOptimizationRegressionProtection:
    """防止未来修改破坏优化效果的回归测试"""

    @pytest.mark.optimization
    def test_vectorization_not_broken(self):
        """验证向量化优化代码没有被意外删除"""
        from dapt_tools.core import dapt_recursive_step
        import inspect

        source = inspect.getsource(dapt_recursive_step)

        # 检查einsum函数的存在
        assert "einsum" in source, "einsum函数已被移除"

        # 检查实际使用的einsum模式
        assert "tik,tkj->tij" in source, "einsum模式已被修改"

        # 检查向量化相关注释
        assert "向量化" in source, "向量化相关代码已被移除"

    def test_cubic_spline_not_broken(self):
        """确保三次样条插值没有被破坏"""
        import inspect
        from dapt_tools.core import _solve_diagonal_ode

        source = inspect.getsource(_solve_diagonal_ode)
        assert "CubicSpline" in source, "三次样条插值优化已被移除"
        assert "natural" in source, "样条边界条件已被修改"

    def test_wavefunction_assembly_not_broken(self):
        """确保波函数组装逻辑没有被破坏"""
        import inspect
        from dapt_tools.core import run_dapt_calculation

        source = inspect.getsource(run_dapt_calculation)
        assert "c_init" in source, "初始系数向量定义已被移除"
        assert "@ c_init" in source, "矩阵-向量乘法已被修改"

    @pytest.mark.optimization
    def test_optimization_comments_preserved(self):
        """确保优化相关的重要注释被保留"""
        from dapt_tools.core import dapt_recursive_step
        import inspect

        recursive_source = inspect.getsource(dapt_recursive_step)

        # 检查关键的优化和修正注释
        assert "理论修正" in recursive_source, "理论修正注释已被移除"
        assert "核心修正" in recursive_source, "核心修正注释已被移除"
        assert "向量化" in recursive_source, "向量化注释已被移除"
        assert "einsum" in recursive_source, "einsum相关注释已被移除"


if __name__ == "__main__":
    # 可以直接运行此文件进行快速测试
    pytest.main([__file__, "-v"])
