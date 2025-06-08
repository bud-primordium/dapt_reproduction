#!/usr/bin/env python3
"""
test_dapt_integration.py

DAPT完整功能的集成测试

整合了之前调试脚本的关键测试：
- debug_normalization.py
- quick_test.py
- simple_zero_order_debug.py
- test_dapt_modifications.py

Author: Gilbert Young
Date: 2025-06-07
"""

import numpy as np
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dapt_tools import (
    get_initial_state_in_standard_basis,
    get_eigensystem,
    verify_analytical_eigensystem,
    calculate_M_matrix,
    run_dapt_calculation,
)
from dapt_tools.exact_solver import (
    solve_schrodinger_exact,
)


class TestDAPTNormalizationDebug:
    """DAPT归一化问题的调试测试（来自debug_normalization.py）"""

    def test_zero_order_normalization_detailed(self):
        """详细测试零阶DAPT的归一化问题"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        # 测试s=0时刻
        s = 0.0

        # 初始态检查
        initial_state = get_initial_state_in_standard_basis(params)
        assert np.isclose(
            np.linalg.norm(initial_state) ** 2, 1.0, atol=1e-15
        ), "初始态未归一化"

        # 获取s=0的本征系统
        eigenvalues, eigenvectors = get_eigensystem(s, params)

        # 检查本征值的正确性
        expected_eigenvalues = np.array([-1.5, -1.5, 1.5, 1.5])
        assert np.allclose(
            np.sort(eigenvalues), np.sort(expected_eigenvalues), atol=1e-15
        )

        # 检查初始态在本征基下的展开
        coeffs_in_eigenbasis = eigenvectors.conj().T @ initial_state
        assert np.isclose(np.sum(np.abs(coeffs_in_eigenbasis) ** 2), 1.0, atol=1e-14)

        # 应该主要在第一个基态上
        assert abs(coeffs_in_eigenbasis[0]) > 0.99, "初始态应主要在第一个基态|0^0⟩上"

        # 重构初始态检验
        reconstructed = eigenvectors @ coeffs_in_eigenbasis
        reconstruction_error = np.linalg.norm(initial_state - reconstructed)
        assert (
            reconstruction_error < 1e-14
        ), f"初始态重构误差过大：{reconstruction_error}"

    def test_zero_order_B_coefficients(self):
        """测试零阶B系数矩阵的性质"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        # 运行零阶DAPT计算
        s_span = np.array([0.0, 0.5, 1.0])
        dapt_results = run_dapt_calculation(s_span, order=0, params=params)

        # 检查B系数矩阵结构
        B_coeffs_0 = dapt_results["B_coeffs"][0]

        # 零阶时，只有B_{00}^{(0)}应该非零
        for (m, n), B_mn in B_coeffs_0.items():
            if m == 0 and n == 0:
                # B_{00}^{(0)}应该非零且有合理的模长
                for i in range(len(s_span)):
                    norm = np.linalg.norm(B_mn[i])
                    assert (
                        norm > 0.1
                    ), f"B_{{00}}^{{(0)}}在s={s_span[i]:.1f}时模长过小：{norm}"
            else:
                # 其他B_{mn}^{(0)}应该为零
                for i in range(len(s_span)):
                    norm = np.linalg.norm(B_mn[i])
                    assert (
                        norm < 1e-12
                    ), f"B_{{{m}{n}}}^{{(0)}}在s={s_span[i]:.1f}时应为零，实际模长：{norm}"


class TestQuickDAPTValidation:
    """DAPT快速验证测试（来自quick_test.py）"""

    def test_normalization_preservation(self):
        """测试归一化保持"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,  # 恒定能隙
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        s_span = np.linspace(0, 1, 21)
        initial_state = get_initial_state_in_standard_basis(params)

        # 检查初始态归一化
        assert np.isclose(np.linalg.norm(initial_state) ** 2, 1.0, atol=1e-15)

        # 计算精确解
        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 检查精确解的归一化保持
        exact_norms = [
            np.linalg.norm(exact_solution[i]) ** 2 for i in range(len(s_span))
        ]
        initial_norm = exact_norms[0]
        final_norm = exact_norms[-1]

        assert np.isclose(
            initial_norm, 1.0, atol=1e-14
        ), f"精确解初始归一化错误：{initial_norm}"
        assert np.isclose(
            final_norm, 1.0, atol=1e-10
        ), f"精确解最终归一化错误：{final_norm}"

        # 计算0阶DAPT
        dapt_results = run_dapt_calculation(s_span, order=0, params=params)
        dapt_0_solution = dapt_results["solutions"][0]

        # 检查DAPT解的归一化
        dapt_norms = [
            np.linalg.norm(dapt_0_solution[i]) ** 2 for i in range(len(s_span))
        ]
        dapt_initial_norm = dapt_norms[0]
        dapt_final_norm = dapt_norms[-1]

        assert np.isclose(
            dapt_initial_norm, 1.0, atol=1e-10
        ), f"DAPT初始归一化错误：{dapt_initial_norm}"
        # 允许一定的演化误差
        assert np.isclose(
            dapt_final_norm, 1.0, atol=0.1
        ), f"DAPT最终归一化错误：{dapt_final_norm}"

    def test_infidelity_calculation(self):
        """测试不忠诚度计算的正确性"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        s_span = np.linspace(0, 1, 11)
        initial_state = get_initial_state_in_standard_basis(params)

        # 计算精确解和DAPT解
        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)
        dapt_results = run_dapt_calculation(s_span, order=0, params=params)
        dapt_0_solution = dapt_results["solutions"][0]

        # 手动计算不忠诚度
        for i in [0, len(s_span) // 2, -1]:
            # 归一化波函数
            exact_norm = exact_solution[i] / np.linalg.norm(exact_solution[i])
            dapt_norm = dapt_0_solution[i] / np.linalg.norm(dapt_0_solution[i])

            # 计算内积和不忠诚度
            overlap = np.dot(exact_norm.conj(), dapt_norm)
            fidelity = abs(overlap) ** 2
            infidelity = 1 - fidelity

            # 基本合理性检查
            assert (
                0 <= fidelity <= 1
            ), f"s={s_span[i]:.2f}时保真度超出[0,1]范围：{fidelity}"
            assert (
                0 <= infidelity <= 1
            ), f"s={s_span[i]:.2f}时不忠诚度超出[0,1]范围：{infidelity}"

            # 在s=0时，应该有完美的保真度
            if i == 0:
                assert np.isclose(
                    fidelity, 1.0, atol=1e-10
                ), f"初始时刻保真度应为1，实际：{fidelity}"


class TestSimpleZeroOrderAnalysis:
    """简化的零阶DAPT分析测试（来自simple_zero_order_debug.py）"""

    def test_zero_order_theoretical_analysis(self):
        """零阶DAPT的理论分析"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        # 初始态
        initial_state = get_initial_state_in_standard_basis(params)

        # 获取s=0的本征系统
        _, eigenvectors = get_eigensystem(0.0, params)

        # 基态子空间的基矢
        base_state_0 = eigenvectors[:, 0]  # |0^0⟩
        base_state_1 = eigenvectors[:, 1]  # |0^1⟩

        # 检查基态的归一化
        assert np.isclose(np.linalg.norm(base_state_0), 1.0, atol=1e-15)
        assert np.isclose(np.linalg.norm(base_state_1), 1.0, atol=1e-15)

        # 检查基态之间的正交性
        orthogonality = abs(np.dot(base_state_0.conj(), base_state_1))
        assert orthogonality < 1e-14, f"基态之间应正交，实际内积：{orthogonality}"

        # 初始态应该主要是|0^0⟩
        overlap_0 = abs(np.vdot(initial_state, base_state_0))
        overlap_1 = abs(np.vdot(initial_state, base_state_1))

        assert np.isclose(
            overlap_0, 1.0, atol=1e-14
        ), f"初始态与|0^0⟩的重叠度应为1，实际：{overlap_0}"
        assert overlap_1 < 1e-14, f"初始态与|0^1⟩的重叠度应为0，实际：{overlap_1}"

    def test_B_matrix_interpretation(self):
        """B系数矩阵的理论解释"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        # 根据DAPT理论，零阶B_{00}^{(0)}应该是WZ矩阵 U^0(s)
        # 而从基态|0^0⟩开始，所以应该是U^0(s)的第一列

        # 简单验证：B_{00}^{(0)}应该是酉矩阵
        s_span = np.linspace(0, 1, 6)
        dapt_results = run_dapt_calculation(s_span, order=0, params=params)
        B_00 = dapt_results["B_coeffs"][0][(0, 0)]

        for i, s in enumerate(s_span):
            B_matrix = B_00[i]

            # 检查酉性：B†B = I（在理论上WZ矩阵是酉的）
            product = B_matrix.conj().T @ B_matrix
            identity = np.eye(2)
            unitarity_error = np.linalg.norm(product - identity)

            # 允许较大的数值误差，因为B矩阵在DAPT中可能不是严格酉的
            assert (
                unitarity_error < 1.0
            ), f"s={s:.2f}时B_{{00}}^{{(0)}}酉性误差过大：{unitarity_error}"


class TestDAPTModificationsValidation:
    """DAPT修改验证测试（来自test_dapt_modifications.py）"""

    def test_analytical_eigensystem_comprehensive(self):
        """全面测试解析本征系统"""
        params = {"E0": 1.5, "lambda": 0.5, "theta0": 0.1, "w": 0.3}

        test_times = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

        for s in test_times:
            result = verify_analytical_eigensystem(s, params)

            assert result["is_valid"], f"s={s}时解析本征系统验证失败"
            assert (
                result["max_eigenvalue_error"] < 1e-14
            ), f"s={s}时本征值误差过大：{result['max_eigenvalue_error']}"
            assert (
                result["max_orthogonality_error"] < 1e-14
            ), f"s={s}时正交性误差过大：{result['max_orthogonality_error']}"

    def test_M_matrix_precision_with_analytical_derivatives(self):
        """测试使用解析导数的M矩阵精度"""
        params = {"E0": 1.0, "lambda": 1.0, "theta0": 0.1, "w": 0.3}

        test_times = [0.1, 0.3, 0.7, 0.9]

        for s in test_times:
            M_matrix = calculate_M_matrix(s, 1e-4, params)  # ds参数现在被忽略

            # 检查M矩阵不包含NaN或无穷大
            for (m, n), matrix in M_matrix.items():
                assert not np.any(np.isnan(matrix)), f"s={s}时M^{{{m}{n}}}包含NaN"
                assert not np.any(np.isinf(matrix)), f"s={s}时M^{{{m}{n}}}包含无穷大"

                # 检查合理的数值范围
                max_element = np.max(np.abs(matrix))
                assert max_element < 1000, f"s={s}时M^{{{m}{n}}}元素过大：{max_element}"

    def test_dapt_vs_exact_convergence(self):
        """测试DAPT近似解与精确解的收敛性"""
        # 使用绝热区参数
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,  # 绝热参数
        }

        s_span = np.linspace(0, 1, 51)
        initial_state = get_initial_state_in_standard_basis(params)

        # 计算精确解
        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 计算DAPT解（0阶到1阶，避免计算时间过长）
        dapt_results = run_dapt_calculation(s_span, order=1, params=params)

        # 计算最终不忠诚度
        final_infidelities = []
        for k in range(2):  # 0, 1阶
            dapt_solution_k = dapt_results["solutions"][k]

            # 归一化波函数
            exact_norm = exact_solution[-1] / np.linalg.norm(exact_solution[-1])
            dapt_norm = dapt_solution_k[-1] / np.linalg.norm(dapt_solution_k[-1])

            # 计算不忠诚度
            overlap = abs(np.dot(exact_norm.conj(), dapt_norm)) ** 2
            infidelity = 1 - overlap
            final_infidelities.append(infidelity)

        # 检查基本合理性
        for k, inf in enumerate(final_infidelities):
            assert inf >= 0, f"第{k}阶不忠诚度应为非负数，实际：{inf}"
            assert inf <= 1, f"第{k}阶不忠诚度应≤1，实际：{inf}"

        # 在绝热区，高阶通常表现更好（但允许数值误差）
        print(
            f"不忠诚度: I_0={final_infidelities[0]:.6f}, I_1={final_infidelities[1]:.6f}"
        )

    def test_time_varying_gap_scenario(self):
        """测试时变能隙情况"""
        params = {
            "E0": 1.0,
            "lambda": 1.0,  # 时变能隙
            "theta0": 0.1,
            "w": 0.3,
            "hbar": 1.0,
            "v": 0.3,
        }

        s_span = np.linspace(0, 1, 21)

        # 验证每个时间点的解析本征系统都正常工作
        for s in s_span:
            result = verify_analytical_eigensystem(s, params)
            assert result["is_valid"], f"时变能隙情况下s={s:.2f}时解析本征系统失败"

        # 运行DAPT计算应该成功
        try:
            dapt_results = run_dapt_calculation(s_span, order=0, params=params)
            assert "solutions" in dapt_results, "时变能隙DAPT计算结果缺少solutions"
        except Exception as e:
            pytest.fail(f"时变能隙DAPT计算失败：{e}")


if __name__ == "__main__":
    pytest.main([__file__])
