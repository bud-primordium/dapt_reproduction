#!/usr/bin/env python3
"""
test_exact_solver.py

精确薛定谔方程求解器的单元测试

Author: Gilbert Young
Date: 2025-06-07
"""

import numpy as np
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dapt_tools.exact_solver import (
    solve_schrodinger_exact,
    calculate_exact_energy_expectation,
    verify_exact_solution_properties,
)
from dapt_tools.hamiltonian import (
    get_initial_state_in_standard_basis,
    get_hamiltonian,
)


class TestSolveSchrodingerExact:
    """测试精确薛定谔方程求解函数"""

    def test_exact_solution_normalization(self):
        """测试精确解的归一化保持"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.5,
            "hbar": 1.0,
            "v": 0.5,
        }

        s_span = np.linspace(0, 1, 21)
        initial_state = get_initial_state_in_standard_basis(params)

        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 检查每个时刻的归一化
        for i, s in enumerate(s_span):
            norm = np.linalg.norm(exact_solution[i])
            assert np.isclose(
                norm, 1.0, atol=1e-10
            ), f"s={s:.2f}时精确解未归一化，范数：{norm}"

    def test_exact_solution_initial_condition(self):
        """测试精确解的初始条件"""
        params = {
            "E0": 1.0,
            "lambda": 0.5,
            "theta0": 0.2,
            "w": 0.3,
            "hbar": 1.0,
            "v": 0.8,
        }

        s_span = np.linspace(0, 1, 11)
        initial_state = get_initial_state_in_standard_basis(params)

        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 检查初始条件
        initial_diff = np.linalg.norm(exact_solution[0] - initial_state)
        assert initial_diff < 1e-14, f"精确解初始条件不匹配，差异：{initial_diff}"

    def test_exact_solution_time_independent_case(self):
        """测试时间无关情况的精确解"""
        params = {
            "E0": 2.0,
            "lambda": 0.0,  # 无时变
            "theta0": 0.0,  # 无相位变化
            "w": 0.0,
            "hbar": 1.0,
            "v": 1.0,
        }

        s_span = np.linspace(0, 0.5, 6)  # 短时间以避免相位累积
        initial_state = get_initial_state_in_standard_basis(params)

        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 时间无关系统中，波函数应该有规律的相位演化
        # 检查模长保持不变
        initial_abs = np.abs(exact_solution[0])
        for i in range(1, len(s_span)):
            current_abs = np.abs(exact_solution[i])
            assert np.allclose(
                current_abs, initial_abs, atol=1e-12
            ), f"时间无关情况下波函数模长发生变化"

    def test_exact_solution_energy_conservation(self):
        """测试精确解的能量守恒（绝热情况）"""
        params = {
            "E0": 1.0,
            "lambda": 0.1,  # 小的时变
            "theta0": 0.05,
            "w": 0.1,
            "hbar": 1.0,
            "v": 0.1,  # 绝热参数
        }

        s_span = np.linspace(0, 1, 51)
        initial_state = get_initial_state_in_standard_basis(params)

        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 计算能量期望值
        energy_expectation = calculate_exact_energy_expectation(
            exact_solution, s_span, params
        )

        # 在绝热极限下，能量变化应该很小
        energy_variation = np.max(energy_expectation) - np.min(energy_expectation)
        assert energy_variation < 0.2, f"绝热极限下能量变化过大：{energy_variation}"


class TestCalculateExactEnergyExpectation:
    """测试精确解能量期望值计算"""

    def test_energy_expectation_basic(self):
        """测试能量期望值计算的基本功能"""
        params = {
            "E0": 1.5,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.2,
            "hbar": 1.0,
            "v": 0.5,
        }

        s_span = np.linspace(0, 1, 11)
        initial_state = get_initial_state_in_standard_basis(params)

        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)
        energy_expectation = calculate_exact_energy_expectation(
            exact_solution, s_span, params
        )

        # 检查输出形状
        assert energy_expectation.shape == (len(s_span),), "能量期望值数组形状不正确"

        # 检查数值合理性
        assert np.all(np.isfinite(energy_expectation)), "能量期望值包含无穷大或NaN"

        # 对于恒定哈密顿量，初始基态的能量期望值应该接近基态能量
        initial_energy = energy_expectation[0]
        assert abs(initial_energy - (-params["E0"])) < 0.1, "初始能量期望值不合理"

    def test_energy_expectation_ground_state(self):
        """测试基态的能量期望值"""
        params = {
            "E0": 2.0,
            "lambda": 0.0,
            "theta0": 0.0,
            "w": 0.0,
            "hbar": 1.0,
            "v": 1.0,
        }

        # 手动构造基态
        _, eigenvectors = get_hamiltonian(
            0.0, params
        ), get_initial_state_in_standard_basis(params)
        ground_state = get_initial_state_in_standard_basis(params)

        # 单时间点
        s_span = np.array([0.0])
        exact_solution = np.array([ground_state])

        energy_expectation = calculate_exact_energy_expectation(
            exact_solution, s_span, params
        )

        # 基态的能量期望值应该等于基态能量
        expected_energy = -params["E0"]
        assert np.isclose(
            energy_expectation[0], expected_energy, atol=1e-10
        ), f"基态能量期望值错误，期望：{expected_energy}，实际：{energy_expectation[0]}"


class TestVerifyExactSolutionProperties:
    """测试精确解性质验证函数"""

    def test_verification_valid_solution(self):
        """测试有效解的验证"""
        params = {
            "E0": 1.0,
            "lambda": 0.0,
            "theta0": 0.1,
            "w": 0.2,
            "hbar": 1.0,
            "v": 0.5,
        }

        s_span = np.linspace(0, 1, 11)
        initial_state = get_initial_state_in_standard_basis(params)

        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)
        verification = verify_exact_solution_properties(exact_solution, s_span, params)

        # 检查返回字典的结构
        required_keys = [
            "normalization",
            "energy",
            "numerical_stability",
            "overall_valid",
        ]
        for key in required_keys:
            assert key in verification, f"验证结果缺少键：{key}"

        # 检查归一化
        assert verification["normalization"]["is_conserved"], "归一化验证失败"
        assert verification["normalization"]["max_error"] < 1e-10, "归一化误差过大"

        # 检查数值稳定性
        assert verification["numerical_stability"]["is_stable"], "数值稳定性验证失败"
        assert not verification["numerical_stability"]["contains_nan"], "解包含NaN"
        assert not verification["numerical_stability"]["contains_inf"], "解包含无穷大"

        # 检查整体有效性
        assert verification["overall_valid"], "整体验证失败"

    def test_verification_invalid_solution(self):
        """测试无效解的验证"""
        # 构造包含NaN的无效解
        s_span = np.linspace(0, 1, 5)
        params = {"E0": 1.0, "lambda": 0.0, "theta0": 0.0, "w": 0.0}

        invalid_solution = np.ones((len(s_span), 4), dtype=complex)
        invalid_solution[2, 1] = np.nan  # 插入NaN

        verification = verify_exact_solution_properties(
            invalid_solution, s_span, params
        )

        # 应该检测到无效性
        assert verification["numerical_stability"]["contains_nan"], "未检测到NaN"
        assert not verification["numerical_stability"]["is_stable"], "错误地认为解稳定"
        assert not verification["overall_valid"], "错误地认为解有效"

    def test_verification_unnormalized_solution(self):
        """测试非归一化解的验证"""
        s_span = np.linspace(0, 1, 5)
        params = {"E0": 1.0, "lambda": 0.0, "theta0": 0.0, "w": 0.0}

        # 构造非归一化解
        unnormalized_solution = np.ones((len(s_span), 4), dtype=complex) * 2.0

        verification = verify_exact_solution_properties(
            unnormalized_solution, s_span, params, tolerance=1e-10
        )

        # 应该检测到归一化问题
        assert not verification["normalization"]["is_conserved"], "未检测到归一化问题"
        assert verification["normalization"]["max_error"] > 1e-10, "归一化误差计算错误"


class TestExactSolverIntegration:
    """精确解模块的集成测试"""

    def test_exact_solver_workflow(self):
        """测试完整的精确求解工作流程"""
        params = {
            "E0": 1.5,
            "lambda": 0.2,
            "theta0": 0.1,
            "w": 0.3,
            "hbar": 1.0,
            "v": 0.4,
        }

        s_span = np.linspace(0, 1, 21)
        initial_state = get_initial_state_in_standard_basis(params)

        # 步骤1：求解精确解
        exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

        # 步骤2：计算能量期望值
        energy_expectation = calculate_exact_energy_expectation(
            exact_solution, s_span, params
        )

        # 步骤3：验证解的性质
        verification = verify_exact_solution_properties(exact_solution, s_span, params)

        # 检查整个工作流程的一致性
        assert exact_solution.shape == (len(s_span), 4), "精确解形状不正确"
        assert energy_expectation.shape == (len(s_span),), "能量期望值形状不正确"
        assert verification["overall_valid"], "解的整体验证失败"

        # 检查物理合理性
        initial_energy = energy_expectation[0]
        final_energy = energy_expectation[-1]
        energy_change = abs(final_energy - initial_energy)

        print(f"初始能量: {initial_energy:.6f}")
        print(f"最终能量: {final_energy:.6f}")
        print(f"能量变化: {energy_change:.6f}")

        # 在这个参数范围内，能量变化应该是合理的
        assert energy_change < 2.0, "能量变化过大，可能存在数值问题"


@pytest.mark.parametrize("v_value", [0.1, 0.5, 1.0, 2.0])
def test_exact_solution_different_velocities(v_value):
    """参数化测试：不同演化速度下的精确解"""
    params = {
        "E0": 1.0,
        "lambda": 0.0,
        "theta0": 0.1,
        "w": 0.2,
        "hbar": 1.0,
        "v": v_value,
    }

    s_span = np.linspace(0, 1, 11)
    initial_state = get_initial_state_in_standard_basis(params)

    exact_solution = solve_schrodinger_exact(s_span, params, initial_state)

    # 基本检查
    assert exact_solution.shape == (len(s_span), 4)
    assert not np.any(np.isnan(exact_solution))
    assert not np.any(np.isinf(exact_solution))

    # 归一化检查
    norms = [np.linalg.norm(exact_solution[i]) for i in range(len(s_span))]
    for i, norm in enumerate(norms):
        assert np.isclose(
            norm, 1.0, atol=1e-10
        ), f"v={v_value}, s={s_span[i]:.2f}时归一化失败"


if __name__ == "__main__":
    pytest.main([__file__])
