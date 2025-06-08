"""
test_hamiltonian.py

DAPT哈密顿量模块的单元测试

Author: Gilbert Young
Date: 2025-06-07
"""

import numpy as np
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dapt_tools.hamiltonian import (
    get_hamiltonian,
    get_eigensystem,
    get_eigenvector_derivatives,
    verify_analytical_eigensystem,
    calculate_energy_gap,
    get_initial_state_in_standard_basis,
)


class TestGetHamiltonian:
    """测试哈密顿量构造函数"""

    def test_hamiltonian_shape_and_type(self):
        """测试哈密顿量的形状和数据类型"""
        params = {"E0": 1.5, "lambda": 0.0, "theta0": 0.1, "w": 0.5}

        H = get_hamiltonian(0.5, params)

        # 检查形状
        assert H.shape == (4, 4), f"哈密顿量形状应为(4,4)，实际为{H.shape}"

        # 检查数据类型
        assert H.dtype == np.complex128, f"哈密顿量应为复数类型，实际为{H.dtype}"

    def test_hamiltonian_hermiticity(self):
        """测试哈密顿量的厄米性"""
        params = {"E0": 1.0, "lambda": 0.5, "theta0": 0.2, "w": 0.3}

        # 测试多个时间点
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            H = get_hamiltonian(s, params)
            H_dagger = H.conj().T

            # 检查厄米性：H = H†
            assert np.allclose(
                H, H_dagger, atol=1e-15
            ), f"哈密顿量在s={s}时不满足厄米性"

    def test_hamiltonian_structure(self):
        """测试哈密顿量的块矩阵结构"""
        params = {"E0": 1.0, "lambda": 0.0, "theta0": 0.0, "w": 0.0}

        H = get_hamiltonian(0.5, params)

        # 检查左上和右下块应为零矩阵
        assert np.allclose(H[:2, :2], 0.0, atol=1e-15), "左上2×2块应为零矩阵"
        assert np.allclose(H[2:, 2:], 0.0, atol=1e-15), "右下2×2块应为零矩阵"

        # 检查非对角块的关系：H[2:,:2] = H[:2,2:]†
        assert np.allclose(
            H[2:, :2], H[:2, 2:].conj().T, atol=1e-15
        ), "非对角块不满足厄米关系"

    def test_constant_gap_case(self):
        """测试恒定能隙情况"""
        params = {"E0": 2.0, "lambda": 0.0, "theta0": 0.0, "w": 0.0}  # 恒定能隙

        H1 = get_hamiltonian(0.0, params)
        H2 = get_hamiltonian(0.5, params)
        H3 = get_hamiltonian(1.0, params)

        # 恒定参数时，哈密顿量应该相同（除了相位因子）
        # 这里我们检查哈密顿量的绝对值结构是否一致
        assert np.allclose(
            np.abs(H1), np.abs(H2), atol=1e-15
        ), "恒定能隙情况下哈密顿量结构应保持一致"


class TestAnalyticalEigensystem:
    """测试解析本征系统实现"""

    def test_analytical_eigenvalues_precision(self):
        """测试解析本征值的精度"""
        params = {"E0": 1.5, "lambda": 0.0, "theta0": 0.1, "w": 0.5}

        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            eigenvalues, _ = get_eigensystem(s, params)

            # 检查本征值数量
            assert len(eigenvalues) == 4, f"应有4个本征值，实际为{len(eigenvalues)}"

            # 检查简并性：应有两个-E(s)和两个+E(s)
            E_s = params["E0"] + params["lambda"] * (s - 0.5) ** 2
            expected_eigenvalues = np.array([-E_s, -E_s, E_s, E_s])

            assert np.allclose(
                np.sort(eigenvalues), np.sort(expected_eigenvalues), atol=1e-15
            ), f"s={s}时本征值不符合预期。预期：{expected_eigenvalues}，实际：{eigenvalues}"

    def test_analytical_eigenvectors_orthonormality(self):
        """测试解析本征矢量的正交归一性"""
        params = {"E0": 1.0, "lambda": 0.5, "theta0": 0.2, "w": 0.3}

        for s in [0.0, 0.5, 1.0]:
            _, eigenvectors = get_eigensystem(s, params)

            # 计算重叠矩阵
            overlap_matrix = eigenvectors.conj().T @ eigenvectors
            identity = np.eye(4)

            # 检查正交归一性
            assert np.allclose(
                overlap_matrix, identity, atol=1e-15
            ), f"s={s}时本征矢量不满足正交归一性，最大偏差：{np.max(np.abs(overlap_matrix - identity))}"

    def test_verify_analytical_eigensystem_function(self):
        """测试解析本征系统验证函数"""
        params = {"E0": 1.5, "lambda": 0.5, "theta0": 0.1, "w": 0.3}

        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = verify_analytical_eigensystem(s, params)

            assert result["is_valid"], f"s={s}时解析本征系统验证失败"
            assert result["max_eigenvalue_error"] < 1e-14, f"s={s}时本征值误差过大"
            assert result["max_orthogonality_error"] < 1e-14, f"s={s}时正交性误差过大"

    def test_eigenvector_derivatives(self):
        """测试解析本征矢量导数"""
        params = {"E0": 1.0, "lambda": 1.0, "theta0": 0.1, "w": 0.5}

        for s in [0.1, 0.5, 0.9]:
            derivatives = get_eigenvector_derivatives(s, params)

            # 检查导数矩阵的形状
            assert derivatives.shape == (
                4,
                4,
            ), f"导数矩阵形状应为(4,4)，实际为{derivatives.shape}"

            # 检查导数矩阵的数据类型
            assert derivatives.dtype == np.complex128, "导数矩阵应为复数类型"

            # 导数矩阵不应包含NaN或无穷大
            assert not np.any(np.isnan(derivatives)), f"s={s}时导数矩阵包含NaN"
            assert not np.any(np.isinf(derivatives)), f"s={s}时导数矩阵包含无穷大"

    def test_eigenvector_continuity(self):
        """测试本征矢量的自动连续性"""
        params = {"E0": 1.0, "lambda": 0.5, "theta0": 0.1, "w": 0.2}

        s_values = np.linspace(0, 1, 21)
        min_overlap = 1.0

        prev_eigenvectors = None
        for s in s_values:
            _, eigenvectors = get_eigensystem(s, params)

            if prev_eigenvectors is not None:
                # 计算相邻时刻的重叠度
                for i in range(4):
                    overlap = abs(
                        np.dot(prev_eigenvectors[:, i].conj(), eigenvectors[:, i])
                    )
                    min_overlap = min(min_overlap, overlap)

            prev_eigenvectors = eigenvectors.copy()

        # 解析公式应该自动保证连续性
        assert min_overlap > 0.999, f"解析本征矢量连续性不佳，最小重叠度：{min_overlap}"


class TestCalculateEnergyGap:
    """测试能隙计算函数"""

    def test_energy_gap_constant(self):
        """测试恒定能隙情况"""
        params = {"E0": 2.0, "lambda": 0.0, "theta0": 0.1, "w": 0.5}

        gap = calculate_energy_gap(0.5, params)
        expected_gap = 2 * params["E0"]  # 能隙应为2E0

        assert np.isclose(
            gap, expected_gap, atol=1e-12
        ), f"恒定能隙计算错误。预期：{expected_gap}，实际：{gap}"

    def test_energy_gap_time_dependent(self):
        """测试时变能隙情况"""
        params = {"E0": 1.0, "lambda": 1.0, "theta0": 0.0, "w": 0.0}

        # 在s=0.5时，E(s)应达到最小值E0
        gap_min = calculate_energy_gap(0.5, params)
        expected_gap_min = 2 * params["E0"]

        assert np.isclose(
            gap_min, expected_gap_min, atol=1e-12
        ), f"最小能隙计算错误。预期：{expected_gap_min}，实际：{gap_min}"

        # 在s=0或s=1时，能隙应更大
        gap_edge = calculate_energy_gap(0.0, params)
        E_edge = params["E0"] + params["lambda"] * (0.0 - 0.5) ** 2
        expected_gap_edge = 2 * E_edge

        assert np.isclose(
            gap_edge, expected_gap_edge, atol=1e-12
        ), f"边界能隙计算错误。预期：{expected_gap_edge}，实际：{gap_edge}"

        # 确保边界能隙大于最小能隙
        assert gap_edge > gap_min, "边界能隙应大于最小能隙"


class TestGetInitialState:
    """测试初始态构造函数"""

    def test_initial_state_normalization(self):
        """测试初始态的归一化"""
        params = {"E0": 1.5, "lambda": 0.0, "theta0": 0.1, "w": 0.5}

        initial_state = get_initial_state_in_standard_basis(params)

        # 检查归一化
        norm = np.linalg.norm(initial_state)
        assert np.isclose(norm, 1.0, atol=1e-15), f"初始态未归一化，范数为：{norm}"

    def test_initial_state_structure(self):
        """测试初始态的结构"""
        params = {"E0": 1.0, "lambda": 0.0, "theta0": 0.0, "w": 0.0}  # 零相位

        initial_state = get_initial_state_in_standard_basis(params)
        expected = np.array([0.5, 0.5, 0.0, -np.sqrt(2) / 2], dtype=complex)

        assert np.allclose(
            initial_state, expected, atol=1e-15
        ), f"零相位时初始态结构错误。预期：{expected}，实际：{initial_state}"

    def test_initial_state_phase_dependence(self):
        """测试初始态的相位依赖性"""
        params1 = {"E0": 1.0, "lambda": 0.0, "theta0": 0.0, "w": 0.0}
        params2 = {"E0": 1.0, "lambda": 0.0, "theta0": np.pi / 2, "w": 0.0}

        state1 = get_initial_state_in_standard_basis(params1)
        state2 = get_initial_state_in_standard_basis(params2)

        # 相位变化应该影响初始态
        assert not np.allclose(
            state1, state2, atol=1e-10
        ), "不同初始相位应产生不同的初始态"

        # 但两者都应该归一化
        assert np.isclose(np.linalg.norm(state1), 1.0, atol=1e-15)
        assert np.isclose(np.linalg.norm(state2), 1.0, atol=1e-15)

    def test_initial_state_consistency_with_eigenvectors(self):
        """测试初始态与本征矢量的一致性"""
        params = {"E0": 1.5, "lambda": 0.0, "theta0": 0.1, "w": 0.5}

        initial_state = get_initial_state_in_standard_basis(params)
        _, eigenvectors = get_eigensystem(0.0, params)

        # 初始态应该是基态|0^0(0)⟩
        ground_state = eigenvectors[:, 0]

        # 检查一致性（允许全局相位差）
        overlap = abs(np.dot(initial_state.conj(), ground_state))
        assert np.isclose(
            overlap, 1.0, atol=1e-14
        ), f"初始态与基态本征矢量不一致，重叠度：{overlap}"


@pytest.mark.parametrize("s_value", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_hamiltonian_at_different_times(s_value):
    """参数化测试：不同时间点的哈密顿量"""
    params = {"E0": 1.0, "lambda": 0.5, "theta0": 0.1, "w": 0.3}

    H = get_hamiltonian(s_value, params)

    # 基本检查
    assert H.shape == (4, 4)
    assert H.dtype == np.complex128
    assert np.allclose(H, H.conj().T, atol=1e-15)  # 厄米性


@pytest.mark.parametrize("E0,lambda_val", [(1.0, 0.0), (1.5, 0.5), (2.0, 1.0)])
def test_energy_parameters(E0, lambda_val):
    """参数化测试：不同能量参数"""
    params = {"E0": E0, "lambda": lambda_val, "theta0": 0.1, "w": 0.2}

    eigenvalues, _ = get_eigensystem(0.5, params)

    # 检查本征值与参数的关系
    E_s = E0 + lambda_val * (0.5 - 0.5) ** 2  # = E0
    expected = [-E_s, -E_s, E_s, E_s]

    assert np.allclose(np.sort(eigenvalues), np.sort(expected), atol=1e-15)


class TestZeroOrderDAPTDebug:
    """整合的零阶DAPT调试测试"""

    def test_zero_order_normalization(self):
        """测试零阶DAPT的归一化问题（来自debug_normalization.py）"""
        from dapt_tools import run_dapt_calculation

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

        # 初始态
        initial_state = get_initial_state_in_standard_basis(params)
        assert np.isclose(np.linalg.norm(initial_state) ** 2, 1.0, atol=1e-15)

        # 获取s=0的本征系统
        eigenvalues, eigenvectors = get_eigensystem(s, params)

        # 检查初始态在本征基下的展开
        coeffs_in_eigenbasis = eigenvectors.conj().T @ initial_state
        assert np.isclose(np.sum(np.abs(coeffs_in_eigenbasis) ** 2), 1.0, atol=1e-14)

        # 应该主要在第一个基态上
        assert abs(coeffs_in_eigenbasis[0]) > 0.99, "初始态应主要在第一个基态上"

    def test_zero_order_simple_analysis(self):
        """简化的零阶DAPT分析测试（来自simple_zero_order_debug.py）"""
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

        # 初始态应该就是|0^0⟩
        overlap = np.abs(np.vdot(initial_state, base_state_0))
        assert np.isclose(
            overlap, 1.0, atol=1e-14
        ), f"初始态与|0^0⟩的重叠度应为1，实际：{overlap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
