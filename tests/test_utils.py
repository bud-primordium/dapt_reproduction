"""
test_utils.py

DAPT工具函数模块的单元测试

Author: Gilbert Young
Date: 2025-06-07
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
import sys
import os
from unittest.mock import patch, MagicMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dapt_tools.utils import (
    calculate_infidelity,
    calculate_infidelity_series,
    plot_infidelity_comparison,
    calculate_epsilon_parameter,
    format_scientific_notation,
    save_results_to_file,
    load_results_from_file,
    setup_matplotlib_style,
)


class TestCalculateInfidelity:
    """测试不忠诚度计算函数"""

    def test_identical_states(self):
        """测试相同态的不忠诚度"""
        # 创建归一化的随机态
        psi = np.array([1, 1j, -1, 2], dtype=complex)
        psi = psi / np.linalg.norm(psi)

        infidelity = calculate_infidelity(psi, psi)

        assert np.isclose(
            infidelity, 0.0, atol=1e-14
        ), f"相同态的不忠诚度应为0，实际为：{infidelity}"

    def test_orthogonal_states(self):
        """测试正交态的不忠诚度"""
        psi1 = np.array([1, 0, 0, 0], dtype=complex)
        psi2 = np.array([0, 1, 0, 0], dtype=complex)

        infidelity = calculate_infidelity(psi1, psi2)

        assert np.isclose(
            infidelity, 1.0, atol=1e-15
        ), f"正交态的不忠诚度应为1，实际为：{infidelity}"

    def test_antiparallel_states(self):
        """测试反平行态的不忠诚度"""
        psi1 = np.array([1, 1j, 0, 0], dtype=complex)
        psi1 = psi1 / np.linalg.norm(psi1)
        psi2 = -psi1  # 反平行

        infidelity = calculate_infidelity(psi1, psi2)

        # 反平行态的保真度为1（相位不影响），所以不忠诚度为0
        assert np.isclose(
            infidelity, 0.0, atol=1e-15
        ), f"反平行态的不忠诚度应为0，实际为：{infidelity}"

    def test_non_normalized_states(self):
        """测试非归一化态的处理"""
        psi1 = np.array([2, 0, 0, 0], dtype=complex)  # 范数为2
        psi2 = np.array([3, 0, 0, 0], dtype=complex)  # 范数为3

        infidelity = calculate_infidelity(psi1, psi2)

        # 非归一化态应该被自动归一化处理，结果应该是0
        assert np.isclose(
            infidelity, 0.0, atol=1e-15
        ), f"同方向非归一化态的不忠诚度应为0，实际为：{infidelity}"

    def test_phase_independence(self):
        """测试相位无关性"""
        psi1 = np.array([1, 1j, 0, 0], dtype=complex)
        psi1 = psi1 / np.linalg.norm(psi1)

        # 加上全局相位
        psi2 = np.exp(1j * np.pi / 3) * psi1

        infidelity = calculate_infidelity(psi1, psi2)

        assert np.isclose(
            infidelity, 0.0, atol=1e-15
        ), f"相位变化不应影响不忠诚度，实际为：{infidelity}"

    def test_range_validity(self):
        """测试不忠诚度的取值范围"""
        # 测试多个随机态对
        np.random.seed(42)

        for _ in range(10):
            psi1 = np.random.rand(4) + 1j * np.random.rand(4)
            psi2 = np.random.rand(4) + 1j * np.random.rand(4)

            psi1 = psi1 / np.linalg.norm(psi1)
            psi2 = psi2 / np.linalg.norm(psi2)

            infidelity = calculate_infidelity(psi1, psi2)

            assert (
                0 <= infidelity <= 1
            ), f"不忠诚度应在[0,1]范围内，实际为：{infidelity}"

    def test_symmetry(self):
        """测试对称性：I(ψ₁,ψ₂) = I(ψ₂,ψ₁)"""
        psi1 = np.array([1, 1j, 0.5, 0], dtype=complex)
        psi2 = np.array([0.5, 0, 1, 1j], dtype=complex)

        psi1 = psi1 / np.linalg.norm(psi1)
        psi2 = psi2 / np.linalg.norm(psi2)

        infidelity_12 = calculate_infidelity(psi1, psi2)
        infidelity_21 = calculate_infidelity(psi2, psi1)

        assert np.isclose(
            infidelity_12, infidelity_21, atol=1e-15
        ), f"不忠诚度应满足对称性，I(1,2)={infidelity_12}，I(2,1)={infidelity_21}"


class TestCalculateInfidelitySeries:
    """测试不忠诚度时间序列计算函数"""

    def test_infidelity_series_basic(self):
        """测试基本的时间序列计算"""
        num_times = 10
        num_states = 4

        # 创建测试数据
        psi_exact = np.random.rand(num_times, num_states) + 1j * np.random.rand(
            num_times, num_states
        )
        psi_approx = np.random.rand(num_times, num_states) + 1j * np.random.rand(
            num_times, num_states
        )

        # 归一化
        for i in range(num_times):
            psi_exact[i] = psi_exact[i] / np.linalg.norm(psi_exact[i])
            psi_approx[i] = psi_approx[i] / np.linalg.norm(psi_approx[i])

        infidelity_series = calculate_infidelity_series(psi_exact, psi_approx)

        # 检查输出形状
        assert infidelity_series.shape == (
            num_times,
        ), f"不忠诚度序列形状错误，预期：({num_times},)，实际：{infidelity_series.shape}"

        # 检查取值范围
        assert np.all(
            (0 <= infidelity_series) & (infidelity_series <= 1)
        ), "不忠诚度序列值超出[0,1]范围"

    def test_infidelity_series_identical(self):
        """测试相同序列的不忠诚度"""
        num_times = 5
        psi_series = np.random.rand(num_times, 4) + 1j * np.random.rand(num_times, 4)

        # 归一化
        for i in range(num_times):
            psi_series[i] = psi_series[i] / np.linalg.norm(psi_series[i])

        infidelity_series = calculate_infidelity_series(psi_series, psi_series)

        assert np.allclose(
            infidelity_series, 0.0, atol=1e-15
        ), "相同序列的不忠诚度应全为0"


class TestCalculateEpsilonParameter:
    """测试ε参数计算函数"""

    def test_epsilon_constant_gap(self):
        """测试恒定能隙情况的ε参数"""
        params = {"E0": 1.5, "lambda": 0.0, "hbar": 1.0, "v": 0.5}  # 恒定能隙

        epsilon_func, epsilon_min = calculate_epsilon_parameter(params)

        # 检查在不同s值处的ε值
        for s in [0.0, 0.5, 1.0]:
            epsilon_s = epsilon_func(s)
            expected = np.sqrt(2) * params["hbar"] * params["v"] / params["E0"]
            assert np.isclose(
                epsilon_s, expected, atol=1e-15
            ), f"恒定能隙s={s}时ε参数计算错误"

        # 检查最小ε值
        expected_min = np.sqrt(2) * params["hbar"] * params["v"] / params["E0"]
        assert np.isclose(
            epsilon_min, expected_min, atol=1e-15
        ), f"恒定能隙最小ε参数计算错误"

    def test_epsilon_time_dependent_gap(self):
        """测试时变能隙情况的ε参数"""
        params = {"E0": 1.0, "lambda": 1.0, "hbar": 1.0, "v": 0.3}  # 时变能隙

        epsilon_func, epsilon_min = calculate_epsilon_parameter(params)

        # 在s=0.5时，E(s) = E0 (最小值)，ε应该最大
        epsilon_center = epsilon_func(0.5)
        expected_center = np.sqrt(2) * params["hbar"] * params["v"] / params["E0"]

        assert np.isclose(
            epsilon_center, expected_center, atol=1e-15
        ), f"最大ε参数计算错误"

        # 在s=0时，E(s) = E0 + λ/4，ε应该更小
        epsilon_edge = epsilon_func(0.0)
        E_edge = params["E0"] + params["lambda"] * 0.25
        expected_edge = np.sqrt(2) * params["hbar"] * params["v"] / E_edge

        assert np.isclose(epsilon_edge, expected_edge, atol=1e-15), f"边界ε参数计算错误"

        # 边界处的ε应该小于中心处的ε
        assert epsilon_edge < epsilon_center, "边界处的ε参数应小于中心处"


class TestFormatScientificNotation:
    """测试科学记数法格式化函数"""

    def test_format_large_numbers(self):
        """测试大数的格式化"""
        result = format_scientific_notation(1234.567, precision=2)
        # 具体格式可能因实现而异，主要检查是否包含合理的数值
        assert isinstance(result, str), "格式化结果应为字符串"
        assert len(result) > 0, "格式化结果不应为空"

    def test_format_small_numbers(self):
        """测试小数的格式化"""
        result = format_scientific_notation(0.00123, precision=2)
        assert isinstance(result, str), "格式化结果应为字符串"
        assert len(result) > 0, "格式化结果不应为空"

    def test_format_zero(self):
        """测试零的格式化"""
        result = format_scientific_notation(0.0, precision=2)
        assert isinstance(result, str), "格式化结果应为字符串"
        # 零的格式化应该包含"0"
        assert "0" in result, f"零的格式化应包含'0'，实际：{result}"


class TestDataIOOperations:
    """测试数据输入输出操作"""

    def test_save_and_load_cycle(self):
        """测试保存和加载的完整循环"""
        # 创建测试数据
        test_data = {
            "s_values": np.linspace(0, 1, 11).tolist(),
            "infidelity_0": np.random.rand(11).tolist(),
            "infidelity_1": np.random.rand(11).tolist(),
        }

        test_filename = "test_data.json"

        try:
            # 保存数据
            save_results_to_file(test_data, test_filename)

            # 加载数据
            loaded_data = load_results_from_file(test_filename)

            # 验证数据一致性
            for key in test_data.keys():
                assert key in loaded_data, f"加载的数据缺少键：{key}"

        finally:
            # 清理测试文件
            if os.path.exists(test_filename):
                os.remove(test_filename)

    def test_load_nonexistent_file(self):
        """测试加载不存在文件的错误处理"""
        with pytest.raises(FileNotFoundError):
            load_results_from_file("nonexistent_file.json")


class TestPlotInfidelityComparison:
    """测试不忠诚度比较绘图函数"""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_basic_functionality(self, mock_savefig, mock_show):
        """测试绘图的基本功能"""
        s_values = np.linspace(0, 1, 21)
        infidelity_data = {
            "Zeroth order": np.random.rand(21) * 0.1,
            "First order": np.random.rand(21) * 0.05,
        }

        # 测试基本绘图
        try:
            fig, ax = plot_infidelity_comparison(
                s_values, infidelity_data, title="测试图"
            )
            plt.close(fig)  # 关闭图形以释放内存
            # 如果没有抛出异常，说明绘图功能基本正常
            assert True
        except Exception as e:
            pytest.fail(f"绘图函数抛出异常：{e}")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_with_epsilon(self, mock_savefig, mock_show):
        """测试带ε参数的绘图"""
        s_values = np.linspace(0, 1, 11)
        infidelity_data = {"Zeroth order": np.random.rand(11) * 0.1}

        try:
            fig, ax = plot_infidelity_comparison(
                s_values, infidelity_data, epsilon_value=0.5
            )
            plt.close(fig)
            assert True
        except Exception as e:
            pytest.fail(f"带ε参数的绘图抛出异常：{e}")


class TestSetupMatplotlibStyle:
    """测试matplotlib样式设置函数"""

    def test_style_setup(self):
        """测试样式设置功能"""
        # 保存原始设置
        original_font_size = plt.rcParams.get("font.size", 10)

        try:
            setup_matplotlib_style()

            # 样式函数应该成功执行而不抛出异常
            assert True

        finally:
            # 恢复原始设置
            plt.rcParams["font.size"] = original_font_size


# 参数化测试
@pytest.mark.parametrize("v_value", [0.1, 0.5, 1.0, 2.0])
def test_epsilon_scaling(v_value):
    """参数化测试：ε参数随v的缩放关系"""
    params = {"E0": 1.0, "lambda": 0.0, "hbar": 1.0, "v": v_value}

    epsilon_func, epsilon_min = calculate_epsilon_parameter(params)
    epsilon_actual = epsilon_func(0.5)
    expected = np.sqrt(2) * v_value / 1.0

    assert np.isclose(
        epsilon_actual, expected, atol=1e-15
    ), f"v={v_value}时ε参数缩放关系不正确"


@pytest.mark.parametrize("overlap", [0.0, 0.5, 0.8, 0.9, 0.99, 1.0])
def test_infidelity_values(overlap):
    """参数化测试：不同重叠度对应的不忠诚度"""
    # 构造具有指定重叠度的态
    psi1 = np.array([1, 0, 0, 0], dtype=complex)

    if overlap == 1.0:
        psi2 = psi1.copy()
    elif overlap == 0.0:
        psi2 = np.array([0, 1, 0, 0], dtype=complex)
    else:
        # 构造具有特定重叠度的态
        angle = np.arccos(overlap)
        psi2 = overlap * psi1 + np.sqrt(1 - overlap**2) * np.array(
            [0, 1, 0, 0], dtype=complex
        )
        psi2 = psi2 / np.linalg.norm(psi2)

    infidelity = calculate_infidelity(psi1, psi2)
    expected_infidelity = 1 - overlap**2

    assert np.isclose(
        infidelity, expected_infidelity, atol=1e-10
    ), f"重叠度{overlap}时不忠诚度计算错误"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
