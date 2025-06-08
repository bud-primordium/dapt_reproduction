"""
DAPT Tools Package

量子系统的退化绝热微扰理论(DAPT)计算工具包
"""

# Author: Gilbert Young
# Date: 2025-06-07

# 导入哈密顿量相关函数
from .hamiltonian import (
    get_hamiltonian,
    get_eigensystem,
    get_eigenvector_derivatives,
    verify_analytical_eigensystem,
    calculate_energy_gap,
    get_initial_state_in_standard_basis,
)

# 导入核心DAPT计算函数
from .core import (
    calculate_M_matrix,
    solve_wz_phase,
    dapt_recursive_step,
    run_dapt_calculation,
)

# 导入精确解模块
from .exact_solver import (
    solve_schrodinger_exact,
    calculate_exact_energy_expectation,
    verify_exact_solution_properties,
)

# 导入工具函数
from .utils import (
    calculate_infidelity,
    calculate_infidelity_series,
    calculate_epsilon_parameter,
    format_scientific_notation,
    save_results_to_file,
    load_results_from_file,
    setup_matplotlib_style,
)

__version__ = "2.5.1"
__author__ = "Gilbert Young"
__email__ = "gilbertyoung0015@gmail.com"

__all__ = [
    # 哈密顿量模块
    "get_hamiltonian",
    "get_eigensystem",
    "get_eigenvector_derivatives",
    "calculate_energy_gap",
    "get_initial_state_in_standard_basis",
    "verify_analytical_eigensystem",
    # 核心计算模块
    "calculate_M_matrix",
    "solve_wz_phase",
    "dapt_recursive_step",
    "run_dapt_calculation",
    # 精确解模块
    "solve_schrodinger_exact",
    "calculate_exact_energy_expectation",
    "verify_exact_solution_properties",
    # 工具模块
    "calculate_infidelity",
    "calculate_infidelity_series",
    "calculate_epsilon_parameter",
    "format_scientific_notation",
    "save_results_to_file",
    "load_results_from_file",
    "setup_matplotlib_style",
]
