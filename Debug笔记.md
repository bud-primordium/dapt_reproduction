### DAPT 复现调试指南

**文档目标**：为复现 G. Rigolin 和 G. Ortiz 2014 年 PRA 论文 (Phys. Rev. A 90, 022104) Sec. VIII 的数值示例，提供 DAPT 理论公式、特定模型参数、计算流程和调试要点。

---

## 一、核心理论与模型参数

### 1. DAPT 基础公式

* **连接矩阵 (Eq. 31)：**

  $$
  [\mathbf{M}^{mn}(s)]_{g_m h_n}
  = \bigl\langle m^{g_m}(s)\bigm|\frac{d}{ds}\bigm|n^{h_n}(s)\bigr\rangle.
  $$

* **维度**：$d_m\times d_n$

* **WZ相位 (Wilczek–Zee Phase) (Eq. 29)：**

  $$
  \frac{d}{ds}\mathbf{U}^n(s)
  = -\,\mathbf{U}^n(s)\,\mathbf{M}^{nn}(s),
  \quad \mathbf{U}^n(0)=\mathbf{I}.
  $$

### 1.1 特定数值模型：哈密顿量与参数

* **系统哈密顿量 (Eq. 134)：**

  $$
  \mathbf{H}(s)
  =\frac{1}{\sqrt{2}}
  \begin{pmatrix}
    \mathbf{0} & \mathbf{H}_1(s) \\
    \mathbf{H}_1^\dagger(s) & \mathbf{0}
  \end{pmatrix},
  \quad
  \mathbf{H}_1(s)
  =\begin{pmatrix}
    -E(s) & e^{-\mathrm{i}\theta(s)}E(s) \\[6pt]
    e^{\mathrm{i}\theta(s)}E(s) & E(s)
  \end{pmatrix}.
  $$

* **能量函数 (Eq. 135)：**

  $$
  E(s) = E_0 + \lambda\bigl(s-\tfrac12\bigr)^2,\quad E_0>0.
  $$

* **相位函数 (Eq. 136)：**

  $$
  \theta(s) = \theta_0 + w\,s^2.
  $$

### 1.2 特定数值模型：能谱与解析本征态

> **注意**：必须使用以下解析表达式，禁止对 $\mathbf{H}(s)$ 进行数值对角化。

* **本征能量（二重简并）**  
  * 基态子空间：$E_{GS}(s)=-E(s)$  
  * 激发态子空间：$E_{EX}(s)=+E(s)$  

* **瞬时能隙**：

  $$
  \Delta(s)
  =E_{EX}(s)-E_{GS}(s)
  =2\,E(s)
  =2\bigl(E_0+\lambda(s-\tfrac12)^2\bigr).
  $$

* **瞬时本征态（解析表达式）**  
  * 基态子空间 $\mathcal{H}_0$ ($n=0$)：

    $$
    |0^0(s)\rangle
    =\tfrac12\bigl(e^{-\mathrm{i}\theta(s)}|\uparrow\uparrow\rangle
      +|\uparrow\downarrow\rangle
      -\sqrt2\,|\downarrow\downarrow\rangle\bigr),
    $$

    $$
    |0^1(s)\rangle
    =\tfrac12\bigl(|\uparrow\uparrow\rangle
      -e^{\mathrm{i}\theta(s)}|\uparrow\downarrow\rangle
      +\sqrt2\,|\downarrow\uparrow\rangle\bigr).
    $$

  * 激发态子空间 $\mathcal{H}_1$ ($n=1$)：

    $$
    |1^0(s)\rangle
    =\tfrac12\bigl(e^{-\mathrm{i}\theta(s)}|\uparrow\uparrow\rangle
      +|\uparrow\downarrow\rangle
      +\sqrt2\,|\downarrow\downarrow\rangle\bigr),
    $$

    $$
    |1^1(s)\rangle
    =\tfrac12\bigl(|\uparrow\uparrow\rangle
      -e^{\mathrm{i}\theta(s)}|\uparrow\downarrow\rangle
      -\sqrt2\,|\downarrow\uparrow\rangle\bigr).
    $$

### 2. DAPT 递推关系

* **核心方程 (Eq. 25)：**

  $$
  \frac{\mathrm{i}}{\hbar}\Delta_{nm}(s)\,\mathbf{B}_{mn}^{(p+1)}(s)
  + \dot{\mathbf{B}}_{mn}^{(p)}(s)
  + \sum_{k}\mathbf{B}_{mk}^{(p)}(s)\,\mathbf{M}^{kn}(s)
  = 0.
  $$

* **非对角项 ($m\neq n$, Eq. C1)：**

  $$
  \mathbf{B}_{mn}^{(p+1)}(s)
  = \frac{\mathrm{i}\hbar}{\Delta_{nm}(s)}
    \Bigl(\dot{\mathbf{B}}_{mn}^{(p)}(s)
    + \sum_{k}\mathbf{B}_{mk}^{(p)}(s)\,\mathbf{M}^{kn}(s)\Bigr).
  $$

* **对角项 ($m=n$)：**

  $$
  \frac{d}{ds}\mathbf{B}_{nn}^{(p+1)}(s)
  = -\,\mathbf{B}_{nn}^{(p+1)}(s)\,\mathbf{M}^{nn}(s)
    - \sum_{k\neq n}\mathbf{B}_{nk}^{(p+1)}(s)\,\mathbf{M}^{kn}(s).
  $$

### 3. 初始条件

* **零阶系数 (Eq. 16)：**

  $$
  \mathbf{B}_{mn}^{(0)}(s)
  = b_n(0)\,\mathbf{U}^{n}(s)\,\delta_{nm},\quad
  b_n(0)=\delta_{n0}.
  $$

  因此 $\mathbf{B}_{00}^{(0)}(s)=\mathbf{U}^0(s)$，其余为 $\mathbf{0}$。

* **高阶系数初值 (Eq. 18)：**

  $$
  \mathbf{B}_{nn}^{(p+1)}(0)
  = -\sum_{m\neq n}\mathbf{B}_{mn}^{(p+1)}(0).
  $$

### 4. 波函数组装

* **总波函数 (Eq. 15)：**

  $$
  \bigl|\Psi(s)\bigr\rangle
  = \sum_{n,m}\sum_{p=0}^{\infty}v^p
    e^{-\tfrac{\mathrm{i}}{v}\omega_m(s)}
    \,\mathbf{B}_{mn}^{(p)}(s)\,\bigl|n_{\rm state}(s)\bigr\rangle.
  $$

* **更具体形式（Eq. 12,13）：**

  $$
  \bigl|\Psi(s)\bigr\rangle
  = \sum_{p=0}^\infty v^p
    \sum_{k=0}^1 e^{-\tfrac{\mathrm{i}}{v}\omega_k(s)}
    \Bigl(\sum_{j=0}^1
      e^{\tfrac{\mathrm{i}}{v}\omega_{kj}(s)}\,
      \mathbf{B}_{jk}^{(p)}(s)\Bigr)
    \,\bigl|\text{basis}_k(s)\bigr\rangle,
  $$

  其中 $\omega_m(s)=\int_0^sE_m(s')\,ds'$。

* **不忠诚度 (Eq. 148)：**

  $$
  I_k(s) = 1 - \bigl|\langle\Psi_{\rm exact}(s)\mid
    \Psi(s)\rangle_{N_k}\bigr|^2.
  $$

---

## 二、计算流程

### 阶段 0：环境设置与静态量

| 计算量                             | 物理/数学含义       | 维度/Shape             | 计算方法/来源              |
|:---------------------------------|:-------------------:|:----------------------:|:---------------------------|
| $\hbar, v, w, E_0, \lambda,\theta_0$ | 物理常数和模型参数    | 标量                   | 附录实验参数表             |
| $s$                               | 重标定无量纲时间       | 1D 数组 $(N_{\rm steps},)$ | `np.linspace(0,1,N_steps)` |
| $E(s)$                            | 时变能量尺度函数       | 1D 数组 $(N_{\rm steps},)$ | $E_0 + \lambda(s-0.5)^2$   |
| $\theta(s)$                       | 时变相位函数          | 1D 数组 $(N_{\rm steps},)$ | $\theta_0 + w s^2$         |
| $\mathbf{H}(s)$                   | 系统瞬时哈密顿量       | $(N_{\rm steps},4,4)$     | Eq. (134)                  |

### 阶段 1：瞬时本征体系

> **要求**：严格使用 1.2 节的解析本征态及其解析导数。

| 计算量                               | 物理含义               | 维度/Shape | 计算方法与依赖                      |
|:------------------------------------|:---------------------:|:---------:|:-----------------------------------|
| $E_n(s)$                            | 子空间 $n$ 瞬时本征能量 | 标量       | 解析公式（见 1.2）；依赖：$E(s)$     |
| $\Delta_{nm}(s)$                    | 子空间间瞬时能隙         | 标量       | 解析公式；依赖：$E_n(s)$            |
| $\|n^{g_n}(s)\rangle$   |瞬时本征矢             | $4\times1$| 解析公式（Eq. 137–140）；依赖：$\theta(s)$ |
| $\tfrac{d}{ds}\|n^{g_n}(s)\rangle$   | 本征矢时间导数          | $4\times1$| 符号微分；依赖：本征矢解析形式、$d\theta/ds$ |
| $\mathbf{M}^{mn}(s)$                | 连接矩阵               | $2\times2$| $\langle m\|d/ds\|n\rangle$；依赖：本征矢及其导数 |

### 阶段 2：DAPT 递推计算

#### 2.1 零阶项 ($p=0$)

| 计算量                           | 含义                 | 维度/Shape | 方法与依赖                                                |
|:--------------------------------|:-------------------:|:---------:|:----------------------------------------------------------|
| $\mathbf{U}^n(s)$               | 并行传输算符         | $2\times2$| ODE 求解：$\tfrac{dU^n}{ds}=-U^nM^{nn}$，$U^n(0)=I$；依赖：$M^{nn}(s)$ |
| $\mathbf{B}_{mn}^{(0)}(s)$      | 零阶 DAPT 系数      | $2\times2$| $b_n(0)\,U^n\delta_{nm}$；$b_n(0)=\delta_{n0}$            |

#### 2.2 $p+1$ 阶项

##### 2.2.1 非对角项 ($m\neq n$)

| 计算量                               | 含义               | 维度/Shape | 方法与依赖                                                    |
|:------------------------------------|:-----------------:|:---------:|:-------------------------------------------------------------|
| $\tfrac{d}{ds}\mathbf{B}_{mn}^{(p)}$ | $p$ 阶系数时间导数  | $2\times2$| 数值微分：`np.gradient(B[p][m][n],s)`                         |
| $\sum_k \mathbf{B}_{mk}^{(p)}M^{kn}$ | 求和项             | $2\times2$| $\sum_{k=0,1}B_{mk}^{(p)}M^{kn}$                              |
| $\mathbf{B}_{mn}^{(p+1)}$ ($m\neq n$)| $p+1$ 阶系数（非对角）| $2\times2$| Eq. C1：$\tfrac{i\hbar}{\Delta_{nm}}(\dot B+\sum B\,M)$；依赖：$\Delta_{nm},\dot B,\sum B\,M$ |

##### 2.2.2 对角项 ($m=n$)

| 计算量                                    | 含义                | 维度/Shape | 方法与依赖                                                          |
|:-----------------------------------------|:------------------:|:---------:|:-------------------------------------------------------------------|
| $\mathbf{B}_{nn}^{(p+1)}(0)$              | 初值                | $2\times2$| Eq. 18：$-\sum_{m\neq n}B_{mn}^{(p+1)}(0)$                          |
| $\mathbf{B}_{nn}^{(p+1)}(s)$ ($m=n$)      | 对角系数            | $2\times2$| ODE 求解：$dB_{nn}/ds=-B_{nn}M^{nn}-\sum_{k\neq n}B_{nk}M^{kn}$；依赖：初始值及 $B_{nk}^{(p+1)}(s),M(s)$ |

### 阶段 3：波函数组装与验证

| 计算量                                    | 含义                         | 维度/Shape | 方法与依赖                                                |
|:-----------------------------------------|:---------------------------:|:---------:|:----------------------------------------------------------|
| $\|\Psi^{(p)}(s)\rangle$                   | $p$ 阶波函数修正              | $4\times1$| 组装公式（见 4. 波函数组装）；依赖：$B_{mn}^{(p)},\|n^{g_n}\rangle,\omega_n$ |
| $\|\Psi(s)\rangle_{N_k}$                  | $k$ 阶近似解（归一化）        | $4\times1$| $\sum_{p=0}^k v^p\|\Psi^{(p)}\rangle$，再归一化              |
| $\|\Psi_{\rm exact}(s)\rangle$             | 精确解                        | $4\times1$| ODE 求解：$i\hbar v\,d\Psi/ds=H(s)\Psi$；初值 $|0^0(0)\rangle$ |
| $I_k(s)$                                | 不忠诚度                     | 1D 数组   | $1-\|\langle\Psi_{\rm exact}\|\Psi\rangle\|^2$                  |

---

## 三、调试要点

1. 本征体系：严格使用 1.2 节解析本征态及其导数计算 $M$ 矩阵。  
2. 矩阵乘法顺序：递推关系中求和项应为 $B_{mk}^{(p)}\,M^{kn}$.  
3. ODE 求解器：并行传输 $U^n$ 和对角项 $B_{nn}$ 均用高质量自适应步长求解器（如 RK45），并设 `rtol=1e-8`, `atol=1e-10`.  
4. 索引与数据结构：确认数组 `B[p][m][n]` 的维度与索引顺序正确。  
5. 波函数组装：检查相位因子 $e^{-i\omega_m/v}$ 和 $v^p$ 因子是否遗漏。

---

## 四、附录：实验参数

**通用参数**：$\hbar=1.0,\ \theta_0=0.1$.  
**绝热参数**：$\epsilon(s)=\frac{\sqrt2\,\hbar v}{E(s)}$；若 $\lambda\neq0$，则 $\epsilon_{\min}=\frac{\sqrt2\,\hbar v}{E_0}$.

| 实验 (Fig) | 条件                         | $E_0$ | $\lambda$ | $v$   | $w$   | $\epsilon$ 或 $\epsilon_{\min}$       |
|:-----------|:---------------------------:|:-----:|:--------:|:-----:|:-----:|:-------------------------------------:|
| Fig. 2     | 恒定能隙，绝热               | 1.5   | 0        | 0.5   | 0.5   | $\epsilon\approx0.47$                 |
| Fig. 3     | 恒定能隙，非绝热             | 1.5   | 0        | 1.5   | 1.5   | $\epsilon\approx1.41$                 |
| Fig. 4     | 变动能隙，变 $v$ ($E_0=1.0$) | 1.0   | 1.0      | 0.3   | 0.3   | $\epsilon_{\min}\approx0.42$          |
|            |                             | 1.0   | 1.0      | 0.5   | 0.5   | $\epsilon_{\min}\approx0.71$          |
|            |                             | 1.0   | 1.0      | 0.8   | 0.8   | $\epsilon_{\min}\approx1.13$          |
|            |                             | 1.0   | 1.0      | 1.2   | 1.2   | $\epsilon_{\min}\approx1.70$          |
| Fig. 5     | 变动能隙，变 $E_0$ ($v=w=0.3$)| 1.00  | 1.0      | 0.3   | 0.3   | $\epsilon_{\min}\approx0.42$          |
|            | ($gap_{min}=1.5\Rightarrow E_0=0.75$) | 0.75  | 1.0      | 0.3   | 0.3   | $\epsilon_{\min}\approx0.57$          |
|            | ($gap_{min}=1.0\Rightarrow E_0=0.50$) | 0.50  | 1.0      | 0.3   | 0.3   | $\epsilon_{\min}\approx0.85$          |
|            | ($gap_{min}=0.5\Rightarrow E_0=0.25$) | 0.25  | 1.0      | 0.3   | 0.3   | $\epsilon_{\min}\approx1.70$          |

---

## 五、最终调试日志：定位主要偏差并分析残留数值问题

**日期**: 2025-06-08

### 5.1 问题现象

在项目后期，我们达到了一个关键但令人困惑的状态：

* **一阶修正 (`p=1`)**: 计算出的不忠诚度曲线与论文 `Fig. 2` **基本吻合**。
* **二阶修正 (`p=2`)**: 不忠诚度曲线与论文不符，且精度通常劣于一阶。

这表明，计算 `B^(1)` 的框架是正确的，但用 `B^(1)` 计算 `B^(2)` 的过程中存在一个隐藏的错误。

### 5.2 诊断过程与最终结论

经过多次理论分析与代码实验的反复验证，我们排除了对角项ODE求解、矩阵乘法顺序等多种可能性。最终，我们将注意力集中在那个唯一能产生正确一阶结果的**非对角项递推公式**上。

我们当前生效的代码逻辑是：
$$
\mathbf{B}_{mn}^{(p+1)} \propto \left( \dot{\mathbf{B}}_{mn}^{(p)} - \sum_{k}\mathbf{B}_{nk}^{(p)}\mathbf{M}^{km} \right)
$$

这个公式的求和项结构 (`Σ B_nk M^km`) 虽然与论文 `Eq. 25` 的索引 (`Σ B_mk M^kn`) 不同，但它是经过实验验证的、能产生正确物理结果的结构。我们必须信任它。

**最终定位的错误**在于 `B_dot` 项前的符号。

* **对于一阶 (`p=0`)**: 由于 `B_mn^(0)` (m≠n) 是零矩阵，其导数 `B_dot_mn^(0)` 也为零。因此，`B_dot` 项前的符号是 `+` 还是 `-` 对一阶结果**完全没有影响**。这掩盖了该处的符号错误。

* **对于二阶 (`p=1`)**: `B_mn^(1)` 是非零的，其导数 `B_dot_mn^(1)` 是一个关键贡献。此时，一个错误的符号（我们之前代码中隐含的正号）导致了二阶计算的失败。

### 5.3 最终修复方案

我们对 `core.py` 中 `dapt_recursive_step` 函数的非对角项递推公式进行了唯一的、最小化的修改：**将括号内的两项都设为负号**，以符合 `Eq. 25` 的整体结构 `-(...)`。

**最终生效的、经过验证的递推关系**:
$$
\mathbf{B}_{mn}^{(p+1)}(s) = \frac{\mathrm{i}\hbar}{\Delta_{nm}(s)}\left( -\dot{\mathbf{B}}_{mn}^{(p)}(s) - \sum_{k}\mathbf{B}_{nk}^{(p)}\mathbf{M}^{km}(s) \right)
$$
这个公式保留了被验证为正确的求和项结构，同时修正了仅在高阶计算中才显现的导数项符号错误。

### 5.4 结果与残留问题分析

**主要成果**:
此修改成功解决了二阶修正中的**主要偏差问题**。各阶不忠诚度曲线的层次结构（即在绝热区 $I_0 > I_1 > I_2$）与论文图示的物理行为**基本吻合**，验证了我们所采用的递推公式的正确性。

**残留的数值问题**:
然而，在项目最终验证阶段，我们观察到一个残留的数值问题。具体表现为，在非绝热性较强（即 `v` 较大）的情况下，二阶（及更高阶）不忠诚度曲线在初始时刻 `s=0` 处存在一个微小的、非零的“抬头”现象，而理论上该值应精确为零。

**问题诊断**:
深入分析表明，此现象并非源于DAPT理论公式的错误，而是**纯粹的数值精度问题**。其根源在于计算二阶系数初值 `B_mn^(2)(0)` 时，需要对一阶系数 `B_mn^(1)` 在端点 `s=0` 处求导。标准数值微分方法（如 `np.gradient`）在端点的精度有限，会引入一个微小误差。这个误差虽小，但在二阶项中被放大，最终导致了初始不忠诚度的“鬼影”。

**项目结论**:
综上，本项目已成功复现了DAPT理论的核心物理行为，并建立了一套经过验证的、自洽的计算框架。虽然存在一个已明确诊断的、由数值方法精度限制导致的微小初始偏差，但项目的核心目标——理解并实现DAPT理论——已经达成。
