# Naming Conventions for Physics Lab Python Code

> 原则: AI 时代，变量名首要读者是 AI 和你自己，而非键盘。
> 长的、语义明确的名字让 AI 给出更正确的代码。

---

## 一、缩写规则

### ✅ 可用：领域标准术语

这些是物理学/工程学的标准缩写，业内共识，不需要展开：

| 缩写 | 全称 | 备注 |
|---|---|---|
| BC | boundary condition | 边值条件 |
| IC | initial condition | 初始条件 |
| FFT | fast Fourier transform | 快速傅里叶变换 |
| ODE | ordinary differential equation | 常微分方程 |
| PDE | partial differential equation | 偏微分方程 |
| RK4 | 4th order Runge-Kutta | 经典 RK 法 |
| GS | Gauss-Seidel | 迭代法 |
| SOR | successive over-relaxation | 逐次超松弛 |
| PDF | probability density function | 概率密度函数 |
| SNR / CNR | signal-to-noise / contrast-to-noise | 信噪比 |
| FWHM | full width at half maximum | 半高全宽 |
| PSD | power spectral density | 功率谱密度 |
| RMS / RMSE | root mean square (error) | 均方根(误差) |
| DOF | degrees of freedom | 自由度 |
| EMF | electromotive force | 电动势 |
| ADC / DAC | analog-digital converter | 模数/数模转换 |
| V_in / V_out | input/output voltage | 输入/输出电压 |
| C_d | drag coefficient | 阻力系数 |
| R^2 | coefficient of determination | 决定系数 |

### ❌ 不用：纯粹为少打字的缩写

| 不用 | 用这个 | 理由 |
|---|---|---|
| `bnds` | `bounds` | 非标准，多一个字母不可读 |
| `tmp` | `temp` 或 `temporary` | 除非是交换变量的极小作用域 |
| `val` | `value` | 3 个 vs 5 个字母，可读性翻倍 |
| `ret` | `result` | 同 |
| `calc` | `calculate` 或 `compute` | 同 |
| `param` | `params` 或保留全称 | `param` 可能被误解为 parameter 的单数，复数更清楚 |
| `num` | `n` 或 `count` | `n` 在数学语境约定俗成，`num` 是半吊子缩写 |
| `info` | `metadata` 或具体名 | `info` 太泛 |
| `diff` | `difference` 或 `delta` | 除非是 `np.diff` 的结果变量 |

---

## 二、中文 vs 英文 vs 日文

### 变量名/函数名

**全用英文**。代码里不要出现中文/日文变量名。

```
# ❌
def 計算平均(数据):
    ...

# ✅
def compute_mean(data):
    ...
```

### 注释

注释用中文。实验报告代码主要是自己看，中文最自然。

```
# 根据仪器精度计算不确定度
sigma = 0.5 * 10 ** (-decimal_places)
```

### LaTeX 标签

用英文写变量名，注释用中文解释即可。

```
plt.xlabel(r"Input voltage $V_{\mathrm{in}}$ [V]")
```

---

## 三、命名风格速查

| 类型 | 风格 | 示例 |
|---|---|---|
| 常量 | `UPPER_SNAKE` | `FIGURE_SAVE_PATH`, `MIN_RESOLUTION` |
| 全局变量 | 基本不用 | 物理常数 `g=9.8` 例外 |
| 函数 | `snake_case` | `fit_linear()`, `infer_uncertainty()` |
| 类 | `PascalCase` | `MeasurementSet`, `LinearFitResult` |
| 参数 | `snake_case` | `y_err`, `sigfigs`, `floor_unc` |
| 局部变量 | `snake_case` | `result`, `x_smooth`, `residuals` |
| 文件/目录 | `snake_case` | `rawdata.csv`, `plot-exp-1.png` |

---

## 四、命名长度原则

```
作用域越小 → 名字可以越短
作用域越大 → 名字需要越长
```

| 作用域 | 原则 | 示例 |
|---|---|---|
| 循环下标 | 1 个字母 | `i`, `j`, `k` |
| 列表推导临时变量 | 1-2 字母 | `x` for x in values |
| 10 行内的局部变量 | 短词 | `result`, `slope`, `r2` |
| 函数参数 | 完整词 | `dataframe`, `figure_path` |
| 模块级变量/常量 | 完整 + 大写 | `FIGURE_SAVE_PATH` |
| 函数名 | 动词短语 | `fit_linear()`, `plot_experiment()` |

---

## 五、物理量命名（国际惯例）

| 物理量 | 变量名 | 单位（注释） |
|---|---|---|
| 电压 | `V`, `voltage` | V |
| 电流 | `I`, `current` | A |
| 频率 | `f`, `frequency` | Hz |
| 质量 | `m`, `mass` | kg |
| 长度 | `l`, `L`, `length` | m |
| 时间 | `t`, `time` | s |
| 角度 | `theta`, `angle` | rad |
| 角速度 | `omega`, `dot_theta` | rad/s |
| 温度 | `T` | K (开尔文) |
| 粘性系数 | `eta` | Pa*s |
| 密度 | `rho` | kg/m^3 |
