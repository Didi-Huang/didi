# didi API Reference

> 自动生成自 /home/xiangqi/projects/didi/didi.py
> 最后更新: 2026-05-25

---

## 画图辅助

### didi.figure(chang=3.15, kuan=3.15, direction='in', length=5, width=2, zuobiaozhou='OFF', hengzuobiaokeduxian=True) -> (fig, ax)

创建预设样式的 matplotlib figure。

- chang, kuan: 宽高（英寸）
- zuobiaozhou='ON' 时额外画坐标轴（零线）
- hengzuobiaokeduxian=False 隐藏 x 轴刻度

### didi.legend(loc='lower right', zuobiao=None, kuangkuang=True, zuobiaozhoushu=1, ax=None) -> legend

自定义图例。支持副坐标轴合并。

- loc: 语义位置（upper left, center, lower right 等）
- zuobiao: 手动锚点 (x, y) 比例坐标
- zuobiaozhoushu: 设为 2 时自动合并副轴图例
- kuangkuang: 是否显示边框

### didi.legend_like(ax, text, xy=(0.96, 0.96), loc='upper right', size=9, frame=True) -> AnchoredText

在坐标轴内部创建类似于图例的文本块。

### didi.linetag(k, kerr, b, berr, r2, sci_threshold=3, digits=1, loc='upper left', zuobiao=None, kuangkuang=True)

在图上标注线性拟合结果。

- k, kerr: 斜率及误差
- b, berr: 截距及误差
- r2: 决定系数
- 输出格式：y = (k +/- kerr)x + (b +/- berr), R^2 = ...

### didi.drawdxs(p, x0=0, x1=10, number=50, label=False) -> (x, y)

根据多项式系数画拟合曲线。

- p: np.polyfit 格式的多项式系数（高次到低次）
- label: True 时标注「近似線」，字符串时作为自定义 label

---

## 输出配置

### didi.setpgf(texsystem="xelatex", font_family="serif")

配置 matplotlib 输出 pgf 格式（适用于 LaTeX 报告嵌入）。
后端 pgf，xelatex + xeCJK + amsmath + siunitx。

### didi.savepgf(savepath, filename, tight=True)

保存当前图为 .pgf 文件。自动补全文件名。

### didi.savepdf(savepath, filename, tight=True)

保存当前图为 .pdf 文件。

### didi.setpdf()

配置 pdf 输出（类似 setpgf 但输出 pdf）。

---

## 拟合

### didi.zxecf(x, y, dim=1, error=True, printt=True, draw=True, drawcolor='red', tag=False, drawlabel=False) -> [params]

多项式最小二乘拟合。

- x, y: 数据
- dim: 多项式次数
- error: 是否返回误差
- printt: 是否打印结果
- draw: 是否画拟合线
- tag: 是否在图例中显示公式

### didi.dxs(x, params, dim) -> ndarray

根据多项式系数计算拟合值。

---

## 数据 I/O

### didi.loadcsv(path, type, begin, end) -> DataFrame

从 CSV 文件读取指定行范围的数据。

---

## 工具

### didi.start()

启动计时器 + 系统通知 + 播放音效（macOS only）。

### didi.ending()

计时结束 + 打印耗时 + 系统通知 + 播放音效（macOS only）。

### didi.pm(a)

带颜色格式化打印矩阵/二维数组。

### didi.play_note(...)

播放音符序列。可选正弦波/方波。

---

## 线性代数（数值计算课遗留）

### didi.GJ(a) / GJ_1(a) / G(a) / GS(a)

Gauss-Jordan 消元法、Gauss 消元法、Gauss-Seidel 迭代法。

### didi.m_Ab(x, y, m) -> ndarray

构造多项式拟合的正规方程矩阵。
