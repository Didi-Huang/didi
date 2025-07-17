import os
from pync import Notifier
import pandas as pd
#/Users/martina/Library/Sounds
import subprocess
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from scipy.optimize import least_squares
from matplotlib.offsetbox import AnchoredText
import time

class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            return None
        return time.time() - self.start_time

# 模块级定时器实例
_timer = Timer()

def start():
    Notifier.notify("実行開始！！", title="通知")
    subprocess.Popen('afplay /Users/martina/Library/Sounds/begining.wav', shell=True)
    _timer.start()

def ending():
    elapsed = _timer.elapsed()

    # ANSI 颜色
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

    if elapsed is not None:
        # 控制台彩色输出
        print(f"{GREEN}実行完了！！{RESET} (所要時間：{CYAN}{elapsed:.2f}{RESET} )秒")
        # 通知栏（不能彩色，但能弹窗）
        Notifier.notify(f"所要時間：{elapsed:.2f} 秒", title="実行完了！！")
    else:
        print("実行完了！！ (時間記録なし)")
        Notifier.notify("時間記録なし", title="実行完了！！")

    subprocess.Popen('afplay /Users/martina/Library/Sounds/ending.wav', shell=True)
    
def loadcsv(path,type,begin,end):    
    if type == '-':
        result = pd.read_csv(path, header=None, skiprows=begin, nrows=end-begin)
    elif type == '|':
        result = pd.read_csv(path, header=None, skiprows=begin, nrows=end-begin)
    return result
def legend(loc='lower right', zuobiao=None, kuangkuang=True, zuobiaozhoushu=1, ax=None):
    """
    自定义图例函数（坐标轴挂载）：
    - 支持 loc 语义位置、手动 zuobiao 锚点
    - 支持合并副坐标轴图例
    - 支持传入 ax，避免 plt.gca() 出错
    """
    loc_dict = {
        'upper left':  (0.04, 0.96),
        'upper right': (0.96, 0.96),
        'lower left':  (0.04, 0.04),
        'lower right': (0.96, 0.04),
        'middle left': (0.04, 0.5),
        'middle right':(0.96, 0.5),
        'middle top':  (0.5, 0.96),
        'middle bottom': (0.5, 0.04),
        'center':      (0.5, 0.5)
    }

    if ax is None:
        ax = plt.gca()  # 默认行为，但最好显式传 ax
    lines1, labels1 = ax.get_legend_handles_labels()

    if zuobiaozhoushu >= 2:
        fig = ax.figure
        if len(fig.axes) > 1:
            ax2 = fig.axes[1]
            lines2, labels2 = ax2.get_legend_handles_labels()
        else:
            lines2, labels2 = [], []
    else:
        lines2, labels2 = [], []

    kwargs = {
        'handles': lines1 + lines2,
        'labels': labels1 + labels2,
        'frameon': kuangkuang,
        'framealpha': 1 if kuangkuang else 0,
        'edgecolor': "black" if kuangkuang else 'white',
        'facecolor': "white" if kuangkuang else "none",
        'fancybox': False,
        'borderpad': 0.5,
        'handletextpad': 0.5,
        'labelspacing': 0.5,
        'handlelength': 0.7,
        'borderaxespad': 1.0
    }

    if zuobiao is not None:
        legend = ax.legend(
            loc='center',
            bbox_to_anchor=zuobiao,
            bbox_transform=ax.transAxes,
            **kwargs
        )
    elif loc in loc_dict:
        legend = ax.legend(
            loc='center',
            bbox_to_anchor=loc_dict[loc],
            bbox_transform=ax.transAxes,
            **kwargs
        )
    else:
        legend = ax.legend(loc=loc, **kwargs)

    legend.get_frame().set_linewidth(0.5 if kuangkuang else 0)
    return legend
#def dxs(x, params, dim):#多项式
#    if dim == 0:
#        # 使用 params 顺序（高次 → 低次）
#        return sum(p * x**(len(params)-1 - i) for i, p in enumerate(params))
#    else:
#        if len(params) < dim + 1:
#            raise ValueError(f"dim={dim} 需要至少 {dim+1} 个参数，但只给了 {len(params)} 个")
#        trimmed_params = params[:dim+1]  # 保留前 dim+1 项
#        return sum(p * x**(dim - i) for i, p in enumerate(trimmed_params))
def dxs(x, params, dim):
    if len(params) < dim + 1:
        raise ValueError(f"dim={dim} 需要至少 {dim+1} 个参数，但只给了 {len(params)} 个")
    return np.polyval(params[:dim+1], x)

def linetag(k, kerr, b, berr, r2, sci_threshold=3, digits=1, loc='upper left', zuobiao=None, kuangkuang=True):
    import numpy as np
    import matplotlib.pyplot as plt

    def format_with_error(val, err, digits=1):
        if val == 0 or err == 0 or np.isnan(err) or np.isinf(err):
            return f"{val:.3g}"  # 或者 return "0"
        exponent = int(np.floor(np.log10(abs(val))))
        use_sci = abs(exponent) >= sci_threshold
        if use_sci:
            val_coeff = val / 10**exponent
            err_coeff = err / 10**exponent
            if err_coeff == 0 or np.isnan(err_coeff) or np.isinf(err_coeff):
                return f"{val_coeff:.3g} \\times 10^{{{exponent}}}"
            err_digits = max(0, -int(np.floor(np.log10(err_coeff))) + (digits - 1))
            fmt = f".{err_digits}f"
            val_str = format(val_coeff, fmt)
            err_str = format(err_coeff, fmt)
            return rf"({val_str} \pm {err_str}) \times 10^{{{exponent}}}"
        else:
            err_digits = max(0, -int(np.floor(np.log10(err))) + (digits - 1))
            fmt = f".{err_digits}f"
            val_str = format(val, fmt)
            err_str = format(err, fmt)
            return rf"{val_str} \pm {err_str}"

    # 定义位置映射
    loc_dict = {
        'upper left':  (0.04, 0.96),
        'upper right': (0.96, 0.96),
        'lower left':  (0.04, 0.04),
        'lower right': (0.96, 0.04),
        'middle left': (0.04, 0.5),
        'middle right':(0.96, 0.5),
        'middle top':  (0.5, 0.96),
        'middle bottom': (0.5, 0.04),
        'center':      (0.5, 0.5)
    }

    if zuobiao is not None:
        anchor = zuobiao
        ha, va = 'center', 'center'
    elif loc in loc_dict:
        anchor = loc_dict[loc]
        ha = 'left' if 'left' in loc else 'right' if 'right' in loc else 'center'
        va = 'top' if 'upper' in loc else 'bottom' if 'lower' in loc else 'center'
    else:
        anchor = (0.04, 0.96)
        ha, va = 'left', 'top'

    # 构造文字内容
    k_str = format_with_error(k, kerr, digits)
    b_str = format_with_error(b, berr, digits)
    textstr = (
        "近似直線:\n"
        + rf"$y = ({k_str})x + ({b_str})$" + "\n"
        + rf"$R^2 = {r2:.3f}$"
    )

    # 绘制文字
    plt.text(
        anchor[0], anchor[1],
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        ha=ha, va=va,
        bbox=dict(
            facecolor='white' if kuangkuang else 'none',
            edgecolor='black' if kuangkuang else 'none',
            boxstyle='square,pad=0.4',
            linewidth=0.5 if kuangkuang else 0
        )
    )
    
def drawdxs(p,x0=0,x1=10,number=50,label=False):#x_data.min(), x_data.max()
    x = np.linspace(x0, x1, num=number)
    y = np.polyval(p, x)
    if label == True:
        plt.plot(x, y, color='red', linestyle='-', linewidth=0.5, label='近似線\u3000\u3000')
    elif label == False:
        plt.plot(x, y, color='red', linestyle='-', linewidth=0.5)
    elif label not in [True, False]:
        plt.plot(x, y, linestyle='-', linewidth=1, label=label)
    
    return x,y

def figure(chang=3.15, kuan=3.15,
           direction='in', length=5, width=1.3,
           zuobiaozhou='OFF', hengzuobiaokeduxian=True):
    fig, axx = plt.subplots(figsize=(chang, kuan))

    axx.tick_params(axis='both', which='major', direction=direction, length=length, width=width, labelsize=9)
    axx.tick_params(axis='both', which='minor', direction=direction, length=(length)/2, width=0.8*width)
    axx.set_facecolor('white')

    if zuobiaozhou == 'ON':
        axx.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axx.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    if not hengzuobiaokeduxian:
        axx.tick_params(axis='x', length=0)

    return fig, axx

def setpgf(texsystem="xelatex", font_family="serif"):
    import matplotlib as mpl
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": texsystem,
        "font.family": font_family,
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.preamble": r"""
            \usepackage{xeCJK}
            \setCJKmainfont{Hiragino Mincho ProN}
            \usepackage{amsmath, amssymb}
            \usepackage{upgreek}
            \usepackage{siunitx}
        """
    })
# 自定义 setpdf：使用动态 pdf 打印器，支持 LaTeX 公式 + 日语 + 希腊字 + 英文

def setpdf():
    import matplotlib as mpl
    mpl.use("pgf")
    mpl.rcParams.update({
        "text.usetex": True,
        "pgf.texsystem": "xelatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "pgf.preamble": (
            r"\usepackage{xeCJK}" "\n"
            r"\setCJKmainfont{Hiragino Mincho ProN}" "\n"
            r"\usepackage{amsmath, amssymb}" "\n"
            r"\usepackage{siunitx}"
        ),
    })
    
def savepgf(savepath, filename, tight=True):
    """
    保存当前图为 .pgf 文件，路径为 savepath/filename.pgf.\hs{}
    如果 filename 不带扩展名会自动加上.\hs{}
    可选是否启用 tight 裁剪.\hs{}
    """
    if not filename.endswith(".pgf"):
        filename += ".pgf"
    full_path = os.path.join(savepath, filename)
    plt.tight_layout()
    if tight:
        plt.savefig(full_path, bbox_inches='tight', format='pgf')
    else:
        plt.savefig(full_path, format='pgf')
    plt.close()
def savepdf(savepath, filename, tight=True):
    """
    保存当前图为 .pdf 文件，路径为 savepath/filename.pdf.\hs{}
    如果 filename 不带扩展名会自动加上.\hs{}
    可选是否启用 tight 裁剪.\hs{}
    """
    import os
    import matplotlib.pyplot as plt

    if not filename.endswith(".pdf"):
        filename += ".pdf"
    full_path = os.path.join(savepath, filename)
    plt.tight_layout()
    if tight:
        plt.savefig(full_path, bbox_inches='tight', format='pdf')
    else:
        plt.savefig(full_path, format='pdf')
    plt.close()
def zxecf(x, y, dim=1,error=False,printt=False,draw=False,drawcolor='red',tag=True,kuangkuang=True,drawlabel=True,zuobiao=None):#最小二乘法拟合
#
#
#
#
#
    init_params=np.zeros(dim+1)
    init_params[0] = 1.0  # 初始值为1.0
#def fws(data_x,data_y,init_params):
    def get_residuals(params):#残差ベルトルを計算する
        return(y - dxs(x, params,len(params)-1))
    result = least_squares(get_residuals, init_params)

    
    if dim == 1:
        params, cov = np.polyfit(x,y, deg=dim, cov=True)
        slope = params[0]#斜率
        intercept = params[1]#截距 
        # 计算斜率和截距的误差
        slope_err = np.sqrt(cov[0][0])
        intercept_err = np.sqrt(cov[1][1])
        # 手动计算 R^2
        y_fit = slope * x + intercept
        ss_res = np.sum((y - y_fit) ** 2)         # 残差平方和
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # 总平方和
        r_squared = 1 - ss_res / ss_tot
        x0 = -1*intercept / slope  # x 截距
        #零点误差　限界振動数の不確かさ
        zero_error = np.sqrt( ((intercept/(slope*slope))**2)* slope_err**2 + (intercept_err/slope)**2 -2*(intercept/(slope)**3)*cov[0][1] )
        if printt==True:
            print('パラメータ:', result.x)
            print(f"傾き = {slope:.3e} ± {slope_err:.3e}")
            print(f"y切片 = {intercept:.3f} ± {intercept_err:.3f}")
            print(f'x切片={x0:.3e}±{zero_error:.3e}')
        if draw==True:
            #plt.axline((0, intercept), slope=slope, color=drawcolor, linestyle='-', linewidth=0.5,label='近似線\u3000\u3000\u3000')  # 拟合线
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = slope * x_fit + intercept
            ax = plt.gca()  # ⬅️ 确保加上
            if drawlabel == True:
                ax.plot(x_fit, y_fit, color=drawcolor, linestyle='-', linewidth=0.5, label='近似線\u3000\u3000')
            elif drawlabel == False:
                ax.plot(x_fit, y_fit, color=drawcolor, linestyle='-', linewidth=0.5)
            if tag == True:
                linetag(slope, slope_err, intercept, intercept_err, r_squared, loc='upper left', digits=1,kuangkuang=kuangkuang,zuobiao=zuobiao)
        return (slope, slope_err, intercept, intercept_err, r_squared, x0, zero_error)

def lingxingmarker(chang=1.0, kuan=0.3):
    """
    返回一个横向扁棱形（◊）marker,用于 matplotlib 自定义.\hs{}
    - width: 水平方向宽度
    - height: 垂直方向高度（越小越扁）
    """
    verts = np.array([
        [-chang, 0.0],
        [0.0, kuan],
        [chang, 0.0],
        [0.0, -kuan],
        [-chang, 0.0],
    ])
    return MarkerStyle(Path(verts))
def bee(hz,playtime=1):
    fs = 44100  # 采样率
    t = np.linspace(0, playtime, int(fs * playtime), endpoint=False)
    f = hz  # 频率 A4

    # 方波：典型的 8bit 音色
    square_wave = 0.3 * np.sign(np.sin(2 * np.pi * f * t))

    # 三角波
    triangle_wave = 0.3 * 2 * np.abs(2 * ((t * f) % 1) - 1) - 0.3

    # 锯齿波
    sawtooth_wave = 0.3 * (2 * (t * f % 1) - 1)

    # 噪声（GameBoy 的鼓声）
    noise = 0.2 * np.random.uniform(-1, 1, size=t.shape)

    # 播放
    sd.play(square_wave, samplerate=fs)
    #time.sleep(playtime)  # 等待播放完成
    sd.wait()


def playmusic(yuepu, jiepai, basefreq=440, yipaishijian=0.3, fs=44100):
    unit_time = yipaishijian  # 每个节拍的时长
    full_wave = []

    for note, beat in zip(yuepu, jiepai):
        duration = unit_time * beat
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        if note in ['k', 'kong', None]:  # 休止符：静音波形
            wave = np.zeros_like(t)
        else:
            hz = basefreq * 2 ** ((note - 1) / 12)
            wave = 0.3 * np.sign(np.sin(2 * np.pi * hz * t))  # 方波

        # 加个淡入淡出避免爆音
        fade_len = int(0.01 * fs)
        if len(wave) >= 2 * fade_len:
            envelope = np.ones_like(wave)
            envelope[:fade_len] = np.linspace(0, 1, fade_len)
            envelope[-fade_len:] = np.linspace(1, 0, fade_len)
            wave *= envelope

        full_wave.append(wave)

    final_wave = np.concatenate(full_wave)
    sd.play(final_wave, samplerate=fs)
    sd.wait()
    
def legend_like(ax, text, xy=(0.96, 0.96), loc='upper right', size=9, frame=True):
    """
    在 ax 内部创建不会漂移的图例式文本块
    - text：字符串，换行分 label
    - xy：在 ax.transAxes 比例坐标
    - loc：对齐方式（upper right / center / ...）
    """
    anchored = AnchoredText(
        text,
        loc=loc,
        prop=dict(size=size),
        frameon=frame,
        bbox_transform=ax.transAxes,
        borderpad=0.4
    )
    ax.add_artist(anchored)
    return anchored

def pm(a):#print_matrix(a)
    for row in a:
        print("  ".join(f"\033[92m{x:.1f}\033[0m" for x in row))

def GJ(a):
    for i in range(len(a)):
        w1 = a[i][i]
        for j in range(i,len(a[0])):
            a[i][j] = a[i][j]/w1
        for k in range(len(a)):
            w2 = a[k][i]
            if k != i:
                for j in range(i,len(a[0])):
                    a[k][j] = a[k][j] - w2*a[i][j]
        pm(a)
    return a

def GJ_1(a):
    n = len(a)
    I = np.identity(n)
    a = np.hstack((a, I))
    for i in range(len(a)):
        w1 = a[i][i]
        for j in range(i,len(a[0])):
            a[i][j] = a[i][j]/w1
        for k in range(len(a)):
            w2 = a[k][i]
            if k != i:
                for j in range(i,len(a[0])):
                    a[k][j] = a[k][j] - w2*a[i][j]   
    result = a[:, n:] 
    pm(result)
    return result

#Gaussの消去法
def G(a):
    n= len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            alpha = -a[i][k]/a[k][k]
            for j in range(k,n+1):
                a[i][j]=a[i][j]+alpha*a[k][j]
    pm(a)
    x= np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i]=( a[i][len(a[0])-1]- np.dot(a[i][i+1:n], x[i+1:n]))/a[i][i]
    x1= x[::-1]
    print("\033[33m", np.round(x1, 3), "\033[0m")
    return x1

def GS(a):  # Gauss-Seidel 法
    n_row = len(a)
    x = np.zeros(n_row)  # 初期値：全部设为0
    for n in range(50):  # 最多50次
        e = 0
        for i in range(n_row):
            w = a[i][-1]  # b_i
            for j in range(n_row):
                if j != i:
                    w -= a[i][j] * x[j]
            new_xi = w / a[i][i]
            e += abs(x[i] - new_xi)
            x[i] = new_xi
        print(f"{n+1} 次: x = {x}, 誤差 e = {e:.7f}")
        if e < 1e-7:
            break
    return x

#A a = b

def m_Ab(x, y, m):
    Ab= np.zeros((m+1, m+2))
    
    for i in range(m+1):
        for j in range(m+1):
            Ab[i][j] = np.sum(x**(i + j))  # A部分：x^i * x^j = x^{i+j}
        Ab[i][-1] = np.sum(y * x**i)      # b部分：y * x^i

    return Ab