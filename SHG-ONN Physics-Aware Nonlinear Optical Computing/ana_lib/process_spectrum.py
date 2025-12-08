from .digitize import *

#这段代码定义了两个函数：splice_region 和 process_spec，用于处理光谱数据。
# 主要目的是从原始光谱中提取特定区域的数据，并对其进行归一化和数字化处理，最终输出一个固定维度的二值化数组。以下是每个函数的具体解释：

def splice_region(s, bounds):
    # 输入：s：输入的光谱数据（一维数组）。bounds：一个包含两个浮点数的列表 [left_ratio, right_ratio]，表示要截取的光谱区域的比例范围。
    len_spec = len(s)
    # 计算光谱数据的长度 len_spec。
    # 根据 bounds 中的比例计算左边界索引 left_ind 和右边界索引 right_ind。使用这两个索引从光谱数据中截取子区域。
    left_ratio, right_ratio = bounds
    left_ind = np.round(int(left_ratio*len_spec))
    right_ind = int(np.round(right_ratio*len_spec))
    out = s[left_ind:right_ind]
    # 返回截取后的子区域 out。
    # 假设 s 是一个长度为 448 的光谱数据，bounds=[0.2, 0.8]，那么 splice_region 将返回 s 中第 0.2*448 到第 0.8*448 个元素（包括左边的，不包括右边的）
    return out

spectrum_top = 25
#输入：spectrum：输入的光谱数据（一维数组）。output_dim：期望输出的二值化数组的长度。bounds：可选参数，默认为 [0.0, 1.0]，表示要截取的光谱区域的比例范围。
def process_spec(spectrum, output_dim, bounds=[0.0, 1.0]):
    #截取区域：使用 splice_region 函数根据 bounds 参数截取光谱数据的特定区域。
    spectrum = splice_region(spectrum, bounds)
    #将截取后的光谱数据除以 spectrum_top（默认为 25），进行归一化处理。
    spectrum = spectrum/spectrum_top
    #调用 digitizex 函数将归一化后的光谱数据转换为固定长度的二值化数组。
    #返回经过归一，插值和平均后的非二值数组，长度为output_dim。
    return digitizex(spectrum, output_dim)