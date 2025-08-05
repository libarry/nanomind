import torch

def get_qmin_qmax(bits, sym):
    """获取量化范围"""
    if sym:
        q_max = 2 ** (bits - 1) - 1  # int4: 7
        q_min = -q_max - 1  # int4: -8
    else:
        q_max, q_min = 2 ** bits - 1, 0
    return q_max, q_min


def get_scale_zero(x, bits=8, sym=True):
    """
    获取动态量化的scale和zero_point参数（per-token）
    模拟量化参数计算
    """
    q_max, q_min = get_qmin_qmax(bits, sym)  # int4对称量化: [-8, 7]
    init_shape = x.shape
    reshaped_x = x.reshape((-1, x.shape[-1]))
    xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
    tmp = torch.zeros_like(xmax)
    xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

    
    if sym:
        # 对称量化：使用绝对值最大值
        xmax = torch.maximum(torch.abs(xmin), xmax)
        scale = (xmax / q_max).clamp(min=1e-8)  # 避免除零
        scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
        zero = torch.zeros_like(scale)
    else:
        # 非对称量化
        scale = (xmax - xmin) / (q_max - q_min)
        zero = torch.round(-xmin / scale)
        scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)   
        zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

    return scale, zero