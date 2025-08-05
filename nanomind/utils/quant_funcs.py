# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
独立的量化函数实现，不依赖外部模块
"""
import torch
import torch.nn as nn


class StatMinMaxObserver(nn.Module):
    """用于统计激活值最大最小值的观察器，PyTorch 实现"""
    
    def __init__(self, bit=8, symmetric=False, per_tensor=True):
        super(StatMinMaxObserver, self).__init__()
        self.bit = bit
        self.symmetric = symmetric
        self.per_tensor = per_tensor
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('sample_count', torch.tensor(0))
        
    def forward(self, x):
        """更新统计信息"""
        return self.update(x)
    
    def update(self, x):
        """更新最大最小值统计"""
        if self.per_tensor:
            batch_min = x.min()
            batch_max = x.max()
        else:
            # 按 channel 统计
            batch_min = x.view(-1, x.shape[-1]).min(0)[0]
            batch_max = x.view(-1, x.shape[-1]).max(0)[0]
        
        # 更新全局统计
        if self.sample_count == 0:
            self.min_val.copy_(batch_min)
            self.max_val.copy_(batch_max)
        else:
            self.min_val = torch.min(self.min_val, batch_min)
            self.max_val = torch.max(self.max_val, batch_max)
        
        self.sample_count += 1
        return self.min_val, self.max_val
    
    def get_min_max(self, device=None):
        """获取统计的最大最小值"""
        min_val = self.min_val
        max_val = self.max_val
        
        if device is not None and device != "cpu":
            min_val = min_val.to(device)
            max_val = max_val.to(device)
        
        return min_val, max_val
    
    def reset(self):
        """重置统计信息"""
        self.min_val.fill_(float('inf'))
        self.max_val.fill_(float('-inf'))
        self.sample_count.fill_(0)


def linear_quantization_params(bit, x_min, x_max, q_signed=True, sym=False):
    """
    计算线性量化的 scale 和 zero_point 参数
    
    Args:
        bit: 量化位数
        x_min: 输入张量的最小值
        x_max: 输入张量的最大值
        q_signed: 是否使用有符号量化
        sym: 是否使用对称量化
    
    Returns:
        scale, zero_point
    """
    # 确保输入是 tensor
    if not isinstance(x_min, torch.Tensor):
        x_min = torch.tensor(x_min)
    if not isinstance(x_max, torch.Tensor):
        x_max = torch.tensor(x_max)
    
    # 确保 min <= 0, max >= 0
    x_min = torch.min(x_min, torch.zeros_like(x_min))
    x_max = torch.max(x_max, torch.zeros_like(x_max))
    
    # 计算量化范围
    if q_signed:
        qmin = -(2 ** (bit - 1))
        qmax = 2 ** (bit - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bit - 1
    
    if sym:
        # 对称量化
        abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
        scale = abs_max / (qmax if q_signed else qmax / 2)
        zero_point = torch.zeros_like(scale)
        if not q_signed:
            zero_point = torch.full_like(scale, qmax / 2)
    else:
        # 非对称量化
        scale = (x_max - x_min) / (qmax - qmin)
        scale = torch.clamp(scale, min=1e-8)  # 避免除零
        zero_point = qmin - x_min / scale
        zero_point = torch.round(zero_point)
        zero_point = torch.clamp(zero_point, qmin, qmax)
    
    # 确保 scale 不为零
    scale = torch.clamp(scale, min=1e-8)
    
    return scale, zero_point


def fake_quantize_linear(input_tensor, scale, zero_point, bit=8, signed=True):
    """
    伪量化函数
    
    Args:
        input_tensor: 输入张量
        scale: 缩放因子
        zero_point: 零点
        bit: 量化位数
        signed: 是否有符号
    
    Returns:
        伪量化后的张量
    """
    if signed:
        qmin = -(2 ** (bit - 1))
        qmax = 2 ** (bit - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bit - 1
    
    # 量化
    q_input = torch.round(input_tensor / scale + zero_point)
    q_input = torch.clamp(q_input, qmin, qmax)
    
    # 反量化
    dq_input = (q_input - zero_point) * scale
    
    return dq_input