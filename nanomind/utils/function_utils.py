# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import math
from collections import OrderedDict
import torch
from torch import nn
import numpy as np
from scipy.linalg import qr

npu_available = False
try:
    import torch_npu
except ImportError:
    pass
else:
    npu_available = True
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


def get_init_scale(w_smax, x_smax, alpha=0.5):
    return (w_smax.pow(1 - alpha) / x_smax.pow(alpha)).clamp(min=1e-5)


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


def get_random_orthg(size):
    h = np.random.randn(size, size)
    q, r = qr(h)
    q_modified = q @ np.diag(np.sign(np.diag(r)))
    return torch.from_numpy(q_modified)


def get_init_weight(dim, ):
    return get_random_orthg(dim)


def get_inverse(matrix):
    dtype = matrix.dtype
    if not npu_available:
        return matrix.double().inverse().to(dtype)
    else:
        device = matrix.device
        return matrix.cpu().double().inverse().to(device=device, dtype=dtype)


def get_n_set_parameters_byname(model, required_names):
    params = []
    for r_name in required_names:
        for name, param in model.named_parameters():
            if name.find(r_name) > -1:
                params.append(param)
    for param in params:
        param.requires_grad = True
    return iter(params)


def set_require_grad_all(model, requires_grad):
    for _, param in model.named_parameters():
        param.requires_grad = requires_grad
    return