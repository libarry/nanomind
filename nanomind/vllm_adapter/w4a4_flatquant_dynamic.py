import math
import torch
from typing import Any, Dict, Optional
try:
    from quant_ops.ops.quantization.int4_matmul.ops import int4_matmul_cuda, int4_matmul_pytorch, pack_int4_to_int32
    from quant_ops.ops.quantization.int4_matmul.kernels import CUDA_KERNELS_AVAILABLE
    from quant_ops.ops.quantization.flatquant.ops import flatquant_cuda, flatquant_pytorch
except ImportError:
    pass
    


def pack_int4_weights_with_npu(weight_tensor):
    """
    使用npu_convert_weight_to_int4pack将权重打包为int4格式
    """
    return pack_int4_to_int32_manual(weight_tensor)


def pack_int4_to_int32_manual(int4_tensor):
    """
    手动将int4张量打包到int32中（备用方案）
    """
    # 确保最后一维是8的倍数
    assert int4_tensor.shape[-1] % 8 == 0, "最后一维必须是8的倍数"
    
    # 将int4值限制在[-8, 7]范围内
    int4_clamped = torch.clamp(int4_tensor, -8, 7)
    
    # 转换为uint4 [0, 15]
    uint4_tensor = int4_clamped + 8
    
    # 重塑并打包
    shape = list(uint4_tensor.shape)
    shape[-1] = shape[-1] // 8
    
    uint4_reshaped = uint4_tensor.view(*shape[:-1], -1, 8)
    
    # 打包到int32中
    packed = torch.zeros(*shape, dtype=torch.int32, device=uint4_tensor.device)
    for i in range(8):
        packed += (uint4_reshaped[..., i].to(torch.int32) << (i * 4))
    
    return packed


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


class NanomindW4A4FlatQuantDynamicLinearMethod:
    input_size = 0
    output_size = 0

    def __init__(self):
        self.transpose_weight = False
        self.sym = True  # 使用对称量化

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取权重参数字典
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            权重参数字典
        """
        # 确保输入维度是8的倍数，用于int4打包
        assert input_size % 8 == 0, f"input_size ({input_size}) must be divisible by 8 for int4 packing"
        NanomindW4A4FlatQuantDynamicLinearMethod.input_size = input_size
        NanomindW4A4FlatQuantDynamicLinearMethod.output_size = output_size
        params_dict = {
            # 原始int8保存的int4权重数据，未打包
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取per-tensor量化参数字典
        
        Args:
            params_dtype: 参数数据类型
            
        Returns:
            per-tensor参数字典
        """
        params_dict = {}
        # FlatQuant变换矩阵（左变换和右变换）
        # 实际使用时从配置文件或权重文件中加载
        left_trans_dim, right_trans_dim = get_decompose_dim(NanomindW4A4FlatQuantDynamicLinearMethod.input_size)
        params_dict["left_trans"] = torch.empty(left_trans_dim, left_trans_dim, dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim, right_trans_dim, dtype=params_dtype)
        
        # 量化截断比例参数
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """
        获取per-channel量化参数字典
        
        Args:
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            per-channel参数字典
        """
        params_dict = {}
        # 权重量化scale (per-channel)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        # 权重量化offset (per-channel)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
    
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        应用W4A4 FlatQuant动态量化的前向传播
        
        Args:
            layer: 线性层模块
            x: 输入张量
            bias: 偏置项
            tp_rank: 张量并行rank
            
        Returns:
            输出张量
        """
        original_dtype = x.dtype
        input_shape = x.shape
        in_features = input_shape[-1]
        
        # 获取FlatQuant变换矩阵维度
        M = layer.left_trans.shape[0]
        N = layer.right_trans.shape[0]
        
        # 确保 M * N == in_features
        assert M * N == in_features, f"FlatQuant transform matrices dimension mismatch: M({M}) * N({N}) != in_features({in_features})"
        # 确保变换矩阵类型与输入匹配
        left_trans_matched = layer.left_trans.to(original_dtype)
        right_trans_matched = layer.right_trans.to(original_dtype)
        
        # 重塑输入：[batch_size * seq_len, M, N] 
        x_reshaped = x.view(-1, in_features)

        quantize_x, scale = flatquant_cuda(
            x_reshaped,
            left_trans_matched,
            right_trans_matched,
            clip_ratio=layer.aclnn_clip_ratio,
            pack_int32=True
        )

        output = int4_matmul_cuda(quantize_x, layer.weight, scale, layer.weight_scale)

        # 添加bias，确保dtype一致
        output = output.view(*input_shape[:-1], -1).to(original_dtype)
        if bias is not None:
            output = output + bias.to(original_dtype)
        return output

    def process_weights_after_loading(self, layer):
        """
        权重加载后的处理步骤
        
        Args:
            layer: 线性层模块
        """
        # 1. 打包int4权重到int32格式
        # 原始权重为int8保存的int4数据，需要打包为int32
        weight_packed = pack_int4_to_int32(layer.weight.data).T.contiguous()
        
        # 转置权重（如果需要）不建议转置，npu_quant_matmul算子处理连续权重的逻辑和处理转置权重逻辑不同
        if self.transpose_weight:
            weight_packed = weight_packed.transpose(0, 1).contiguous()
        layer.weight = torch.nn.Parameter(weight_packed, requires_grad=False)
        
        # 2. 确保weight_scale和weight_offset为float32
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32).view(-1)
        layer.weight_offset.data = layer.weight_offset.data.to(torch.float32).view(-1)
        
        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.contiguous())
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data)
        # 4. 确保clip_ratio为float32标量

        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32))

        layer.aclnn_clip_ratio = layer.clip_ratio.item()
        print(f"W4A4 FlatQuant Dynamic layer initialized: "
              f"weight_shape={layer.weight.shape}, "
              f"transform_dims=({layer.left_trans.shape[0]}, {layer.right_trans.shape[0]}), "
              f"clip_ratio={layer.clip_ratio.item():.3f}") 


class NanomindW4A4FlatQuantDynamicFakeLinearMethod:
    input_size = 0
    output_size = 0

    def __init__(self):
        self.transpose_weight = False
        self.sym = True  # 使用对称量化

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取权重参数字典（伪量化版本）
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            权重参数字典
        """
        NanomindW4A4FlatQuantDynamicFakeLinearMethod.input_size = input_size
        NanomindW4A4FlatQuantDynamicFakeLinearMethod.output_size = output_size
        params_dict = {
            # 原始浮点权重，用于伪量化模拟
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        """
        获取per-tensor量化参数字典（伪量化版本）
        
        Args:
            params_dtype: 参数数据类型
            
        Returns:
            per-tensor参数字典
        """
        params_dict = {}
        # FlatQuant变换矩阵（左变换和右变换）
        left_trans_dim, right_trans_dim = get_decompose_dim(NanomindW4A4FlatQuantDynamicFakeLinearMethod.input_size)
        params_dict["left_trans"] = torch.empty(left_trans_dim, left_trans_dim, dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim, right_trans_dim, dtype=params_dtype)
        
        # 量化截断比例参数
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        """
        获取per-channel量化参数字典（伪量化版本）
        
        Args:
            output_size: 输出维度
            params_dtype: 参数数据类型
            
        Returns:
            per-channel参数字典
        """
        params_dict = {}
        # 权重量化scale (per-channel)
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=torch.float32)
        # 权重量化offset (per-channel)
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_qmin_qmax(bits, sym):
        """获取量化范围"""
        if sym:
            q_max = torch.tensor(2 ** (bits - 1) - 1)  # int4: 7
            q_min = -q_max - 1  # int4: -8
        else:
            q_max, q_min = torch.tensor(2 ** bits - 1), 0
        return q_max, q_min

    @staticmethod
    def get_scale_zero(x, clip_ratio, sym=True):
        """
        获取动态量化的scale和zero_point参数（per-token）
        模拟量化参数计算
        """
        q_max, q_min = NanomindW4A4FlatQuantDynamicFakeLinearMethod.get_qmin_qmax(4, sym)  # int4对称量化: [-8, 7]
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)

        # 应用clip_ratio,此处乘法为正确用法，但算子是做除法
        xmax = xmax * clip_ratio
        xmin = xmin * clip_ratio
        
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

    @staticmethod
    def kronecker_matmul(x, hadL, hadR):
        """kronecker乘积矩阵乘法"""
        init_shape = x.shape
        x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
        x = torch.matmul(x, hadR)
        x = torch.matmul(hadL.T, x)
        return x.reshape(init_shape)

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        应用W4A4 FlatQuant动态伪量化的前向传播
        
        Args:
            layer: 线性层模块
            x: 输入张量
            bias: 偏置项
            tp_rank: 张量并行rank
            
        Returns:
            输出张量
        """
        original_dtype = x.dtype
        input_shape = x.shape
        in_features = input_shape[-1]
        
        # 获取FlatQuant变换矩阵维度
        M = layer.left_trans.shape[0]
        N = layer.right_trans.shape[0]
        
        # 确保 M * N == in_features
        assert M * N == in_features, f"FlatQuant transform matrices dimension mismatch: M({M}) * N({N}) != in_features({in_features})"
        
        # 确保变换矩阵类型与输入匹配
        left_trans_matched = layer.left_trans.to(original_dtype)
        right_trans_matched = layer.right_trans.to(original_dtype)
        
        # 1. 先进行kronecker乘积变换（模拟FlatQuant的前处理）
        x_transformed = NanomindW4A4FlatQuantDynamicFakeLinearMethod.kronecker_matmul(
            x, left_trans_matched, right_trans_matched
        )
        
        # 2. 对变换后的数据进行动态量化模拟
        scale, zero = NanomindW4A4FlatQuantDynamicFakeLinearMethod.get_scale_zero(
            x_transformed, layer.clip_ratio, sym=True
        )
        
        # 执行量化：q = round((x - zero) / scale)
        x_quantized = torch.round(x_transformed / scale + zero).clamp(-8, 7)  # int4范围
        
        # 3. 反量化激活
        x_dequant = (x_quantized - zero) * scale
        

        x_dequant = x_dequant.view(-1, M * N)
        
        # 权重反量化
        weight_dequant = layer.weight.float() * layer.weight_scale
        
        # 5. 浮点矩阵乘法，确保类型匹配
        bias_matched = bias.to(original_dtype) if bias is not None else None

        output = torch.nn.functional.linear(
            x_dequant.to(original_dtype), 
            weight_dequant.to(original_dtype), 
            bias_matched
        )
        # 恢复原始batch维度
        out_features = layer.weight.shape[0]  # 使用原始权重的输出维度
        output = output.view(*input_shape[:-1], out_features)

        return output

    def process_weights_after_loading(self, layer):
        """
        权重加载后的处理步骤（伪量化版本）
        
        Args:
            layer: 线性层模块
        """
        # 1. 确保weight_scale为float32
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        
        layer.clip_ratio = torch.nn.Parameter(layer.clip_ratio.data.to(torch.float32))
        
        print(f"W4A4 FlatQuant Dynamic Fake layer initialized: "
              f"weight_shape={layer.weight.shape}, "
              f"transform_dims=({layer.left_trans.shape[0]}, {layer.right_trans.shape[0]}), "
              f"clip_ratio={layer.clip_ratio.item():.3f}") 