# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
模型信息提取器
负责从模型中提取用于保存的权重和元数据
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Tuple
import logging

from ..components.flat_linear import FlatQuantizedLinear, FakeQuantizedLinear, FlatNormWrapper
from ..components.quantizer import ActivationQuantizer, WeightQuantizer
from ..components.quantizer import asym_quant
from ..processors.flat_quant import QuantType

# 模块提取器注册表
_REGISTRY: Dict[type, Callable] = {}

def register_exporter(module_type: type) -> Callable:
    """注册一个模块导出器的装饰器"""
    def decorator(func: Callable) -> Callable:
        _REGISTRY[module_type] = func
        return func
    return decorator


class ModelExporter:
    """
    模型导出器
    - 遍历模型的所有模块
    - 根据模块类型调用注册的提取函数
    - 聚合所有模块的数据和元数据
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def export_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        导出模型信息
        
        Args:
            model: PyTorch 模型
            
        Returns:
            一个包含 'meta' 和 'data' 的字典
        """
        exported_meta = {}
        exported_data = {}
        
        for name, module in model.named_modules():
            # 只有叶子模块才需要处理
            if len(list(module.children())) > 0 and type(module) not in _REGISTRY:
                continue
            
            # 获取该模块类型的提取器，如果没有则使用默认提取器
            exporter_func = _REGISTRY.get(type(module), _extract_default_module)
            
            meta, data = exporter_func(name, module)
            
            if meta:
                exported_meta[name] = meta
            if data:
                exported_data.update(data)
        
        self.logger.info(f"模型导出完成, 共处理 {len(exported_meta)} 个模块。")

        return {
            'meta': exported_meta,
            'data': exported_data,
        }


@register_exporter(FakeQuantizedLinear)
def _extract_fake_linear(name: str, module: FakeQuantizedLinear) -> Tuple[Dict, Dict]:
    """提取 FakeQuantizedLinear 层的数据"""
    meta = {
        'quant_type': module.model_quant_type,
        'params': {}
    }
    data = {}
    if module.model_quant_type == "W8A8":
        raise NotImplementedError("W8A8 quantization is not supported for saving")

    # 提取量化权重
    if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
        weight_scale, weight_offset = module.weight_quantizer.get_scale_zero(module.weight)
        quant_weight = asym_quant(module.weight, weight_scale, weight_offset, module.weight_quantizer.bits, True)[0]
        quant_weight = quant_weight.to(torch.int8).contiguous()
        data[f"{name}.weight"] = quant_weight.cpu().clone()
        data[f"{name}.weight_scale"] = weight_scale.cpu().clone()
        data[f"{name}.weight_offset"] = weight_offset.cpu().clone()
        meta['params']['weight'] = {
            "shape": quant_weight.shape,
            "dtype": quant_weight.dtype,
        }
        meta['params']['weight_scale'] = {
            "shape": weight_scale.shape,
            "dtype": weight_scale.dtype,
        }
        meta['params']['weight_offset'] = {
            "shape": weight_offset.shape,
            "dtype": weight_offset.dtype,
        }
    return meta, data
        

@register_exporter(FlatQuantizedLinear)
def _extract_flat_linear(name: str, module: FlatQuantizedLinear) -> Tuple[Dict, Dict]:
    """提取 FlatQuantizedLinear 层的数据"""
    meta = {
        'quant_type': getattr(module, 'model_quant_type', QuantType.W4A4_FLATQUANT_DYNAMIC),
        'params': {}
    }
    data = {}

    # 提取量化权重
    if hasattr(module, 'weight_quantizer') and module.weight_quantizer is not None:
        weight_scale, weight_offset = module.weight_quantizer.get_scale_zero(module.weight)
        quant_weight = asym_quant(module.weight, weight_scale, weight_offset, module.weight_quantizer.bits, True)[0]
        quant_weight = quant_weight.to(torch.int8).contiguous()
        data[f"{name}.weight"] = quant_weight.cpu().clone()
        data[f"{name}.weight_scale"] = weight_scale.cpu().clone()
        data[f"{name}.weight_offset"] = weight_offset.cpu().clone()
        meta['params']['weight'] = {
            "shape": quant_weight.shape,
            "dtype": quant_weight.dtype,
        }
        meta['params']['weight_scale'] = {
            "shape": weight_scale.shape,
            "dtype": weight_scale.dtype,
        }
        meta['params']['weight_offset'] = {
            "shape": weight_offset.shape,
            "dtype": weight_offset.dtype,
        }
    # 提取偏置
    if hasattr(module, 'bias') and module.bias is not None:
        data[f"{name}.bias"] = module.bias.detach().cpu().clone()
        meta['params']['bias'] = {
            "shape": module.bias.shape,
            "dtype": module.bias.dtype,
        }
    # 提取变换参数
    if hasattr(module, "save_trans") and module.save_trans is not None:
        if hasattr(module.save_trans, 'get_save_params'):
            save_trans = module.save_trans.get_save_params()
            for key, param in save_trans.items():
                data[f"{name}.{key}"] = param.cpu().clone()
                meta['params'][f"{key}"] = {
                    "shape": param.shape,
                    "dtype": param.dtype,
                }
                

    # 提取激活量化参数
    if hasattr(module, 'act_quantizer') and hasattr(module.act_quantizer, 'get_clip_ratio'):
        clip_ratio = module.act_quantizer.get_clip_ratio()
        if clip_ratio is not None:
            data[f"{name}.clip_ratio"] = clip_ratio.cpu().clone()
            meta['params']['clip_ratio'] = {
                "shape": clip_ratio.shape,
                "dtype": clip_ratio.dtype,
            }
    return meta, data


@register_exporter(FlatNormWrapper)
def _extract_norm_wrapper(name: str, module: FlatNormWrapper) -> Tuple[Dict, Dict]:
    """提取 FlatNormWrapper 层的数据"""
    meta = {'quant_type': 'float', 'params': {}}
    data = {}
    
    norm_module = getattr(module, 'norm', getattr(module, 'norm_layer', None))
    if norm_module is not None:
        if hasattr(norm_module, 'weight') and norm_module.weight is not None:
            data[f"{name}.weight"] = norm_module.weight.cpu().clone()
            meta['params']['weight'] = {
                "shape": norm_module.weight.shape,
                "dtype": norm_module.weight.dtype,
            }
        if hasattr(norm_module, 'bias') and norm_module.bias is not None:
            data[f"{name}.bias"] = norm_module.bias.cpu().clone()
            meta['params']['bias'] = {
                "shape": norm_module.bias.shape,
                "dtype": norm_module.bias.dtype,
            }

    return meta, data

def _extract_default_module(name: str, module: nn.Module) -> Tuple[Dict, Dict]:
    """默认提取器，用于未注册的模块类型，将其参数按float类型保存"""
    params = dict(module.named_parameters(recurse=False))   

    if not params:
        return None, None

    meta = {'quant_type': 'float', 'params': {}}
    data = {}
    for param_name, param in params.items():
        data[f"{name}.{param_name}"] = param.detach().cpu().clone()
        meta['params'][f"{param_name}"] = {
            "shape": param.shape,
            "dtype": param.dtype,
        }

    return meta, data
