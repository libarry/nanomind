from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist

from vllm.distributed import GroupCoordinator
from nanomind.vllm_adapter.quant_utils import get_scale_zero, get_qmin_qmax
 



class NanomindW8A8DynamicLinearMethod:

    def __init__(self):
        self.transpose_weight = True

    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
        output_size: int,
        params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  1,
                                                  dtype=params_dtype)
        params_dict["weight_offset"] = torch.empty(output_size,
                                                   1,
                                                   dtype=params_dtype)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        config = getattr(layer, "_quant_config", {})
        if not isinstance(x, tuple):
            output_dtype = config.get("output_dtype", x.dtype)
            dynamic_scale, dynamic_offset = get_scale_zero(x, bits=8, sym=True)
            q_max, q_min = get_qmin_qmax(8, True)
            quantized_x = torch.round(x / dynamic_scale + dynamic_offset).clamp(q_min, q_max)

        else:
            assert "output_dtype" in config.keys(), (
                f"DynamicLinearMethod needs explicitly specified `output_dtype`"
                f"for pre-quantized input, got config [{config}]")
            output_dtype = config["output_dtype"]
            quantized_x, dynamic_scale = x
        pertoken_scale = (dynamic_scale
                          if config.get("pertoken_scale", True) else None)
        dequantized_w = layer.weight * layer.weight_scale
        dequantized_x = quantized_x * dynamic_scale
        output = torch.matmul(dequantized_x, dequantized_w)
        if bias is not None:
            output += bias
        return ((output, dynamic_scale)
                if config.get("return_scale", False) else output)

    def process_weights_after_loading(self, layer):
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        # cast quantized weight tensors in NZ format (29) for higher inference speed
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
