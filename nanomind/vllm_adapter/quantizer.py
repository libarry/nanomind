import importlib
import sys
import types
from typing import Any, Dict, List, Optional

from vllm.logger import logger

from .w8a8_dynamic import NanomindW8A8DynamicLinearMethod
from .w4a4_flatquant_dynamic import NanomindW4A4FlatQuantDynamicFakeLinearMethod, NanomindW4A4FlatQuantDynamicLinearMethod


CUSTOMIZED_QUANTIZER_TYPE: List[str] = []




class NanomindQuantizer:
    _instance: Optional[object] = None
    patched = False

    def __init__(self, quant_description):
        pass

    @staticmethod
    def build_linear_method():
        raise NotImplementedError(
            "Linear method is not implemented for the current quant type.")

    @staticmethod
    def build_moe_method():
        raise NotImplementedError(
            "MoE method is not implemented for the current quant type.")

    @staticmethod
    def build_attention_method():
        raise NotImplementedError(
            "Attention method is not implemented for the current quant type.")

    @staticmethod
    def get_linear_quant_type(quant_description: Dict[str, Any], prefix: str,
                              packed_modules_mapping: Dict[str, Any]):
        proj_name = prefix.split(".")[-1]
        if proj_name in packed_modules_mapping:
            quant_type = None
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in packed_modules_mapping[proj_name]
            ]
            for shard_prefix in shard_prefixes:
                shard_quant_type = quant_description[shard_prefix]['quant_type']

                if quant_type is None:
                    quant_type = shard_quant_type
                elif shard_quant_type != quant_type:
                    raise ValueError(
                        f"Not all shards of {prefix} are quantized with same quant type."
                        f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                        f"use {quant_type}. Please check quantization config.")
        else:
            quant_type = quant_description[prefix]['quant_type']
        return quant_type

    @classmethod
    def get_quantizer(cls,
                      quant_description: Dict[str, Any],
                      prefix: str,
                      packed_modules_mapping: Optional[Dict[str, Any]] = None):
        if packed_modules_mapping is None:
            packed_modules_mapping = dict()

        quant_type = cls.get_linear_quant_type(quant_description, prefix,
                                                packed_modules_mapping)
        if quant_type in SUPPORT_NANOMIND_QUANTIZER_TYPE.keys():
            cls = SUPPORT_NANOMIND_QUANTIZER_TYPE[quant_type]
            if not cls._instance:
                cls._instance = cls(quant_description)
            return cls._instance
        raise NotImplementedError("Currently, nanomind only supports following quant types:" \
                                  f"{list(SUPPORT_NANOMIND_QUANTIZER_TYPE.keys())}")


class W8A8DYNAMICQuantizer(NanomindQuantizer):

    @staticmethod
    def build_linear_method():
        return NanomindW8A8DynamicLinearMethod()


class W4A4FLATQUANTDYNAMICQuantizer(NanomindQuantizer):
    @staticmethod
    def build_linear_method():
        return NanomindW4A4FlatQuantDynamicFakeLinearMethod()
        # return NanomindW4A4FlatQuantDynamicLinearMethod()


SUPPORT_NANOMIND_QUANTIZER_TYPE = {
    "W8A8_DYNAMIC": W8A8DYNAMICQuantizer,
    "W4A4_FLATQUANT_DYNAMIC": W4A4FLATQUANTDYNAMICQuantizer,
}
