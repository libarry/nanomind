# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
保存格式定义
"""
from enum import Enum

class SaveFormat(Enum):
    """支持的保存格式"""
    SAFETENSORS = "safetensors"
    JSON = "json"
