# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
保存器基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import os

class BaseSaver(ABC):
    """保存器抽象基类"""

    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def save_model(self, model_data: Dict[str, Any]):
        """
        保存模型数据

        Args:
            model_data: 包含 'meta' 和 'data' 的模型数据字典
        """
        raise NotImplementedError
