# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
SafeTensors 格式保存器
"""
import os
import logging
from typing import Dict, Any, Optional

from .base import BaseSaver

class SafeTensorsSaver(BaseSaver):
    """
    将模型权重保存为 SafeTensors 格式
    """
    def __init__(self,
                 output_dir: str,
                 model_name: str = "model",
                 logger: Optional[logging.Logger] = None):
        super().__init__(output_dir, logger)
        self.model_name = model_name
        try:
            import safetensors.torch
            self.safetensors = safetensors.torch
            self._available = True
        except ImportError:
            self.logger.warning("safetensors 未安装，无法使用 SafeTensorsSaver")
            self._available = False

    def save_model(self, model_data: Dict[str, Any]):
        if not self._available:
            self.logger.error("safetensors 不可用，跳过保存。")
            return

        tensors_to_save = model_data.get('data', {})
        if not tensors_to_save:
            self.logger.warning("模型数据中没有找到可保存的张量。")
            return

        output_path = os.path.join(self.output_dir, f"{self.model_name}.safetensors")
        
        try:
            self.safetensors.save_file(tensors_to_save, output_path)
            self.logger.info(f"权重已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存 SafeTensors 文件失败: {e}", exc_info=True)
