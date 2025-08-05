# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
JSON 格式元数据保存器
"""
import os
import json
import logging
from typing import Dict, Any, Optional
import torch
import numpy as np

from .base import BaseSaver

class JsonSaver(BaseSaver):
    """
    将模型元数据保存为 JSON 格式
    """
    def __init__(self,
                 output_dir: str,
                 model_name: str = "model",
                 logger: Optional[logging.Logger] = None):
        super().__init__(output_dir, logger)
        # 使用 model_name 构建元数据文件名
        self.meta_filename = f"{model_name}.json"

    def save_model(self, model_data: Dict[str, Any]):
        metadata_to_save = model_data.get('meta', {})
        if not metadata_to_save:
            self.logger.warning("模型数据中没有找到可保存的元数据。")
            return

        output_path = os.path.join(self.output_dir, self.meta_filename)
        
        try:
            serializable_meta = self._make_json_serializable(metadata_to_save)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_meta, f, indent=2, ensure_ascii=False)
            self.logger.info(f"元数据已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存 JSON 文件失败: {e}", exc_info=True)

    def _make_json_serializable(self, obj: Any) -> Any:
        """递归转换对象为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return f"Tensor(shape={list(obj.shape)}, dtype={str(obj.dtype)})"
        elif isinstance(obj, np.ndarray):
            return f"ndarray(shape={list(obj.shape)}, dtype={str(obj.dtype)})"
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            try:
                return str(obj)
            except Exception:
                return f"<non-serializable: {type(obj).__name__}>"
