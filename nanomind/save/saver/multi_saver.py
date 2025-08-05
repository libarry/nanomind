# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
多格式保存器
"""
from typing import List, Dict, Any, Optional
import logging

from .base import BaseSaver

class MultiSaver(BaseSaver):
    """
    一个可以组合多个保存器的容器
    """
    def __init__(self,
                 output_dir: str,
                 savers: List[BaseSaver],
                 logger: Optional[logging.Logger] = None):
        super().__init__(output_dir, logger)
        if not savers:
            raise ValueError("至少需要提供一个保存器。")
        self.savers = savers
        self.logger.info(f"初始化 MultiSaver，包含 {len(self.savers)} 个保存器: {[s.__class__.__name__ for s in self.savers]}")

    def save_model(self, model_data: Dict[str, Any]):
        """
        依次调用所有子保存器的 save_model 方法

        Args:
            model_data: 包含 'meta' 和 'data' 的模型数据字典
        """
        self.logger.info("开始使用 MultiSaver 保存模型...")
        for saver in self.savers:
            try:
                saver.save_model(model_data)
            except Exception as e:
                self.logger.error(f"子保存器 {saver.__class__.__name__} 执行失败: {e}", exc_info=True)
        self.logger.info("MultiSaver 保存流程完成。")
