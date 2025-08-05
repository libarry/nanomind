# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
保存器工厂
"""
import logging
from typing import List, Union, Optional

from ..formats import SaveFormat
from .base import BaseSaver
from .json_saver import JsonSaver
from .safetensors_saver import SafeTensorsSaver
from .multi_saver import MultiSaver

class SaverFactory:
    """
    根据指定的格式创建和配置保存器
    """
    _format_map = {
        SaveFormat.SAFETENSORS: SafeTensorsSaver,
        SaveFormat.JSON: JsonSaver,
    }

    @classmethod
    def create(cls,
               save_format: Union[SaveFormat, str, List[Union[SaveFormat, str]]],
               output_dir: str,
               logger: Optional[logging.Logger] = None,
               **kwargs) -> BaseSaver:
        """
        创建保存器实例

        Args:
            save_format: 保存格式或格式列表
            output_dir: 输出目录
            logger: 日志记录器
            **kwargs: 传递给具体保存器的额外参数 (如 model_name)

        Returns:
            一个 BaseSaver 实例
        """
        if logger is None:
            logger = logging.getLogger(cls.__name__)

        if not isinstance(save_format, list):
            save_format_list = [save_format]
        else:
            save_format_list = save_format

        savers = []
        for fmt in save_format_list:
            if isinstance(fmt, str):
                try:
                    fmt_enum = SaveFormat(fmt.lower())
                except ValueError:
                    logger.warning(f"不支持的保存格式字符串: '{fmt}'，已跳过。")
                    continue
            else:
                fmt_enum = fmt

            saver_class = cls._format_map.get(fmt_enum)
            if saver_class:
                # 将 output_dir, logger, 以及其他所有kwargs传递给构造函数
                savers.append(saver_class(output_dir=output_dir, logger=logger, **kwargs))
            else:
                logger.warning(f"未找到格式 {fmt_enum} 对应的保存器，已跳过。")

        if not savers:
            raise ValueError("没有成功创建任何保存器。请检查 save_format 参数。")

        if len(savers) == 1:
            logger.info(f"已创建单个保存器: {savers[0].__class__.__name__}")
            return savers[0]
        else:
            logger.info(f"已创建 MultiSaver，包含 {len(savers)} 个保存器。")
            return MultiSaver(output_dir=output_dir, savers=savers, logger=logger)

    @classmethod
    def is_format_supported(cls, save_format: Union[SaveFormat, str]) -> bool:
        """
        检查是否支持某种保存格式

        Args:
            save_format: 保存格式

        Returns:
            如果支持则返回 True，否则返回 False
        """
        if isinstance(save_format, str):
            try:
                save_format = SaveFormat(save_format.lower())
            except ValueError:
                return False
        return save_format in cls._format_map
