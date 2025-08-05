# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
保存模块工具函数
"""
import os
import torch
import logging
from typing import Dict, Any, Optional, Union
from . import SaverFactory, ModelExporter, SaveFormat


def save_model(model: torch.nn.Module,
               output_dir: str,
               model_name: str = "model",
               logger: Optional[logging.Logger] = None) -> str:
    """
    保存模型的便利函数，支持量化和非量化模型。

    Args:
        model: PyTorch 模型
        output_dir: 输出目录
        model_name: 模型名称（不含扩展名）
        logger: 日志记录器

    Returns:
        输出目录的路径
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 1. 导出模型数据
    logger.info("开始导出模型数据...")
    exporter = ModelExporter(logger)
    model_data = exporter.export_model(model)

    # 2. 分别创建并使用 safetensors 和 json 保存器
    logger.info("创建 SafeTensors 保存器以保存权重...")
    safetensors_saver = SaverFactory.create(
        save_format=SaveFormat.SAFETENSORS,
        output_dir=output_dir,
        logger=logger,
        model_name=model_name
    )
    safetensors_saver.save_model(model_data)

    logger.info("创建 JSON 保存器以保存元数据...")
    json_saver = SaverFactory.create(
        save_format=SaveFormat.JSON,
        output_dir=output_dir,
        logger=logger,
        model_name=model_name
    )
    json_saver.save_model(model_data)

    logger.info("模型保存完成!")
    logger.info(f"  - 权重文件: {os.path.join(output_dir, model_name)}.safetensors")
    logger.info(f"  - 元数据文件: {os.path.join(output_dir, model_name)}.json")
    
    return output_dir


def load_model_data(model_path: str,
                    device: Union[str, torch.device] = 'cpu',
                    logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """
    加载模型权重数据。目前只支持 SafeTensors 格式。

    Args:
        model_path: 模型权重文件路径 (.safetensors)
        device: 加载到的设备
        logger: 日志记录器

    Returns:
        包含模型权重张量的字典
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    ext = os.path.splitext(model_path)[1].lower()
    if ext != '.safetensors':
        raise ValueError(f"当前只支持加载 .safetensors 格式文件，但收到: {ext}")

    logger.info("加载 SafeTensors 格式模型...")
    try:
        import safetensors.torch
        model_data = safetensors.torch.load_file(model_path, device=str(device))
    except ImportError:
        raise ImportError("需要安装 safetensors: pip install safetensors")

    logger.info(f"模型加载完成，共 {len(model_data)} 个张量。")
    return model_data
