#!/usr/bin/env python3
# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""
完整的 FlatQuant 训练和保存示例
演示如何使用独立模块进行训练并保存量化模型
"""

import logging
import argparse
import os
import re
import shutil
import glob
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from nanomind import flat_quant_train_and_save, FlatQuantConfig
from nanomind.save import SaveFormat
from nanomind.save.utils import save_model, load_model_data
from nanomind.processors.flat_quant import QuantType


class SimpleLayerConfig:
    """简单的层配置类"""
    def __init__(self, model_quant_type=QuantType.W4A4_FLATQUANT_DYNAMIC):
        self.model_quant_type = model_quant_type
        if model_quant_type == QuantType.W8A8_DYNAMIC:
            self.w_bit = 8
            self.a_bit = 8
        elif model_quant_type == QuantType.W4A4_FLATQUANT_DYNAMIC:
            self.w_bit = 4
            self.a_bit = 4
        else:
            # Fallback to default
            self.w_bit = 4
            self.a_bit = 4

        self.w_sym = True
        self.a_sym = True
        self.is_dynamic = True


def create_dummy_calib_data(tokenizer, model, num_samples=8, seq_len=512):
    """创建虚拟的校准数据"""
    calib_data = []
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Artificial intelligence will change everything.",
        "Deep learning models are becoming more powerful.",
        "Natural language processing is advancing rapidly.",
        "Computer vision tasks are being solved efficiently.",
        "Reinforcement learning enables autonomous systems.",
        "Large language models understand context better."
    ]
    
    for i in range(num_samples):
        text = texts[i % len(texts)]
        repeated_text = (text + " ") * (seq_len // len(text.split()) + 1)
        
        inputs = tokenizer(
            repeated_text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length"
        )
        
        calib_data.append(
            [value.to(model.device) for key, value in inputs.data.items() if isinstance(value, torch.Tensor)])
        

    return calib_data


def create_layer_map(model, rules=None, default_quant_type=QuantType.W4A4_FLATQUANT_DYNAMIC):
    """创建层映射配置"""
    if rules is None:
        rules = {}
    layer_map = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            matched_quant_type = None
            for pattern, quant_type in rules.items():
                if re.match(pattern, name):
                    matched_quant_type = quant_type
                    break
            
            if matched_quant_type:
                layer_map[name] = SimpleLayerConfig(matched_quant_type)
            else:
                layer_map[name] = SimpleLayerConfig(default_quant_type)
    
    return layer_map


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="完整的 FlatQuant 训练和保存示例")
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='microsoft/DialoGPT-small',
                        help='预训练模型名称')
    parser.add_argument('--device', type=str, default='auto',
                        help='运行设备')
    
    # 量化参数
    parser.add_argument('--w_bits', type=int, default=4,
                        help='权重量化位数')
    parser.add_argument('--a_bits', type=int, default=4,
                        help='激活量化位数')
    parser.add_argument('--w_asym', action='store_true',
                        help='权重是否使用非对称量化')
    parser.add_argument('--a_asym', action='store_true',
                        help='激活是否使用非对称量化')
    parser.add_argument('--lwc', action='store_true', default=True,
                        help='是否启用学习权重裁剪')
    parser.add_argument('--lac', action='store_true', default=True,
                        help='是否启用学习激活裁剪')
    parser.add_argument('--add_diag', action='store_true', default=True,
                        help='是否添加对角变换')
    parser.add_argument('--diag_alpha', type=float, default=0.3,
                        help='对角变换初始化参数')
    parser.add_argument('--diag_relu', action='store_true', default=True,
                        help='是否在对角变换中使用ReLU')
    parser.add_argument('--direct_inv', action='store_true',
                        help='是否直接求逆')
    # 训练参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=1,
                        help='训练轮数')
    parser.add_argument('--flat_lr', type=float, default=5e-3,
                        help='学习率')
    parser.add_argument('--warmup', action='store_true',
                        help='是否使用学习率预热')
    parser.add_argument('--deactive_amp', action='store_true',
                        help='是否禁用混合精度')
    parser.add_argument('--amp_dtype', type=str, default='bfloat16',
                        choices=['bfloat16', 'float16'],
                        help='混合精度数据类型')
    parser.add_argument('--quant_by_quant', action='store_true',
                        help='是否使用量化逐层训练')
    
    # 校准数据参数
    parser.add_argument('--num_calib_samples', type=int, default=8,
                        help='校准数据样本数')
    parser.add_argument('--calib_seq_len', type=int, default=256,
                        help='校准数据序列长度')
    
    # 保存参数
    parser.add_argument('--output_dir', type=str, default='./quantized_outputs',
                        help='输出目录')
    parser.add_argument('--model_name_prefix', type=str, default='flat_quantized',
                        help='保存的模型名称前缀')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("完整的 FlatQuant 训练和保存示例")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model_name}")
    logger.info(f"设备: {device}")
    logger.info(f"量化配置: W{args.w_bits}A{args.a_bits}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    try:
        # 1. 加载模型和分词器
        logger.info("步骤 1: 加载模型和分词器")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device,
            config=config
        )
        
        # 2. 创建校准数据
        logger.info("步骤 2: 创建校准数据")
        calib_data = create_dummy_calib_data(
            tokenizer, 
            model, 
            num_samples=args.num_calib_samples,
            seq_len=args.calib_seq_len
        )
        logger.info(f"校准数据创建完成: {len(calib_data)} 样本")
        
        # 3. 创建层映射
        logger.info("步骤 3: 创建层配置")
        rules = {
            r'.*\.down_proj$': QuantType.W8A8_DYNAMIC,
            r'.*\.o_proj$': QuantType.W8A8_DYNAMIC,
        }
        # rules=None
        layer_map = create_layer_map(model, rules=rules)
        logger.info(f"层配置创建完成: {len(layer_map)} 层")

        # 4. 执行训练和保存
        logger.info("步骤 4: 开始 FlatQuant 训练")
        
        # 执行训练并保存（使用新的分片保存功能）
        trainer = flat_quant_train_and_save(
            model=model,
            calib_data=calib_data,
            layer_map=layer_map,
            args=args,
            logger=logger,
            save_dir=args.output_dir,
            model_name="flat_quantized_model",
        )
        
        logger.info("FlatQuant 训练完成!")
        
        # 5. 复制配置文件
        logger.info("步骤 5: 复制配置文件")
        copy_config_files(args.model_name, args.output_dir, logger)

        # 6. 输出保存信息
        logger.info("步骤 6: 保存完成信息")
        logger.info("保存的文件:")
        format_dir = args.output_dir
        if os.path.exists(format_dir):
            files = sorted(os.listdir(format_dir))
            logger.info(f"  {format_dir}")
            for file in files:
                file_path = os.path.join(format_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"    - {file} ({size_mb:.2f} MB)")
        
        logger.info("=" * 60)
        logger.info("示例运行完成!")
        logger.info("=" * 60)
        logger.info("使用说明:")
        logger.info("1. 量化模型已保存到指定目录")
        logger.info("2. 相关的配置文件 (tokenizer_config.json, config.json 等) 已一并复制。")
        logger.info("3. 可以使用 AutoModelForCausalLM.from_pretrained(output_dir) 加载量化模型。")
        logger.info("4. 元数据包含完整的量化配置信息")
        
    except Exception as e:
        logger.error(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def copy_config_files(source_dir, dest_dir, logger):
    """
    从原始权重文件夹复制所有 .json 和 .py 文件到目标文件夹。

    Args:
        source_dir (str): 原始权重的目录。
        dest_dir (str): 保存量化模型的目标目录。
        logger: 日志记录器。
    """
    logger.info(f"开始从 {source_dir} 复制配置文件到 {dest_dir}")
    
    if not os.path.isdir(source_dir):
        logger.error(f"源目录不存在: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    
    # 查找所有 .json 和 .py 文件
    files_to_copy = []
    for extension in ["*.json", "*.py"]:
        files_to_copy.extend(glob.glob(os.path.join(source_dir, extension)))

    if not files_to_copy:
        logger.warning(f"在 {source_dir} 中未找到 .json 或 .py 文件。")
        return

    for file_path in files_to_copy:
        try:
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.copy2(file_path, dest_path)
            logger.info(f"  已成功复制: {file_name}")
        except Exception as e:
            logger.error(f"复制文件 {file_path} 失败: {e}")


if __name__ == "__main__":
    exit(main())
