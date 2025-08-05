from nanomind.vllm_adapter.quant_config import NanomindQuantConfig
from vllm import LLM, SamplingParams
import os
os.environ["VLLM_USE_V1"] = '0'

# 1. 初始化模型（默认使用单卡）
llm = LLM(model="quant_model",
    gpu_memory_utilization=0.8,
    max_model_len=1024,
    quantization="nanomind",)

# 2. 准备采样参数（使用默认配置）
sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

# 3. 定义输入
prompts = [
    "人工智能的未来发展将会",
    "如何学习Python编程？",
    "请解释相对论的基本概念："
]

# 4. 生成输出
outputs = llm.generate(prompts, sampling_params)

# 5. 打印结果
for i, output in enumerate(outputs):
    print(f"输入 {i+1}: {prompts[i]}")
    print(f"输出 {i+1}: {output.outputs[0].text}\n")