#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# 读取 README.md 作为长描述
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "独立的 FlatQuant 量化训练模块"

# 读取版本信息
def get_version():
    """从版本文件或其他方式获取版本号"""
    return "1.0.0"

# 必需依赖
install_requires = [
    "torch>=1.9.0",
    "numpy>=1.19.0", 
    "scipy>=1.7.0",
    "tqdm>=4.60.0",
    "transformers>=4.20.0",
]

# 可选依赖
extras_require = {
    "safetensors": ["safetensors>=0.3.0"],
    "dev": [
        "pytest>=6.0.0",
        "pytest-cov>=2.12.0",
        "black>=21.0.0",
        "flake8>=3.9.0",
    ],
    "all": ["safetensors>=0.3.0"],
}

setup(
    name="nanomind",
    version=get_version(),
    author="Huawei Technologies",
    author_email="",
    description="独立的 FlatQuant 量化训练模块",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/nanomind",  # 请替换为实际的仓库地址
    
    # 包配置
    packages=find_packages(),
    include_package_data=True,
    
    # Python 版本要求
    python_requires=">=3.7",
    
    # 依赖关系
    install_requires=install_requires,
    extras_require=extras_require,
    
    # 分类器
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # 关键词
    keywords="quantization, neural networks, FlatQuant, model compression, deep learning",
    
    # 项目链接
    project_urls={
        "Bug Reports": "https://github.com/your-org/nanomind/issues",
        "Source": "https://github.com/your-org/nanomind",
        "Documentation": "https://github.com/your-org/nanomind/blob/main/README.md",
    },
    
    # 命令行工具（如果有的话）
    entry_points={
        "console_scripts": [
            # 如果需要添加命令行工具，可以在这里定义
            # "nanomind-train=nanomind.cli:main",
        ],
    },
    
    # 数据文件
    package_data={
        "nanomind": [
            "*.md",
            "*.txt",
            "**/*.md",
            "**/*.txt",
        ],
    },
    
    # 排除的包
    exclude_package_data={
        "": ["*.pyc", "*.pyo", "*~", "*.tmp"],
    },
    
    # zip_safe 设置
    zip_safe=False,
)