# -*- coding: utf-8 -*-
"""Setup script for Deep Learning Tutorial."""

from pathlib import Path

from setuptools import find_packages, setup


def read_readme() -> str:
    readme_path = Path(__file__).parent / "README.md"
    return readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""


setup(
    name="deep-learning-tutorial",
    version="0.1.0",
    description="纯Python实现的深度学习教学项目（Perceptron→Transformer 示例与工具）",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Deep Learning Tutorial Contributors",
    python_requires=">=3.7",
    packages=find_packages(exclude=("tests", "examples", "docs", "notebooks")),
    py_modules=[
        "main",
        "deep_learning_fundamentals",
        "deep_learning_cnn",
        "deep_learning_rnn",
        "deep_learning_advanced",
        "deep_learning_advanced_projects",
        "deep_learning_advanced_optimization",
        "deep_learning_exercises",
        "deep_learning_math_theory",
        "deep_learning_cutting_edge",
    ],
    include_package_data=True,
    data_files=[
        ("datasets", ["datasets/mnist_sample.npz", "datasets/text_sequences.txt"]),
    ],
    install_requires=[
        "numpy>=1.19.0,<2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "ruff>=0.1.0",
            "mypy>=0.990",
        ],
        "viz": ["matplotlib>=3.3.0"],
    },
    entry_points={
        "console_scripts": [
            "dl-tutorial=main:main",
        ]
    },
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
