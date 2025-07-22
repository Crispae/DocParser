#!/usr/bin/env python3
"""
Setup script for DocParser
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="docparser",
    version="0.1.0",
    author="DocParser Team",
    author_email="your.email@example.com",
    description="Multi-Engine VLM-Based OCR Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/docparser",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision",
            "torchaudio",
        ],
        "full": [
            "olmocr[gpu]",
            "easyocr",
        ],
    },
    entry_points={
        "console_scripts": [
            "docparser=DocParser.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 