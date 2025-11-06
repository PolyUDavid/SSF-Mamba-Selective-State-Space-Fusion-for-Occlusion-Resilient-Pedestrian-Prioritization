"""
P-SAFE: Multi-Modal AI Framework for Pedestrian-Centric Traffic Signal Control

Setup script for package installation.

Author: David KO
Date: November 2025
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="psafe",
    version="1.0.0",
    author="David KO",
    author_email="",
    description="Multi-Modal AI Framework for Pedestrian-Centric Traffic Signal Control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PolyUDavid/SSF-Mamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "psafe=psafe.cli:main",
        ],
    },
    keywords="traffic signal control, pedestrian safety, multi-modal fusion, computer vision, deep learning",
    project_urls={
        "Bug Reports": "https://github.com/PolyUDavid/SSF-Mamba/issues",
        "Source": "https://github.com/PolyUDavid/SSF-Mamba",
    },
)

