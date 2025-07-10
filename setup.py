"""
Setup script for pyleebounds package.
"""

from setuptools import setup, find_packages
import os

def read_readme():
    """Read README.md file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Python package for Lee 2009 treatment effect bounds under sample selection"

def read_requirements():
    """Read requirements.txt file."""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Default requirements if file not found
        return [
            "numpy>=1.20.0",
            "pandas>=1.3.0", 
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "scipy>=1.7.0"
        ]

setup(
    name="pyleebounds",
    version="0.1.0",
    author="Vasco Yasenov",
    author_email="",
    description="Python package for Lee 2009 treatment effect bounds under sample selection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vyasenov/pyleebounds",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 