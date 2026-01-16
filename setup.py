"""Setup configuration for ML Engine - Kedro 1.1.1 Compatible."""
from setuptools import setup, find_packages

# Read requirements from file
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ml-engine",
    version="0.2.1",
    author="ML Team",
    author_email="ml@example.com",
    description="World-class ML Engine built with Kedro 1.1.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youruser/ml-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "kedro>=1.1.1",
        "pandas>=2.2.0",
        "numpy>=1.26.4",
        "scikit-learn>=1.4.1",
        "xgboost>=2.0.3",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.13.0",
            "black>=24.1.1",
            "flake8>=7.0.0",
            "pylint>=3.0.3",
            "mypy>=1.8.0",
            "isort>=5.13.2",
            "ipython>=8.20.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.6",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
    include_package_data=True,
)
