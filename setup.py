"""Setup configuration for ML Engine - Latest Version (Python 3.12 Compatible)."""
from setuptools import setup, find_packages

# Read requirements from file
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ml-engine",
    version="0.2.0",
    author="ML Team",
    author_email="ml@example.com",
    description="World-class ML Engine built with Kedro - Python 3.12 Compatible",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youruser/ml-engine",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "kedro>=0.19.5",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.1",
        "xgboost>=2.0.3",
        "pyyaml>=6.0.1",
        "click>=8.1.7",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "pylint>=3.0.2",
            "mypy>=1.7.1",
            "isort>=5.13.2",
            "ipython>=8.17.2",
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
