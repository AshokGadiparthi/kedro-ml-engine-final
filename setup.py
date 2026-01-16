from setuptools import setup, find_packages

setup(
    name="ml-engine",
    version="0.1.0",
    author="ML Team",
    description="World-class ML Engine built with Kedro",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "kedro>=0.18.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.7.0", "flake8>=6.0.0", "pylint>=2.17.0"]
    },
)
