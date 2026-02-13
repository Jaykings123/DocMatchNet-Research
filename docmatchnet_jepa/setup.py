from setuptools import find_packages, setup

setup(
    name="docmatchnet-jepa",
    version="0.1.0",
    description="DocMatchNet-JEPA research framework for healthcare recommendation",
    author="Research Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.66.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.19.0",
    ],
)
