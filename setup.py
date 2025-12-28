from setuptools import setup, find_packages

setup(
    name="sarl",
    version="0.1.0",
    description="Structure-Agnostic Representation Learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "full": ["torchvision>=0.11.0", "matplotlib>=3.4.0", "scikit-learn>=0.24.0"],
        "dev": ["pytest>=6.0.0"],
    },
)
