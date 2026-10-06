from setuptools import setup, find_packages

setup(
    name="mrixfields",
    version="0.1.0",
    description="MRIxFields2026: A Generalizable Cross-Field MRI Translation and Harmonization Challenge",
    author="MRIxFields2026 Organizing Team",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "monai",
        "einops",
        "numpy",
        "nibabel",
        "scipy",
        "scikit-image",
        "matplotlib",
        "pillow",
        "lpips",
        "pytorch-msssim",
        "PyYAML",
        "tqdm",
        "SimpleITK",
    ],
)
