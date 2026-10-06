from setuptools import setup, find_packages

setup(
    name="mrix2026-cli",
    version="0.2.0",
    description="CLI tools for MRIxFields2026 — build and upload Synapse submission packages",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "nibabel"],
    entry_points={"console_scripts": ["mrix2026 = mrix2026_cli.cli:main"]},
    python_requires=">=3.10",
)
