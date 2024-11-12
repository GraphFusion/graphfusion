from setuptools import setup, find_packages

setup(
    name="graphfusion",
    version="0.1.0",
    description="GraphFusion Neural Memory Network SDK for adaptive clinical decision support.",
    author="Kiplangat Korir",
    author_email="Korir@GraphFusion.onmicrosoft.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "transformers",
        "networkx",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
        "flask",
        "pytest"
    ],
)
