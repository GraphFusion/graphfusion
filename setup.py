from setuptools import setup, find_packages
import os

# Load the README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="graphfusion",
    version="0.1.1",
    description="GraphFusion Neural Memory Network SDK for adaptive knowledge management.",
    long_description=long_description,  # Using README.md content as long description
    long_description_content_type="text/markdown",  # Indicate markdown content
    author="Kiplangat Korir",
    author_email="Korir@GraphFusion.onmicrosoft.com",
    url="https://github.com/yourusername/graphfusion",  # Replace with the actual URL
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
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
