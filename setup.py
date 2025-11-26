from setuptools import setup, find_packages

setup(
    name="conformal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "tqdm",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for conformal prediction.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/conformal",
)
