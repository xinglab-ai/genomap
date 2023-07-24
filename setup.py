from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="genomap",
    version="1.3.0",
    author="Md Tauhidul Islam",
    author_email="tauhid@stanford.edu",
    description="Genomap converts tabular gene expression data into spatially meaningful images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xinglab-ai/genomap",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("genomap.genoNetus")),
    python_requires=">=3.8",
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
)
