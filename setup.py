from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="test-vasudha-genomap",
    version="0.0.15",
    author="Vasudha Jha",
    author_email="reachvasudha27@gmail.com",
    description="Genomap converts tabular gene expression data into spatially meaningful images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xinglab-ai/genomap",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=("genomap.genoNet", "genomap.genoNetus")),
    python_requires=">=3.7",
    install_requires=open("requirements.txt").readlines(),
    include_package_data=True,
)
