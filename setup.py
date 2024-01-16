
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="partitionrank",
    version="0.0.1",
    author="A P",
    description="PartitionRank",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT Software License",
        "Operating System :: OS Independent",
    ],
)