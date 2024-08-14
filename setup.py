from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="mame-gym",
    version="0.1.0",
    description="A Gymnasium-compatible environment for MAME games",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mode80/mame-gym",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "mame_gym": ["*.lua"],
    },
)
