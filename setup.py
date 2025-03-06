from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prodiphy",
    version="0.1.0",
    author="Sebastian Proost",
    author_email="sebastian.proost@gmail.com",
    description="Probabilistic models to examine differences between (sub-)populations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raeslab/prodiphy",
    project_urls={
        "Bug Tracker": "https://github.com/raeslab/prodiphy/issues",
    },
    install_requires=[
        "pymc>=5.16.2",
        "bambi>=0.14.1",
        "arviz>=0.19.0",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "tabulate>=0.9.0",
        "scipy>=1.7.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0. https://creativecommons.org/licenses/by-nc-sa/4.0/",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
