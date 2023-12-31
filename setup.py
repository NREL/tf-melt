from setuptools import setup

setup(
    name="tfmelt",
    version="0.1.1",
    description="TensorFlow Machine Learning Toolbox (TF-MELT)",
    url="https://github.com/NREL/tf-melt",
    author="Nicholas T. Wimer",
    author_email="nwimer@nrel.gov",
    license="BSD 3-Clause License",
    packages=["tfmelt"],
    install_requires=["tensorflow"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
    ],
)
