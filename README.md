# tf-melt

TF-MELT (TensorFlow Machine Learning Toolbox) is a collection of architectures, processing, and utilities that are transferable over a range of ML applications.

A toolbox for researchers to use for machine learning applications in the TensorFlow language. The goal of this software is to enable fast start-up of machine learning tasks and to provide a reliable and flexible framework for development and deployment. The toolbox contains generalized methods for every aspect of the machine learning workflow while simultaneously providing routines that can be tailored to specific application spaces.

## Environment

First, create a new conda environment and activate:

`conda create -n tf-melt python=3.11`

`conda activate tf-melt`

Finally, install the `tfmelt` as a package through pip either through a local install from a git clone

### Local git clone

If you cloned the repo and would like to install from the local git repo, navigate to the head directory where `setup.py` is located and type:

`pip install .`

If you want to update the pip install to make sure dependencies are current:

`pip install --upgrade .`

### Directly from github

To install the `tfmelt` package directly from github simply type:

pip install git+https://github.com/NREL/tf-melt.git

### Example Notebooks

If you want to run the example notebooks, they require a couple additional packages which can all be pip installed:

1. `scikit-learn`
2. `ipykernel`
3. `matplotlib`

## Contributing

pip install black isort flake8

## Documentation

Coming soon...

## Related repo

There is a parallel repo for implementation in PyTorch under development and will be live shortly:

https://github.com/nrel/pt-melt
