import platform

from setuptools import find_packages, setup

install_requires = ["pytest", "scikit-learn", "matplotlib"]
print(f"Platform: {platform.system()}")

if platform.system() == "Linux":
    print(f"Installing TensorFlow with GPU support for {platform.system()}")
    # Install TensorFlow with GPU support
    install_requires.append("tensorflow[and-cuda]<2.16")
    install_requires.append("tensorflow-probability<0.24")
elif platform.system() == "Darwin":
    # Install TensorFlow for Mac
    install_requires.append("tensorflow<2.16")
    install_requires.append("tensorflow-probability<0.24")

setup(
    name="tfmelt",
    version="0.4.3",
    description="TensorFlow Machine Learning Toolbox (TF-MELT)",
    url="https://github.com/NREL/tf-melt",
    author="Nicholas T. Wimer",
    author_email="nwimer@nrel.gov",
    license="BSD 3-Clause License",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8, <3.12",
)
