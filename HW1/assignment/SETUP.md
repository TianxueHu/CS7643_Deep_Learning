---
layout:     page
title:      Homework 1, Programming Question
permalink:  /AP6c2L5Z4nNd9i4sB86r/hw1-coding-setup/
---

# Setup Instructions

Follow the instructions before you start working on the homework. These steps will be useful for later homeworks as well.

### Getting Started

In this course, we will be using python often (most assignments will need a good amount of python). We have tested the code with Python 3.7.1 and recommend using a virtual environment with Python 3.7.1 for the assignments. 

#### Anaconda

Although many distributions of python are available, we recommend that you use the [Anaconda Python](https://store.continuum.io/cshop/anaconda/). Here are the advantages of using Anaconda:

- Easy seamless install of [python packages](http://docs.continuum.io/anaconda/pkg-docs) (most come standard)
- It does not need root access to install new packages
- Supported by Linux, OS X and Windows
- Free!

We suggest that you use either Linux (preferably Ubuntu) or OS X.
Follow the instructions [here](http://docs.continuum.io/anaconda/install) to install Anaconda python.
Remember to make Anaconda python the default python on your computer.
Common issues are addressed here in the  [FAQ](http://docs.continuum.io/anaconda/faq).

#### Python
If you are comfortable with python, you can skip this section! 

If you are new to python and have sufficient programming experience in using languages like C/C++, MATLAB, etc., you should be able to grasp the basic workings of python necessary for this course easily.

We will be using the [Numpy](http://www.numpy.org/) package extensively as it is the fundamental package for scientific computing providing support for array operations, linear algebra, etc. A good tutorial to get you started is [here](http://cs231n.github.io/python-numpy-tutorial/). For those comfortable with the operations of MATLAB, [this](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html) might prove useful.

For some assignments, we will be using the [Jupyter Notebook](https://jupyter.org/).
Jupyter is a web app for interactive computing developed originally developed as part of the [IPython](https://ipython.org/) interactive shell.
The notebook is a useful environment where text can be embedded with code enabling us to set a flow while you do the assignments.

#### Downloading the starter code
Download and unzip the [starter code]({{ site.baseurl }}/assets/f20cs7643_hw1_starter.zip).

We recommend creating a virtual environment through anaconda for the class with Python=3.7.1 and all the other dependencies installed by running:
```sh
conda create -n cs7643 python=3.7.1
conda activate cs7643
```

Next, navigate to the `assignment` directory unzipped from the starter code. 

Run the following command from your newly created conda environment to install the required packages.

```
pip install -r requirements.txt
```

If you have setup your virtual environment correctly, you should be able to start the Jupyter notebook environment with:

```sh
jupyter notebook
```

assuming you have installed jupyter into your virtual environment.

Now you should be able to see the Jupyter home page when you navigate to `http://localhost:8888/` in your browser.
It shows you a listing of files in the directory you ran the `jupyter notebook` command from and allows you to create new notebooks.
Jupyter notebook files have `.ipynb` extensions.

You will train the classifiers you create on images in the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html).

CIFAR-10 is a toy dataset with 60000 images of size 32 X 32, belonging to 10 classes.
You need to implement various classifiers as a part of this homework.

This homework is based on [assignment 1](http://cs231n.github.io/assignments2017/assignment1/) of the CS231n course at Stanford.

#### Getting the dataset

Make sure you are connected to the internet. Navigate to the `cs231n/datasets` folder and run the following:

```sh
./get_datasets.sh
```

This script will download the python version of the database for you and put it in `cs231n/datasets/cifar-10-batches-py` folder.

Note : If you hit a `dyld` library related error with this script, consider switching to an older OpenSSL package as indicated [here](https://stackoverflow.com/questions/59006602/dyld-library-not-loaded-usr-local-opt-openssl-lib-libssl-1-0-0-dylib).

Now you can proceed to solve questions in HW1.