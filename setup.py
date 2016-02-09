import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="arow",
    version="0.1",
    description=("Cost-sensitive multiclass classification with Adaptive Regularization of Weights"),
    author="Andreas Vlachos",
    #author_email = "andrewjcarter@gmail.com",
    #url = "http://packages.python.org/an_example_pypi_project",
    license="BSD",
    #keywords = "example documentation tutorial",
    #packages=['arow']#, 'tests'],
    long_description=read('README'),
    py_modules=['arow'],
)
