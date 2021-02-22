import os
from setuptools import setup, find_packages

scripts = [
    'bin/run-pizza-cutter-sims',
]

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "pizza_cutter_sims",
    "_version.py")
with open(pth, 'r') as fp:
    exec(fp.read())

setup(
    name='pizza-cutter-sims',
    version=__version__,
    description="sims for testing the pizza cutter",
    author="MRB",
    packages=find_packages(),
    include_package_data=True,
    scripts=scripts,
)
