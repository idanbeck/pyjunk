# Basic setup script for the pyjunk repo/package

from distutils.core import setup
from setuptools import find_packages

setup(
    name='junktools',
    version='0.0.1',
    packages=find_packages(),
    license='MIT License',
)
