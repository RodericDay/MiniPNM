#!/usr/bin/env python

import os
import sys

import minipnm

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='minipnm',
    version=minipnm.__version__,
    description="OpenPNM, requests style",
    author='Roderic Day',
    author_email='roderic.day@gmail.com',
    url='www.pmeal.com',
    license='MIT',
)