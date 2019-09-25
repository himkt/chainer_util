#!/usr/bin/env python

from setuptools import find_packages, setup


install_requires = []
install_requires.append("numpy==1.17.0")
install_requires.append("chainer==7.0.0b2")

setup(
    name="chainer_util",
    version="1.0",
    description="Neural Named Entity Recognizer",
    author="himkt",
    author_email="himkt@klis.tsukuba.ac.jp",
    url="https://github.com/himkt/pyner",
    packages=find_packages(),
    install_requires=install_requires,
)
