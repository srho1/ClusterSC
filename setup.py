from setuptools import setup, find_packages


PKG_NAME = "syclib"
VERSION = "0.1"


setup(name=PKG_NAME, version=VERSION, packages=find_packages(include=f"{PKG_NAME}.*"))
