# -*- coding: utf-8 -*-
u"""
Created on 2015-11-10

@author: cheng.li
"""
import sys
import io
from setuptools import setup

PACKAGE = "VisualPortfolio"
NAME = "VisualPortfolio"
VERSION = "0.1.7"
DESCRIPTION = "VisualPortfolio " + VERSION
AUTHOR = "cheng li"
AUTHOR_EMAIL = "wegamekinglc@hotmail.com"
URL = 'https://github.com/ChinaQuants/VisualPortfolio'

if sys.version_info > (3, 0, 0):
    requirements = "requirements/py3.txt"
else:
    requirements = "requirements/py2.txt"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=['VisualPortfolio'],
    py_modules=['VisualPortfolio.__init__'],
    install_requires=io.open(requirements, encoding='utf8').read(),
)
