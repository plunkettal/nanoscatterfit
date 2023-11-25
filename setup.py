#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===============================
HtmlTestRunner
===============================


.. image:: https://img.shields.io/pypi/v/nanoscatterfit.svg
        :target: https://pypi.python.org/pypi/nanoscatterfit
.. image:: https://img.shields.io/travis/plunkettal/nanoscatterfit.svg
        :target: https://travis-ci.org/plunkettal/nanoscatterfit

A simple package for analyzing nanoparticle dispersions and superstructures


Links:
---------
* `Github <https://github.com/plunkettal/nanoscatterfit>`_
"""

from setuptools import setup, find_packages

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Alexander Plunkett",
    author_email='plunkett-a@hotmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="A simple package for analyzing nanoparticle dispersions and superstructures",
    install_requires=requirements,
    license="MIT license",
    long_description=__doc__,
    include_package_data=True,
    keywords='nanoscatterfit',
    name='nanoscatterfit',
    packages=find_packages(include=['nanoscatterfit']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/plunkettal/nanoscatterfit',
    version='0.1.1',
    zip_safe=False,
)
