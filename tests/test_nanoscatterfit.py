#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
"""Tests for `nanoscatterfit` package."""

import pytest
import pandas as pd
import os

from nanoscatterfit  import *


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
    
def test_import_rawdata():
    # Create a sample .txt file
    sample_data = "q I\n0.1 10\n0.2 20\n0.3 30"
    sample_file = 'test_data.txt'
    with open(sample_file, 'w') as file:
        file.write(sample_data)

    # Call the function with the sample file
    result_df = import_rawdata(sample_file)

    # Check if the result is as expected
    expected_df = pd.DataFrame({'q': [0.1, 0.2, 0.3], 'I': [10, 20, 30]})
    pd.testing.assert_frame_equal(result_df, expected_df)

    # Clean up: remove the sample file after test
    os.remove(sample_file)
