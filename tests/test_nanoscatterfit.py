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


def test_auto_baseline_optimizer_empty_input():
    y = pd.Series([])
    baseline = auto_baseline_optimizer(y)
    assert baseline is None


def test_auto_baseline_optimizer_single_value_input():
    y = pd.Series([1])
    baseline = auto_baseline_optimizer(y)
    assert baseline is None


def test_auto_baseline_optimizer_monotonic_input():
    y = pd.Series([1, 2, 3, 2, 1])
    baseline = auto_baseline_optimizer(y)
    assert np.allclose(baseline, [1, 1, 1, 2, 2])


def test_auto_baseline_optimizer_non_monotonic_input():
    y = pd.Series([3, 2, 1, 4, 3])
    baseline = auto_baseline_optimizer(y)
    assert np.allclose(baseline, [3, 2, 1, 3, 3])


def test_cost_func_one_peak():
    y = pd.Series([1, 2, 3, 2, 1])
    baseline = np.array([1, 1, 1, 2, 2])
    cost = cost_func(y, baseline)
    assert cost == 1


def test_cost_func_two_peaks():
    y = pd.Series([3, 2, 1, 4, 3])
    baseline = np.array([3, 2, 1, 3, 3])
    cost = cost_func(y, baseline)
    assert cost == 2


def test_filter_monotonic_peaks_one_peak():
    y = pd.Series([1, 2, 3, 2, 1])
    peaks = np.array([1, 3, 4, 5, 6])
    filtered_peaks = filter_monotonic_peaks(y, peaks)
    assert np.allclose(filtered_peaks, [1, 3, 4, 5])


def test_filter_monotonic_peaks_two_peaks():
    y = pd.Series([3, 2, 1, 4, 3])
    peaks = np.array([1, 3, 4, 5, 6])
    filtered_peaks = filter_monotonic_peaks(y, peaks)
    assert np.allclose(filtered_peaks, [1, 3, 4, 5])


def test_fit_structurefactor_fcc():
    x = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    y = pd.Series([1, 2, 3, 4, 5])
    result = fit_structurefactor(x, y, 'fcc')
    assert result.success
    assert result.params['cen1'].value == 0.3
    assert result.params['sigma1'].value == 0.03


def test_fit_structurefactor_unknown_structure():
    x = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    y = pd.Series([1, 2, 3, 4, 5])
    result = fit_structurefactor(x, y, 'unknown')
    assert not result.success


def test_fit_formfactor_lognormal_spherical():
    q_data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    I_data = pd.Series([1, 2, 3, 4, 5])
    result = fit_formfactor(q_data, I_data, distribution='lognormal', shape='spherical')
    assert result.success
    assert result.params['mu'].value == 1.3
    assert result.params['sigma'].value == 0.1


# def test_fit_formfactor_normal_cubic():
#     q_data = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
#     I_data = pd.Series([1, 2, 3, 4, 5])
#     result = fit_formfactor(q_data, I_data, distribution='
# %%
