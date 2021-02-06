import numpy as np

from ghg_tools.climate_metrics import (
    AGWP_CO2,
    AGWP_CH4_no_CO2,
    dynamic_GWP,
)


def test_AGWP_CO2():
    """
    References
    ----------
    IPCC, 2013. AR5, WG1, Chapter 8.  Appendix 8.A.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    assert np.isclose(AGWP_CO2(100)/9.17e-14, 1, atol=1e-02)
    assert np.isclose(AGWP_CO2(20)/2.49e-14, 1, atol=1e-02)


def test_AGWP_CH4_no_CO2():
    """
    References
    ----------
    IPCC, 2013. AR5, WG1, Chapter 8.  Appendix 8.A.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    assert np.isclose(AGWP_CH4_no_CO2(20)/2.09e-12, 1, atol=1e-03)
    assert np.isclose(AGWP_CH4_no_CO2(100)/2.61e-12, 1, atol=1e-03)


def compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step):
    """
    Dynamic GWP is just the GWP(time-horizon - emission_year)
    so we can validate the implementation with an alternative
    calculation."""
    total_steps = int(time_horizon/time_step) + 1

    emission_index = int(emission_year/time_step)
    emission_pulse = np.zeros(total_steps)
    emission_pulse[emission_index] = emission
    
    expected_GWP = AGWP_CO2(time_horizon - emission_year)/AGWP_CO2(time_horizon)
    return expected_GWP, emission_pulse


def test_dynamic_GWP():
    # initalize parameters
    time_horizon = 100
    time_step = 0.1
    emission = 1
    
    # First test.
    emission_year = 0
    expected_GWP_0, emission_pulse = compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step)
    actual_GWP_0 = dynamic_GWP(time_horizon, emission_pulse, time_step)
    result_0_isclose = np.isclose(expected_GWP_0, actual_GWP_0, rtol=1e-4)
    assert(result_0_isclose)


    # Second test
    emission_year = 25
    expected_GWP_25, emission_pulse = compute_expected_dynamic_GWP(
        emission, emission_year, time_horizon, time_step)
    actual_GWP_25 = dynamic_GWP(time_horizon, emission_pulse, time_step)
    result_25_isclose = np.isclose(expected_GWP_25, actual_GWP_25, rtol=1e-4)
    assert(result_25_isclose)
