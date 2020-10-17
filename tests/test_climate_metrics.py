import numpy as np

from ghg_tools.climate_metrics import AGWP_CO2, AGWP_CH4_no_CO2


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
