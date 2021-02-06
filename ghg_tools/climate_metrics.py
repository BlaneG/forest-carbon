"""
This module implements methods for computing climate metrics as described
by the International Governmental Panel on Climate Change."""


import numpy as np
from scipy.integrate import trapz
from scipy.signal import convolve

# W m–2 ppbv-1
radiative_efficiency_ppbv = {"co2": 1.37e-5, "ch4": 3.63e-4, "n2o": 3.00e-3}


def ppbv_to_kg_conversion(ghg):
    """
    Convert the radiative efficiency from ppbv normalization to kg normalization.

    References
    --------------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. p. 8SM-15.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    # kg per kmol
    molecular_weight = {"co2": 44.01, "ch4": 16.04, "n2o": 44.013}

    total_mass_atmosphere = 5.1352e18  # kg
    mean_molecular_weight_air = 28.97  # kg per kmol
    molecular_weight_ghg = molecular_weight[ghg]
    mass_ratio = mean_molecular_weight_air/molecular_weight_ghg
    return mass_ratio * (1e9/total_mass_atmosphere)


def get_radiative_efficiency_kg(ghg):
    """Get the radiative efficiency of a GHG in W m–2 kg–1.
    """
    ppv_to_kg = ppbv_to_kg_conversion(ghg)
    return ppv_to_kg * radiative_efficiency_ppbv[ghg]


def CO2_irf(time_horizon):
    """The impulse response function of CO2.

    Parameters
    -----------
    time_horizon : int
        The time since the original CO2 emission occurred.

    References
    --------------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. Equation 8.SM.10
    https://www.ipcc.ch/report/ar5/wg1/
    """
    a0 = 0.2173
    a1 = 0.2240
    a2 = 0.2824
    a3 = 0.2763

    t_1 = 394.4
    t_2 = 36.54
    t_3 = 4.304

    exponential_1 = np.exp(-time_horizon/t_1)
    exponential_2 = np.exp(-time_horizon/t_2)
    exponential_3 = np.exp(-time_horizon/t_3)

    return a0 + a1*exponential_1 + a2*exponential_2 + a3*exponential_3


def impulse_response_function(time_horizon, ghg):
    """The impulse response function for non-CO2/CH4 GHGs.

    References
    -----------
    IPCC 2013. AR5, WG1, Chapter 8 Supplementary Material. Equation 8.SM.8.
    https://www.ipcc.ch/report/ar5/wg1/
    """
    # time_horizon = np.arange(time_horizon)
    life_time = {"ch4": 12.4, "n2o": 121}
    if ghg.lower() == "co2":
        return CO2_irf(time_horizon)
    else:

        return np.exp(-time_horizon/life_time[ghg.lower()])


def AGWP_CO2(t):
    radiative_efficiency = get_radiative_efficiency_kg("co2")

    a0 = 0.2173
    a1 = 0.2240
    a2 = 0.2824
    a3 = 0.2763

    t1 = 394.4
    t2 = 36.54
    t3 = 4.304

    exponential_1 = 1 - np.exp(-t/t1)
    exponential_2 = 1 - np.exp(-t/t2)
    exponential_3 = 1 - np.exp(-t/t3)
    cumulative_concentration = a0*t + a1*t1*exponential_1 + a2*t2*exponential_2 + a3*t3*exponential_3
    return radiative_efficiency * cumulative_concentration


def AGWP_CH4_no_CO2(t):
    """
    Note
    ------
    Does not include indirect effects from CO2 as a result of CH4 conversion to CO2.
    """
    indirect_O3 = 0.5
    indirect_H2O = 0.15
    life_time = 12.4
    radiative_efficiency = get_radiative_efficiency_kg("ch4") \
        * (1+indirect_O3+indirect_H2O)

    return radiative_efficiency * life_time * (1 - np.exp(-t/life_time))


def dynamic_GWP(time_horizon, net_emissions, step_size=0.1):
    """Computes GWP for a vector of net_emissions over a time_horizon.

    Parameters
    ---------------
    time_horizon : int
        The time horizon over which radiative forcing is computed.
    net_emissions : np.array
        Annual emissions/removals.

    TODO
    -----------------
    update the method to take an arbitrary AGWP_GHG method

    References
    --------------
    Levassuer et al. 2010.  https://pubs.acs.org/doi/10.1021/es9030003
    Cherubini et al. 2011.  https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1757-1707.2011.01102.x


    """
    # A step of 0.1 is recommended to reduce the integration error
    t = np.arange(0, time_horizon+1, step_size)
    # AGWP for each time step
    AGWP = AGWP_CO2(t)
    # A convolution: flip AGWP, multiple the two vectors and sum the result
    # Technically we are simplying multiplying AGWP_100 to net emissions from
    # year 0, multiplying AGWP_99 to net emissions from year 1 and so on and
    # then summing the result to compute the total radiative forcing due
    # to the net emission flux over the time horizon.
    dynamic_AGWP_t = convolve(
        net_emissions[0:len(t)],
        AGWP[0:len(t)],
        mode='valid')

    dynamic_GWP_t = dynamic_AGWP_t / AGWP_CO2(100)
    return dynamic_GWP_t
