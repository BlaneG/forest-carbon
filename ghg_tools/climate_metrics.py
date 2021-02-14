"""
This module implements methods for computing climate metrics as described
by the International Governmental Panel on Climate Change."""


import numpy as np
from scipy.integrate import trapz
from scipy.signal import convolve

# W m–2 ppbv-1
RADIATIVE_EFFICIENCY_ppbv = {"co2": 1.37e-5, "ch4": 3.63e-4, "n2o": 3.00e-3}
COEFFICIENT_WEIGHTS = np.array([0.2173, 0.2240, 0.2824, 0.2763])
TIME_SCALES = np.array([394.4, 36.54, 4.304])

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
    return ppv_to_kg * RADIATIVE_EFFICIENCY_ppbv[ghg]


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


    exponential_1 = np.exp(-time_horizon/TIME_SCALES[0])
    exponential_2 = np.exp(-time_horizon/TIME_SCALES[1])
    exponential_3 = np.exp(-time_horizon/TIME_SCALES[3])

    return (
        COEFFICIENT_WEIGHTS[0]
        + COEFFICIENT_WEIGHTS[1]*exponential_1
        + COEFFICIENT_WEIGHTS[2]*exponential_2
        + COEFFICIENT_WEIGHTS[3]*exponential_3
    )


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


    exponential_1 = 1 - np.exp(-t/TIME_SCALES[0])
    exponential_2 = 1 - np.exp(-t/TIME_SCALES[1])
    exponential_3 = 1 - np.exp(-t/TIME_SCALES[2])
    cumulative_concentration = (
        COEFFICIENT_WEIGHTS[0]*t
        + COEFFICIENT_WEIGHTS[1]*TIME_SCALES[0]*exponential_1
        + COEFFICIENT_WEIGHTS[2]*TIME_SCALES[1]*exponential_2
        + COEFFICIENT_WEIGHTS[3]*TIME_SCALES[2]*exponential_3
        )

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


def dynamic_GWP(time_horizon, net_emissions, step_size=0.1, is_unit_impulse=False):
    """Computes CO2 equivalent radiative forcing for net_emissions.

    Notes
    ------------

    Global Warming Potential is defined as the cumulative radiative forcing
    of :math:`GHG_x` emitted in year = 0 over a given time-horizon 
    (:math:`t`):

    .. math:
        GWP(t) = \frac{cumulativeRadiativeForcingGHG\_x(t)}
                    {cumulativeRadiativeForcing\_CO2(t)}

    
    Dynamic GWP ([1]_, [2]_ [3]_, [4]_) is the cumulative radiative forcing due 
    to annual emissions (:math:`t'`) of :math:`GHG_x` over a give time-horizon 
    (:math:`t`) which can be expressed as:

    .. math:
        dynamicGWP_x(t, t')
                    = {\mathbf{emission_x}(t')}\cdot{\mathbf{GWP_x}(t-t')}
                    = \sum_{t'}{\mathbf{emission_x}(t'){\mathbf{GWP_x}(t-t')}}
                    = \frac{
                    \sum_{t'}{cumulativeRadiativeForcingGHG_x(t-t')}}
                    {cumulativeRadiativeForcing_{CO2}(t)}


    Parameters
    ---------------
    time_horizon : int
        The time horizon over which radiative forcing is computed.
    net_emissions : np.array
        Annual emissions/removals.
    is_unit_impulse : bool
        Specifies whether net_emissions is a scipy.signal.unit_impulse
    

    Notes
    ---------------
    net_emissions should not contain both unit_impulse and pdf emission
    distributions due to numerical integration issues.  After numerical
    integration, the output is re-normalized using the step_size when
    net_emissions is_unit_impulse=False. When is_unit_impulse is true,
    this normalization is not required.

    TODO
    -----------------
    update the method to take an arbitrary AGWP_GHG method

    References
    --------------
    .. [1] Fearnside et al. 2000.  https://link.springer.com/article/10.1023/A:1009625122628
    .. [2] Moura Costa et al. 2000.  https://link.springer.com/article/10.1023/A:1009697625521
    .. [3] Levassuer et al. 2010.  https://pubs.acs.org/doi/10.1021/es9030003
    .. [4] Cherubini et al. 2011.  https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1757-1707.2011.01102.x


    """
    # A step of 0.1 is recommended to reduce the integration error
    t = np.arange(0, time_horizon+step_size, step_size)
    # AGWP for each time step
    AGWP = AGWP_CO2(t)
    # A convolution: flip AGWP, multiple the two vectors and sum the result
    # Technically we are simplying multiplying AGWP_100 to net emissions from
    # year 0, multiplying AGWP_99 to net emissions from year 1 and so on and
    # then summing the result to compute the total radiative forcing due
    # to the net emission flux over the time horizon.
    if len(net_emissions) < len(t):
        raise ValueError(
            f"Shapes not aligned {net_emissions.shape}, {t.shape}.")
    dynamic_AGWP_t = convolve(
        net_emissions[0:len(t)+1],
        AGWP[0:len(t)+1],
        mode='valid')

    dynamic_GWP_t = dynamic_AGWP_t / AGWP_CO2(100)

    # If the input is not a unit_impulse, we have to re-normalize
    # the result by the number of steps per year. An alternative
    # approach for users to implement a emission using a uniform
    # distribution (e.g. uniform.pdf(x, loc=emission_year, scale=0.1))
    # One issue with this later approach is that the output can look
    # strange when you plot the results because the pdf will spike to 10.
    if is_unit_impulse:
        return dynamic_GWP_t[0]
    else:
        return dynamic_GWP_t[0] * step_size


def AGTP_CO2(t):
    """

    References
    ------------
    1. 8.SM.15 in https://www.ipcc.ch/site/assets/uploads/2018/07/WGI_AR5.Chap_.8_SM.pdf
    """
    radiative_efficiency = get_radiative_efficiency_kg("co2")

    # Short-term and long-term temperature response 
    # (Kelvin per (Watt per m2)) to radiative forcing
    temperature_response_coefficients = [0.631, 0.429]
    # Temporal scaling factors (years)
    temporal_weights = [8.4, 409.5]

    temperature_response = 0
    for j in range(2):
        short_term_temperature_response = COEFFICIENT_WEIGHTS[0] * temperature_response_coefficients[j]
        temporal_weight_1 = np.exp(-t/temporal_weights[j])
        weighted_short_term_temperature_response = short_term_temperature_response * (1 - temporal_weight_1)

        weighted_long_term_temperature_response = 0
        for i in range(3):
            temporal_weight_2_linear = TIME_SCALES[i] / (TIME_SCALES[i] - temporal_weights[j])
            long_term_temperature_response = COEFFICIENT_WEIGHTS[i+1] * temperature_response_coefficients[j]
            long_term_temperature_response = long_term_temperature_response * temporal_weight_2_linear
            temporal_weight_2_exponential = np.exp(-t/TIME_SCALES[i])
            weighted_long_term_temperature_response += (
                long_term_temperature_response
                * (temporal_weight_2_exponential - temporal_weight_1)
            )
        
        temperature_response += (
            weighted_short_term_temperature_response
            + weighted_long_term_temperature_response
        )
    return radiative_efficiency * temperature_response
        
            
