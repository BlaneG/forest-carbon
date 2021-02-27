from typing import Tuple

import numpy as np

from forest_carbon import CarbonFlux


# Initialization
STEP = 0.1
MANAGED = True

# lifetimes
lifetimes = dict(
    MEAN_FOREST = 50,
    MEAN_DECAY = 20,
    MEAN_BIOENERGY = 1,
    MEAN_SHORT = 5,
    MEAN_LONG = 50,
    HARVEST_YEAR = 90,
)

HARVEST_INDEX = int(lifetimes['HARVEST_YEAR']/STEP)

# INITIAL_CARBON is the above ground carbon (as CO2) available
# at the harvest age.  INITIAL_CARBON is used to ensure a
# mass balance for the harvested carbon that is transferred into
# different pools (energy, harvest residues, long-lived products,
# etc.)
if MANAGED:
    forest_regrowth = CarbonFlux(
        lifetimes['MEAN_FOREST'], lifetimes['MEAN_FOREST'] * 0.45, 'forest regrowth',
        1, emission=False, step_size=STEP
        )
    INITIAL_CARBON = -forest_regrowth.cdf[HARVEST_INDEX]

################################
# helpers
################################
def generate_flux_data(
        mean_forest, mean_decay, mean_short, mean_long,
        decay_tc, bioenergy_tc, short_products_tc, long_products_tc
        ) -> Tuple[dict, np.array]:
    forest_regrowth = CarbonFlux(
        mean_forest, mean_forest * 0.45, 'forest regrowth',
        1, emission=False, step_size=STEP
        )
    decay = CarbonFlux(
        mean_decay, mean_decay*0.5, 'biomass decay',
        INITIAL_CARBON * float(decay_tc), step_size=STEP)
    energy = CarbonFlux(
        1, 0.5, 'energy',
        INITIAL_CARBON * float(bioenergy_tc), step_size=STEP)
    short_lived = CarbonFlux(
        mean_short, mean_short*0.75, 'short-lived products',
        INITIAL_CARBON * float(short_products_tc), step_size=STEP
        )
    long_lived = CarbonFlux(
        mean_long, mean_long*0.5, 'long-lived products',
        INITIAL_CARBON * float(long_products_tc), step_size=STEP)

    x = forest_regrowth.x
    data = {
        'forest_regrowth': forest_regrowth,
        'biomass_decay': decay,
        'energy': energy,
        'short_lived_products': short_lived,
        'long_lived_products': long_lived}
    return data, x