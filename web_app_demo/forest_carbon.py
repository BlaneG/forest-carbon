import numpy as np
from scipy.stats import gamma
import plotly.graph_objects as go


browns = ['#543005', '#8c510a', '#bf812d', '#dfc27d']

colors = {
    'forest regrowth': '#1b7837',  # green
    'biomass decay': browns[0],  # browns
    'energy': browns[1],
    'short-lived products': browns[2],
    'long-lived products': browns[3]
                                }


class CarbonFlux():
    """An object to hold carbon flux data.

    The implementation assumes a gamma distribution."""

    def __init__(
            self,
            mean,
            sd,
            name,
            initial_carbon,
            flow_fraction,
            range_=(0, 120),
            emission=True,
            step_size=0.1,
            ):
        """
        Parameters
        -----------------
        mean : float
            Geometric mean.
        sd : float
            Geometric standard deviation
        name : str
            Name of carbon flux
        initial_carbon : float
            Initial carbon used to validate mass balance when combined with `flow_fraction`.
        flow_fraction : float
            Fraction of input or output carbon flow [0, 1]
        range_ : tuple
            Upper and lower bounds of x-axis
        emission : bool
            Whether carbon flux is an emission or an removal
        step_size : float
            Increment size for x-axis
        """
        self.mean = mean
        self.sd = sd
        self.name = name
        self.initial_carbon = initial_carbon
        self.flow_fraction = flow_fraction
        self.initial_carbon_stock = initial_carbon * flow_fraction
        self.range_ = range_
        self.emission = emission
        self.color = colors[self.name]
        self.step_size = step_size


        # a step size of 1 will cause a pdf mass balance error when
        # mean is <=2.
        self.x = np.arange(self.range_[0], self.range_[1]+step_size, step=step_size)
        a = self.mean**2/self.sd**2
        scale = self.mean/(self.mean**2/self.sd**2)
        self.pdf = gamma.pdf(x=self.x, a=a, scale=scale, loc=0)
        self.cdf = gamma.cdf(x=self.x, a=a, scale=scale, loc=0)

        # emissions are positive, removals are negative
        if not self.emission:
            self.pdf = -self.pdf
            self.cdf = -self.cdf

    def plot_pdf(self):
        """
        Parameters
        ----------------
        distribution : tuple
            Contains the bins, pdf, cdf and distribution name for plotting.
        """
        if self.emission:
            y_label = f'Annual C emission from {self.name}'
        else:
            y_label = f'Annual C removal from {self.name}'
        fig = go.Figure()
        annual_carbon_flux = self.pdf * self.initial_carbon_stock
        fig.add_trace(
            go.Bar(x=self.x, y=annual_carbon_flux,
                   name=self.name, marker_color=self.color))
        fig.update_layout(
            barmode='relative', yaxis_title=y_label, xaxis_title='years')
        fig.update_layout(
            yaxis_title=y_label, xaxis_title='years', template='simple_white')
        return fig

    def plot_cdf(self):
        if self.emission:
            y_label = f'Cumulative C emission from {self.name}'
        else:
            y_label = f'Cumulative C removal from {self.name}'

        fig = go.Figure()
        cumulative_carbon_flux = self.cdf * self.initial_carbon_stock
        fig.add_trace(
            go.Scatter(x=self.x, y=cumulative_carbon_flux,
                       name=self.name, marker_color=self.color))
        fig.update_layout(yaxis_title=y_label, xaxis_title='years')

        return fig


class CarbonModel():
    def __init__(self, carbon_flux_datasets, x, name, initial_carbon):
        """
        Parameters
        -----------
        carbon_flux_datasets : dict[CarbonFlux]
        x : np.array
            time horizon for model
        name: str
        """
        self.x = x
        self.name = name
        self.initial_carbon = initial_carbon
        self.carbon_flux_datasets = self._normalize_to_merchantable(carbon_flux_datasets)
        self._validate_forest_to_product_transfer_coefficients()
        self.net_cumulative_carbon_flux = self._get_net_cumulative_carbon_flux()
        self.net_annual_carbon_flux = self._get_net_annual_carbon_flux()

    def _normalize_to_merchantable(self, datasets):
        """Scaling all of the flows to 1 tonne merchantable CO2"""
        MERCH_TC = 1-datasets['biomass_decay'].initial_carbon_stock/self.initial_carbon
        for key in datasets.keys():
            if key == 'forest_regrowth':
                datasets[key].initial_carbon_stock = datasets[key].initial_carbon_stock * 1/MERCH_TC/self.initial_carbon
                # datasets[key].cdf *= 1/MERCH_TC/self.initial_carbon
            else:
                datasets[key].initial_carbon_stock = datasets[key].initial_carbon_stock * 1/MERCH_TC/self.initial_carbon
                # datasets[key].cdf *= 1/MERCH_TC/self.initial_carbon
        return datasets

    def plot_carbon_balance(self):
        """Cumulative annual carbon emissions."""
        fig = None
        for carbon_flux in self.carbon_flux_datasets.values():
            if fig is None:
                fig = carbon_flux.plot_cdf()
            else:
                new_fig = carbon_flux.plot_cdf()
                fig.add_trace(new_fig.data[0])

        fig.update_layout(
            yaxis_title='Annual CO2 emissions/removals', template='simple_white')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='grey')
        self.add_net_carbon_balance_to_figure(fig)
        return fig

    def add_net_carbon_balance_to_figure(self, fig):
        new_fig = go.Figure(
            go.Scatter(
                x=self.x,
                y=self.net_cumulative_carbon_flux,
                name='Net CO2 flux',
                marker_color='black',
                line_dash='dash',
                fill='tozeroy',
                fillcolor='rgba(213,163,163,0.15)',
                ))
        fig.add_trace(new_fig.data[0])

    def _validate_forest_to_product_transfer_coefficients(self):
        emissions = 0
        removals = self.initial_carbon * self.carbon_flux_datasets['forest_regrowth'].initial_carbon_stock
        print(removals)
        for key in self.carbon_flux_datasets.keys():
            if key != 'forest_regrowth':
                emissions += self.carbon_flux_datasets[key].initial_carbon_stock
        assert np.isclose(emissions/removals, 1), \
            f"transfers from forest expected to sum to {removals}\
                not {emissions}"


    def _get_net_carbon_flux(self, attr):
        net_flux = None
        for carbon_flux in self.carbon_flux_datasets.values():
            flux = np.copy(getattr(carbon_flux, attr))
            if net_flux is None:
                net_flux = flux * carbon_flux.initial_carbon_stock
            else:
                net_flux += flux * carbon_flux.initial_carbon_stock
        return net_flux

    def _get_net_cumulative_carbon_flux(self):
        net_flux = self._get_net_carbon_flux('cdf')
        return net_flux

    def _get_net_annual_carbon_flux(self):
        net_flux = self._get_net_carbon_flux('pdf')
        return net_flux
