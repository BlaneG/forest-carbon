import numpy as np
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

    The implementation assumes a lognormal distribution."""

    def __init__(self, mean, sd, size, name, flow_fraction, range_=(0, 121), emission=True):
        """
        Parameters
        -----------------
        mean : float
            geometric mean.
        sd : float
            geometric standard deviation
        size : int
            number of samples for creating distribution
        name : str
            name of carbon flux
        flow_fraction : float
            fraction of input or output carbon flow [0, 1]
        range_ : tuple
            upper and lower bounds of plot range
        emission : bool

        """
        self.mean = mean
        self.sd = sd
        self.size = size
        self.name = name
        self.flow_fraction = flow_fraction
        self.range_ = range_
        self.emission = emission
        self.color = colors[self.name]

        rvts = np.random.lognormal(
            np.log(self.mean), np.log(self.sd), size=self.size)

        self.pdf, self.bins = np.histogram(
            rvts, bins=range_[1]-range_[0], range=self.range_, density=True)

        # self.bins = 0.5 * (bins[:-1] + bins[1:])
        bin_width = np.abs(self.bins[0] - self.bins[1])
        self.cdf = np.cumsum(self.pdf) * bin_width
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
        fig.add_trace(
            go.Bar(x=self.bins, y=self.pdf*self.flow_fraction,
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
        fig.add_trace(
            go.Scatter(x=self.bins, y=self.cdf*self.flow_fraction,
                       name=self.name, marker_color=self.color))
        fig.update_layout(yaxis_title=y_label, xaxis_title='years')

        return fig


class CarbonModel():
    def __init__(self, carbon_flux_datasets, name):
        """
        Parameters
        -----------
        carbon_flux_datasets : dict[CarbonFlux]
        """
        self.carbon_flux_datasets = carbon_flux_datasets
        self.name = name
        self._validate_forest_to_product_transfer_coeficients()
        self.net_cumulative_carbon_flux = self._get_net_cumulative_carbon_flux()
        self.net_annual_carbon_flux = self._get_net_annual_carbon_flux()

    def plot_carbon_balance(self):
        fig = None
        for carbon_flux in self.carbon_flux_datasets.values():
            if fig is None:
                fig = carbon_flux.plot_cdf()
            else:
                new_fig = carbon_flux.plot_cdf()
                fig.add_trace(new_fig.data[0])

        fig.update_layout(
            yaxis_title='Annual C emissions/removals', template='simple_white')
        fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='grey')
        self.add_net_carbon_balance_to_figure(fig)
        return fig

    def add_net_carbon_balance_to_figure(self, fig):
        net_cumulative_carbon_flux = self.net_cumulative_carbon_flux
        new_fig = go.Figure(
            go.Scatter(
                y=net_cumulative_carbon_flux,
                name='Net C flux',
                marker_color='black',
                line_dash='dash'))
        fig.add_trace(new_fig.data[0])

    def _validate_forest_to_product_transfer_coeficients(self):
        sum_of_outputs = self.carbon_flux_datasets['biomass_decay'].flow_fraction\
                         + self.carbon_flux_datasets['energy'].flow_fraction\
                         + self.carbon_flux_datasets['short_lived_products'].flow_fraction\
                         + self.carbon_flux_datasets['long_lived_products'].flow_fraction
        assert sum_of_outputs == 1, f"transfers from forest expected to sum to 1 not {sum_of_outputs}"

    def _get_net_cumulative_carbon_flux(self):
        net_flux = None
        for carbon_flux in self.carbon_flux_datasets.values():
            if net_flux is None:
                net_flux = np.copy(carbon_flux.cdf * carbon_flux.flow_fraction)
            else:
                net_flux += carbon_flux.cdf * carbon_flux.flow_fraction
        return net_flux

    def _get_net_annual_carbon_flux(self):
        net_flux = None
        for carbon_flux in self.carbon_flux_datasets.values():
            if net_flux is None:
                net_flux = np.copy(carbon_flux.pdf * carbon_flux.flow_fraction)
            else:
                net_flux += carbon_flux.pdf * carbon_flux.flow_fraction
        return net_flux
