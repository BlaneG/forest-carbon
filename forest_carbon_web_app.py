import json

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from forest_carbon import CarbonFlux, CarbonModel
from climate_metrics import AGWP_CO2

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def make_slider_makers(min, max, step_size):
    return {x: f'{x} years' for x in range(min, max+1, step_size)}


distribution_selections = html.Div(
            id='left-column',
            className="four columns",
            children=[
                html.H6("Explore how re-growth rates and product \
                        lifetimes can affect carbon emissions."),
                'Mean age of managed forest (e.g. changing growth rate by \
                    fertilization, brush clearing and improved re-planting):',
                dcc.Slider(
                    id='regrowth-slider',
                    min=55,
                    max=65,
                    step=1,
                    value=60,
                    marks=make_slider_makers(55, 65, 5)
                ),
                html.Div(id='regrowth-selection'),
                html.Br(),
                'Biomass decay half-life:',
                dcc.Slider(
                    id='biomass-decay',
                    min=15,
                    max=25,
                    step=1,
                    value=20,
                    marks=make_slider_makers(15, 25, 2)
                ),
                html.Div(id='decay-selection'),
                html.Br(),
                'short-lived product half-life: (e.g. improving \
                    waste paper and packaging recycling rates)',
                dcc.Slider(
                    id='short-lived',
                    min=0,
                    max=10,
                    step=1,
                    value=5,
                    marks=make_slider_makers(0, 10, 2),
                ),
                html.Div(id='short-selection'),
                html.Br(),
                'long-lived product half-life: (e.g. increasing \
                    the durability of wood products, and \
                        their reuse in second generation products',
                dcc.Slider(
                    id='long-lived',
                    min=30,
                    max=70,
                    step=1,
                    value=50,
                    marks=make_slider_makers(30, 70, 10)
                ),
                html.Div(id='long-selection'),
                html.Br(),
                ])

GWP_calculation = html.Div(
    id='dynamic-GWP-result',
    style={'text-align': 'center', 'font-weight': 'bold'},
    )

GWP_explanation = html.Div(
    style={'text-align': 'center'},
    children=[
        "text",
        html.A("Link", href='www.ipcc.ch/report/ar5/wg1/'),
        "text"]
)

carbon_balance_figure = html.Div(
            id='right-column',
            className="eight columns",
            children=[
                html.H5("Cumulative carbon emissions and removals."),
                html.P(
                    "By changing the area under the 'net C flux' curve, the \
                    climate effect is altered by increasing or decreasing \
                        the amount of carbon in the atmosphere."),
                GWP_calculation,
                dcc.Graph(id='carbon-balance-figure'),
                GWP_explanation
                    ])

app.layout = html.Div([
    html.H1("Above ground forest carbon dynamics from harvesting."),
    html.Div([
        distribution_selections,
        carbon_balance_figure
        ], className='row'),
    html.Div(id='annual-carbon-flux', style={'display': 'none'}),

])


@app.callback(
    Output(component_id='regrowth-selection', component_property='children'),
    [Input(component_id='regrowth-slider', component_property='value')]
)
def update_regrowth_selection(input_value):
    return 'Output: {}'.format(input_value)


@app.callback(
    Output(component_id='decay-selection', component_property='children'),
    [Input(component_id='biomass-decay', component_property='value')]
)
def update_decay_selection(input_value):
    return 'Output: {}'.format(input_value)


@app.callback(
    Output(component_id='short-selection', component_property='children'),
    [Input(component_id='short-lived', component_property='value')]
)
def update_shortlived_selection(input_value):
    return 'Output: {}'.format(input_value)


@app.callback(
    Output(component_id='long-selection', component_property='children'),
    [Input(component_id='long-lived', component_property='value')]
)
def update_longlived_selection(input_value):
    return 'Output: {}'.format(input_value)


@app.callback(
    [
        Output(component_id='carbon-balance-figure', component_property='figure'),
        Output(component_id='annual-carbon-flux', component_property='children')
    ],
    [
        Input(component_id='regrowth-slider', component_property='value'),
        Input(component_id='biomass-decay', component_property='value'),
        Input(component_id='short-lived', component_property='value'),
        Input(component_id='long-lived', component_property='value'),
    ]
)
def update_figure(mean_forest, mean_decay, mean_short, mean_long):
    forest_regrowth = CarbonFlux(
        mean_forest, 1.7, 1000, 'forest regrowth', 1, emission=False
        )
    decay = CarbonFlux(mean_decay, 2, 1000, 'biomass decay', 0.5)
    energy = CarbonFlux(1, 1.05, 1000, 'energy', 0.5*0.1)
    short_lived = CarbonFlux(
        mean_short, 1.5, 1000, 'short-lived products', 0.5 * 0.4
        )
    long_lived = CarbonFlux(mean_long, 1.5, 1000, 'long-lived products', 0.5 * 0.5)

    data = {
        'forest_regrowth': forest_regrowth,
        'biomass_decay': decay,
        'energy': energy,
        'short_lived_products': short_lived,
        'long_lived_products': long_lived}

    carbon_model = CarbonModel(data, 'harvest')
    fig = carbon_model.plot_carbon_balance()
    net_annual_carbon_flux = carbon_model.net_annual_carbon_flux
    return fig, json.dumps(net_annual_carbon_flux.tolist())


@app.callback(
    Output(component_id='dynamic-GWP-result', component_property='children'),
    [
        Input(component_id='annual-carbon-flux', component_property='children'),
    ]
)
def update_GWP(net_annual_carbon_flux):
    
    net_annual_carbon_flux = json.loads(net_annual_carbon_flux)
    # AGWP is the cumulative radiative forcing at time t after the emission
    AGWP = AGWP_CO2(np.arange(0, 101))
    AGWP_from_0_to_100 = np.flip(AGWP[0:101]) * net_annual_carbon_flux[0:101]
    dynamic_AGWP_100 = np.sum(AGWP_from_0_to_100)
    dynamic_GWP_100 = dynamic_AGWP_100 / AGWP_CO2(100)
    return "GWP 100 for net carbon flux: {:.2f} kg CO2 eq".format(dynamic_GWP_100)



if __name__ == '__main__':
    app.run_server(debug=True)
