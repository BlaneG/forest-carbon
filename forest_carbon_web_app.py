import json
from typing import Tuple

import numpy as np
from scipy.integrate import trapz
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from forest_carbon import CarbonFlux, CarbonModel
from climate_metrics import AGWP_CO2, dynamic_GWP

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Initialization
STEP = 0.1
MANAGED=True

# lifetimes
MEAN_FOREST = 50
MEAN_DECAY = 20
MEAN_BIOENERGY = 1
MEAN_SHORT = 5
MEAN_LONG = 50

# Initial Carbon
if MANAGED:
    # for managed forests the 
    HARVEST_YEAR = 90
    HARVEST_INDEX = int(HARVEST_YEAR/STEP)
    forest_regrowth = CarbonFlux(
        MEAN_FOREST, MEAN_FOREST * 0.45, 1000, 'forest regrowth',
        1, emission=False, step_size=STEP
        )
    INITIAL_CARBON = -forest_regrowth.cdf[HARVEST_INDEX]

# biomass transfer coefficients
DECAY_TC = 0.5
LL_PRODUCTS_TC = (1-DECAY_TC) * 0.5
SL_PRODUCTS_TC = (1-DECAY_TC) * 0.4
BIOENERGY_TC = (1-DECAY_TC) * 0.1


################################
# helpers
################################
def generate_flux_data(
        mean_forest, mean_decay, mean_short, mean_long,
        decay_tc, bioenergy_tc, short_products_tc, long_products_tc
        ) -> Tuple[dict, np.array]:
    forest_regrowth = CarbonFlux(
        mean_forest, mean_forest * 0.45, 1000, 'forest regrowth',
        1, emission=False, step_size=STEP
        )
    decay = CarbonFlux(
        mean_decay, mean_decay*0.5, 1000, 'biomass decay',
        INITIAL_CARBON * float(decay_tc), step_size=STEP)
    energy = CarbonFlux(
        1, 0.5, 1000, 'energy',
        INITIAL_CARBON * float(bioenergy_tc), step_size=STEP)
    short_lived = CarbonFlux(
        mean_short, mean_short*0.75, 1000, 'short-lived products',
        INITIAL_CARBON * float(short_products_tc), step_size=STEP
        )
    long_lived = CarbonFlux(
        mean_long, mean_long*0.5, 1000, 'long-lived products',
        INITIAL_CARBON * float(long_products_tc), step_size=STEP)

    x = forest_regrowth.x
    data = {
        'forest_regrowth': forest_regrowth,
        'biomass_decay': decay,
        'energy': energy,
        'short_lived_products': short_lived,
        'long_lived_products': long_lived}
    return data, x


####################################
# Carbon balance figure
#####################################
GWP_calculation = html.Div(
    id='dynamic-GWP-result',
    style={'text-align': 'center', 'font-weight': 'bold'},
    )

GWP_explanation = html.Div(
    style={'text-align': 'center'},
    children=[
        "See IPCC, WG1, Chapter 8 for more information on ",
        html.A("GWP", href='http://www.ipcc.ch/report/ar5/wg1/', target='_blank'),
        "."]
)

carbon_balance_figure = html.Div(
            id='right-column',
            className="eight columns",
            children=[
                html.H3("Cumulative carbon emissions and removals."),
                html.P(
                    "By changing the area under the 'net C flux', the \
                    climate response (measured in kg CO2 equivalent using \
                    GWP) of using forest biomass is altered as a result \
                    of increasing or decreasing  the amount of carbon in \
                    the atmosphere."),
                GWP_calculation,
                dcc.Graph(id='carbon-balance-figure'),
                GWP_explanation
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


########################################
# GWP text inserted above figure
########################################

@app.callback(
    Output(component_id='dynamic-GWP-result', component_property='children'),
    [Input(component_id='annual-carbon-flux', component_property='children')]
)
def update_GWP(net_annual_carbon_flux):

    net_annual_carbon_flux = json.loads(net_annual_carbon_flux)
    # AGWP is the cumulative radiative forcing at time t after the emission
    dynamic_GWP_100 = dynamic_GWP(100, net_annual_carbon_flux)

    return "Global warming potential (100) of net carbon flux: \
        {:.2f} kg CO2 eq".format(dynamic_GWP_100)


##############################
# transfer_coefficients inputs
##############################

transfer_coefficients_input = html.Div(
    className='four columns',
    style={'border-right': 'double', 'padding-right': '0.5em'},
    children=[
        html.H3(
            "Explore how changing how biomass is used affects carbon emissions.",
            ),
        html.P('When trees are harvested, the biomass \
            is transferred to different \'pools\' including harvest residues\
            and products like paper and lumber used buildings.'),
        html.H6('Harvested biomass transfers'),
        html.P('These values represent the ratio of harvested biomass \
               transferred to different \'pools\'.'),
        html.P('Values must sum to 1. Hit the "Update" button below when all \
               transfer coefficients have been updated.',
               style={'padding-top': '1em'}),
        html.P('Harvest residue decay:', style={'font-weight': 'bold'}),
        html.Div(id='TC-table'),
        dcc.Input(
            id='biomass-decay-transfer',
            placeholder='Enter a value between 0-1...',
            type='number',
            min=0,
            max=1,
            step=0.01,
            value=str(DECAY_TC)
        ),
        html.P('Long-lived products:', style={'font-weight': 'bold'}),
        dcc.Input(
            id='long-lived-products-transfer',
            placeholder='Enter a value between 0-1...',
            type='number',
            min=0,
            max=1,
            step=0.01,
            value=LL_PRODUCTS_TC
        ),
        html.P('Short-lived products:', style={'font-weight': 'bold'}),
        dcc.Input(
            id='short-lived-products-transfer',
            placeholder='Enter a value between 0-1...',
            type='number',
            min=0,
            max=1,
            step=0.01,
            value=SL_PRODUCTS_TC
        ),
        html.P('Bioenergy:', style={'font-weight': 'bold'}),
        html.Div([
            dcc.Input(
                id='bioenergy-transfer',
                placeholder='Enter a value between 0-1...',
                type='number',
                min=0,
                max=1,
                step=0.01,
                value=BIOENERGY_TC
            )]),
        html.Div([
            html.Button(
                'Update', id='update-TCs-button',
                n_clicks=0, style={'background-color': 'black', 'color': 'white'}),
            ],
            style={'padding-top': '1em'}),
        html.P(id='validation-text', style={'color': 'red'}),
        dcc.ConfirmDialog(
            id='validate-dialog',
            message='Input Error.  Transfer coefficients must sum to 1.')
        ],

)


@app.callback(
    [
        Output(component_id='carbon-balance-figure', component_property='figure'),
        Output(component_id='annual-carbon-flux', component_property='children'),
        Output('validation-text', 'children'),
        Output('validate-dialog', 'displayed')
        ],
    [
        Input('update-TCs-button', 'n_clicks'),
        Input(component_id='regrowth-slider', component_property='value'),
        Input(component_id='biomass-decay', component_property='value'),
        Input(component_id='short-lived', component_property='value'),
        Input(component_id='long-lived', component_property='value')
        ],
    [
        State(component_id='biomass-decay-transfer', component_property='value'),
        State(component_id='bioenergy-transfer', component_property='value'),
        State(component_id='short-lived-products-transfer', component_property='value'),
        State(component_id='long-lived-products-transfer', component_property='value'),

        ])
def update_figure(
        n_clicks,
        mean_forest, mean_decay, mean_short, mean_long,
        decay_tc, bioenergy_tc, short_products_tc, long_products_tc,
        ):
    total_transfer = validate_transfer_coefficients(
        decay_tc, bioenergy_tc, short_products_tc, long_products_tc
    )
    if np.isclose(total_transfer, 1):
        data, x = generate_flux_data(
            mean_forest, mean_decay, mean_short, mean_long,
            decay_tc, bioenergy_tc, short_products_tc, long_products_tc)

        carbon_model = CarbonModel(data, x=x, name='harvest', initial_carbon=INITIAL_CARBON)
        fig = carbon_model.plot_carbon_balance()
        net_annual_carbon_flux = carbon_model.net_annual_carbon_flux
        return fig, json.dumps(net_annual_carbon_flux.tolist()), '', False

    else:
        return_msg = f'Update transfer coefficients so they sum to 1. \
            The current sum is: {total_transfer}.'
        return dash.no_update, dash.no_update, return_msg, True


def validate_transfer_coefficients(
    decay_tc, bioenergy_tc, short_products_tc, long_products_tc
):
    return np.sum([
                    float(decay_tc),
                    float(bioenergy_tc),
                    float(short_products_tc),
                    float(long_products_tc)
                    ])


#############################
# updating distributions
#############################

def make_slider_makers(min, max, step_size):
    return {x: f'{x} years' for x in range(min, max+1, step_size)}


distribution_selections = html.Div(
    className='row',
    children=[
        html.Div(
            className="six columns",
            children=[
                dcc.Markdown('**Mean age of managed forest** (e.g. changing growth rate by \
                    fertilization, brush clearing and improved re-planting):'),
                dcc.Slider(
                    id='regrowth-slider',
                    min=MEAN_FOREST-15,
                    max=MEAN_FOREST+15,
                    step=1,
                    value=MEAN_FOREST,
                    marks=make_slider_makers(MEAN_FOREST-15, MEAN_FOREST+15, 5)
                ),
                html.Div(id='regrowth-selection'),
                html.Br(),
                dcc.Markdown('**Harvest residue decay half-life**:'),
                dcc.Slider(
                    id='biomass-decay',
                    min=MEAN_DECAY-5,
                    max=MEAN_DECAY+5,
                    step=1,
                    value=MEAN_DECAY,
                    marks=make_slider_makers(MEAN_DECAY-5, MEAN_DECAY+5, 2)
                ),
                html.Div(id='decay-selection'),
                html.Br(),
            ]),
        html.Div(
            className='six columns',
            children=[
                dcc.Markdown('**Short-lived product half-life**: (e.g. improving \
                    waste paper and packaging recycling rates)'),
                dcc.Slider(
                    id='short-lived',
                    min=0,
                    max=MEAN_SHORT+5,
                    step=1,
                    value=MEAN_SHORT,
                    marks=make_slider_makers(0, MEAN_SHORT+5, 2),
                ),
                html.Div(id='short-selection'),
                html.Br(),
                dcc.Markdown('**Long-lived product half-life**: (e.g. increasing \
                    the durability of wood products, and \
                        their reuse in second generation products'),
                dcc.Slider(
                    id='long-lived',
                    min=MEAN_LONG-20,
                    max=MEAN_LONG+20,
                    step=1,
                    value=MEAN_LONG,
                    marks=make_slider_makers(MEAN_LONG-20, MEAN_LONG+20, 10)
                ),
                html.Div(id='long-selection'),
                html.Br(),
            ])])


####################################
# compose application layout
######################################
app.layout = html.Div([
    html.H1("Above ground forest carbon dynamics from harvesting."),
    html.Div([
        transfer_coefficients_input,
        carbon_balance_figure,
        ], className='row'),
    html.Br(),
    html.H3(
        "Explore how re-growth rates and product \
                lifetimes can affect carbon emissions.",
        style={'border-top': 'double'},),
    html.Div([distribution_selections], className='row'),
    html.Div(id='annual-carbon-flux', style={'display': 'none'}),
    html.Div(id='transfer-coefficients', style={'display': 'none'}),
    html.Div(id='transfer-coefficient-validation', style={'display': 'none'})

])

if __name__ == '__main__':
    app.run_server(debug=True)
