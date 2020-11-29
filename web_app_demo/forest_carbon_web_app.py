import json
from typing import Tuple
import sys
sys.path.append('..')

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from ghg_tools.forest_carbon import CarbonFlux, CarbonModel
from ghg_tools.climate_metrics import dynamic_GWP

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # required for Heroku deployment


# Initialization
STEP = 0.1
MANAGED = True

# lifetimes
MEAN_FOREST = 50
MEAN_DECAY = 20
MEAN_BIOENERGY = 1
MEAN_SHORT = 5
MEAN_LONG = 50

HARVEST_YEAR = 90
HARVEST_INDEX = int(HARVEST_YEAR/STEP)

# Initial Carbon
if MANAGED:
    # for managed forests the 

    forest_regrowth = CarbonFlux(
        MEAN_FOREST, MEAN_FOREST * 0.45, 'forest regrowth',
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
        html.A(
            "GWP", href='http://www.ipcc.ch/report/ar5/wg1/',
            target='_blank'),
        "."]
)

carbon_balance_figure = html.Div(
            id='right-column',
            className="eight columns",
            children=[
                html.H3("Cumulative carbon emissions and removals."),
                html.P(
                    "By changing how biomass is used, this can influence the \
                    area under the 'net C flux' curve. The area under this \
                    curve represents the additional carbon that is \
                    temporarily added to the atmosphere.  We can estimate the \
                    climate impact of this temporary increase in atmospheric \
                    carbon using a modification to the well-known global \
                    warming potential (GWP) method which is measured in \
                    units of CO2 equivalents. Strategies that store more \
                    carbon result in a net GWP benefit."),
                GWP_calculation,
                dcc.Graph(id='carbon-balance-figure'),
                GWP_explanation
                    ])


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
    dynamic_GWP_100 = dynamic_GWP(100, net_annual_carbon_flux, step_size=STEP)

    return "Global warming potential (100) of net carbon flux: \
        {:.2f} kg CO2 eq".format(dynamic_GWP_100)


##############################
# transfer_coefficients inputs
##############################
TC_row_1 = html.Div(
    className='row',
    children=[
        html.Div(
            className='six columns',
            children=[
                html.P('Harvest residue decay:',
                       style={'font-weight': 'bold'}),
                dcc.Input(
                    id='biomass-decay-transfer',
                    placeholder='Enter a value between 0-1...',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=str(DECAY_TC)
                )
                ]),
        html.Div(
            className='six columns',
            children=[
                html.P('Long-lived products:', style={'font-weight': 'bold'}),
                dcc.Input(
                    id='long-lived-products-transfer',
                    placeholder='Enter a value between 0-1...',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=LL_PRODUCTS_TC
                )
                ])
    ])

TC_row_2 = html.Div(
    className='row',
    children=[
        html.Div(
            className='six columns',
            children=[
                html.P('Short-lived products:', style={'font-weight': 'bold'}),
                dcc.Input(
                    id='short-lived-products-transfer',
                    placeholder='Enter a value between 0-1...',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=SL_PRODUCTS_TC
                )
                ]),
        html.Div(
            className='six columns',
            children=[
                html.P('Bioenergy:', style={'font-weight': 'bold'}),
                dcc.Input(
                    id='bioenergy-transfer',
                    placeholder='Enter a value between 0-1...',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=BIOENERGY_TC
                )
            ])
    ])

transfer_coefficients_input = html.Div(
    className='four columns',
    style={'border-right': 'double', 'padding-right': '0.5em'},
    children=[
        html.H3(
            "Explore how changing the way biomass is used affects carbon \
                emissions.",
            ),
        html.P('When trees are harvested, biomass \
            is transferred to different \'pools\' including harvest residues\
            and products like paper and lumber used buildings.'),
        html.H6('Harvested biomass transfers'),
        html.P('These values represent the ratio of harvested biomass \
               transferred to different \'pools\'. Increasing the ratio \
                of biomass transferred to longer storage pools helps to \
                reduce the amount of carbon emitted to the atmosphere.'),
        html.P('Values below must sum to 1. Hit the "Update" button below when all \
               transfer coefficients have been updated.',
               style={'padding-top': '1em'}),
        TC_row_1,
        TC_row_2,
        html.Div([
            html.Button(
                'Update', id='update-TCs-button',
                n_clicks=0,
                style={'background-color': 'black', 'color': 'white'}),
            ],
            style={'padding-top': '1em', 'text-align': 'center'}),
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

        carbon_model = CarbonModel(
            data, x=x, name='harvest', initial_carbon=INITIAL_CARBON)
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
                    min=MEAN_FOREST-5,
                    max=MEAN_FOREST+5,
                    step=1,
                    value=MEAN_FOREST,
                    marks=make_slider_makers(MEAN_FOREST-5, MEAN_FOREST+5, 5)
                ),
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
                html.Br(),
                dcc.Markdown('**Long-lived product half-life**: (e.g. increasing \
                    the durability of wood products, and \
                        their reuse in second generation products'),
                dcc.Slider(
                    id='long-lived',
                    min=MEAN_LONG-10,
                    max=MEAN_LONG+10,
                    step=1,
                    value=MEAN_LONG,
                    marks=make_slider_makers(MEAN_LONG-10, MEAN_LONG+10, 5)
                ),
                html.Br(),
            ])])


####################################
# About tab
####################################


about = html.Div(
    style={'padding-left': '25%', 'padding-right': '25%'},
    children=[
        html.H1("About"),
        html.P("Humans have been dependent on biomass resources throughout our \
            existence for food, clothing, shelter communication (paper) and \
            sanitary products. This relationship has changed dramatically \
            over time from our hunter-gather ancestors to the present."),
        html.P("This site provides a high level tool to think about \
            how changing the way we use forest biomass can influence \
            the climate by increasing or decreasing the amount of carbon \
            stored in forests and products. "),
        html.P('The current site considers a single forest stand that is \
            harvested and transferred into both dead biomass and products.'),
        html.H5("References"),
        html.P(children=[
            'The source code for this application is available on ', 
            html.A('github', href='https://github.com/BlaneG/ghg-tools',
                target='_blank'), '.']),
        html.P()

])



####################################
# main page layout
###################################
main_page = html.Div([
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


####################################
# compose application layout
######################################
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Main page', children=[main_page]),
        dcc.Tab(label="About", children=[about]),
        ])
])


if __name__ == '__main__':
    app.run_server(debug=True)
