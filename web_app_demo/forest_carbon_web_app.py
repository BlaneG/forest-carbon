import json

import sys
from dash_bootstrap_components._components.Tooltip import Tooltip
sys.path.append('..')

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from figures import (
    radiative_forcing_plot,
    temperature_response_plot
)
from forest_carbon import CarbonModel
from forest_carbon_model import generate_flux_data, INITIAL_CARBON, lifetimes, STEP
from ghg_tools.climate_metrics import (
    dynamic_AGWP,
    dynamic_GWP,
    temperature_response,
    radiative_forcing_from_emissions_scenario
)


style_defaults = {'width': '90%', 'margin': 'auto'}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = [dbc.themes.LITERA]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server  # required for Heroku deployment


# biomass transfer coefficients
DECAY_TC = 0.5
LL_PRODUCTS_TC = (1-DECAY_TC) * 0.5
SL_PRODUCTS_TC = (1-DECAY_TC) * 0.4
BIOENERGY_TC = (1-DECAY_TC) * 0.1


####################################
# Carbon balance figure
#####################################
GWP_calculation = html.Div(
    id='dynamic-GWP-result',
    style={'text-align': 'center'},
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

carbon_balance_html = [
    html.H5(
        "Cumulative carbon emissions and removals",
        id='figure-section-header'),
    dbc.Tooltip(
        "Changing how biomass is used affects the \
        area under the 'Net CO2 flux' curve which \
        represents carbon that has \
        temporarily been added to the atmosphere.  We can estimate the \
        climate impact of this temporary increase in atmospheric \
        carbon using a modification to the well-known global \
        warming potential (GWP) metric which is measured in \
        units of CO2 equivalents. Strategies that store more \
        carbon can reduce the climate effect measured using GWP.",
        target='figure-section-header'
    ),
    GWP_calculation,
    dcc.Graph(id='carbon-balance-figure'),
    GWP_explanation
]

radiative_forcing_html = [
    html.H3("Radiative forcing from net emissions and removals"),
    html.P(
        'The radiative forcing response represents the change\
         in the planetary energy balance, measured at the top\
         tropopause, in response to annual CO2 emissions and removals.'),
    dcc.Graph(id='radiative-forcing-figure')]
temperature_response_html = [
    html.H3("Temperature response to net emissions and removals"),
    html.P(
        'The temperature response represents the change\
         in the average surface temperature in response\
         to annual CO2 emissions and removals.'),
    dcc.Graph(id='temperature-response-figure')]
figures_children = carbon_balance_html\
    + radiative_forcing_html\
    + temperature_response_html

figures_div = dbc.Col(
            id='figures-div',
            children=figures_children)


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
    dynamic_GWP_100 = dynamic_GWP(100, net_annual_carbon_flux[0:1001], ghg='co2', step_size=STEP)

    return html.Div(
        [html.Span(
            "Dynamic global warming potential (100) for the net carbon flux \
             from harvesting merchantable biomass containing 1 tonne CO2: "),
        html.Span("{:.2f} kg CO2 eq".format(dynamic_GWP_100), style={'font-weight': 'bold'})
        ])


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
    if np.isclose(total_transfer/1, 1):
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



@app.callback(
    Output(component_id='radiative-forcing-figure', component_property='figure'),
    Input(component_id='annual-carbon-flux', component_property='children'),
)
def update_radiative_forcing_graph(annual_carbon_flux):
    annual_carbon_flux = json.loads(annual_carbon_flux)
    rf = radiative_forcing_from_emissions_scenario(
        time_horizon=120,
        emissions=annual_carbon_flux[0:1201], ghg='co2', step_size=0.1, mode='full')
    fig = radiative_forcing_plot(rf, np.arange(0, 120.1, 0.1))
    return fig


@app.callback(
    Output(component_id='temperature-response-figure', component_property='figure'),
    Input(component_id='annual-carbon-flux', component_property='children'),
)
def update_radiative_forcing_graph(annual_carbon_flux):
    annual_carbon_flux = json.loads(annual_carbon_flux)
    rf = temperature_response(
        time_horizon=120,
        emissions=annual_carbon_flux[0:1201], ghg='co2', step_size=0.1, mode='full')
    fig = temperature_response_plot(rf, np.arange(0, 120.1, 0.1))
    return fig
#############################
# Updating half-lives
#############################

def make_slider_makers(min, max, step_size):
    return {x: f'{x}' for x in range(min, max+1, step_size)}


regrowth_slider = dcc.Slider(
                    id='regrowth-slider',
                    min=lifetimes['MEAN_FOREST']-5,
                    max=lifetimes['MEAN_FOREST']+5,
                    step=1,
                    value=lifetimes['MEAN_FOREST'],
                    marks=make_slider_makers(
                        lifetimes['MEAN_FOREST']-5,
                        lifetimes['MEAN_FOREST']+5, 5)
                )

biomass_decay_slider = dcc.Slider(
                    id='biomass-decay',
                    min=lifetimes['MEAN_DECAY']-5,
                    max=lifetimes['MEAN_DECAY']+5,
                    step=1,
                    value=lifetimes['MEAN_DECAY'],
                    marks=make_slider_makers(
                        lifetimes['MEAN_DECAY']-5,
                        lifetimes['MEAN_DECAY']+5, 2)
                )

short_lived_slider = dcc.Slider(
                    id='short-lived',
                    min=0,
                    max=lifetimes['MEAN_SHORT']+5,
                    step=1,
                    value=lifetimes['MEAN_SHORT'],
                    marks=make_slider_makers(
                        0, lifetimes['MEAN_SHORT']+5, 2),
                )

long_lived_slider = dcc.Slider(
                    id='long-lived',
                    min=lifetimes['MEAN_LONG']-10,
                    max=lifetimes['MEAN_LONG']+10,
                    step=1,
                    value=lifetimes['MEAN_LONG'],
                    marks=make_slider_makers(
                        lifetimes['MEAN_LONG']-10,
                        lifetimes['MEAN_LONG']+10, 5)
                )


regrowth_layout = dbc.Col(dbc.FormGroup([
    dbc.Label("Forest re-growth", html_for="regrowth-slider"),
    regrowth_slider,
    dbc.FormText(
            "e.g. changing growth rate by fertilization, brush clearing\
             and improved re-planting",
             color="secondary"
        ),

]))

biomass_decay_layout = dbc.Col(dbc.FormGroup([
    dbc.Label("Biomass decay", html_for="biomass-decay"),
    biomass_decay_slider,
    dbc.FormText(
            'Mainly affected by climate', color="secondary"
        ),
]))

short_lived_layout = dbc.Col(dbc.FormGroup([
    dbc.Label("Short-lived products", html_for="short-lived"),
    short_lived_slider,
    dbc.FormText(
            'e.g. improving waste paper and packaging recycling rates can extend the half-life',
            color="secondary"
        ),
]))

long_lived_layout = dbc.Col(dbc.FormGroup([
    dbc.Label("Long-lived products", html_for="long-lived"),
    long_lived_slider,
    dbc.FormText(
            'e.g. increasing the durability of wood products, and \
             their reuse in second generation products',
            color="secondary"
        ),
]))

half_lives = html.Div([
    html.H5(
        "Carbon transfer half-lives (years)",
        id='carbon-transfer-hovertext'
    ),
    dbc.Tooltip(
        'Explore how re-growth rates and product lifetimes can affect\
         carbon emissions.Half-lives describe the time it takes for half\
         of the carbon to \
         transfer into or out of a carbon pool. For example, the half-life\
         for forests is the time it takes to absorb 50% of the max potential\
         carbon stock of the forest ecosystem.',
        target="carbon-transfer-hovertext",),
    dbc.Row([regrowth_layout, biomass_decay_layout], form=True),
    dbc.Row([short_lived_layout, long_lived_layout], form=True)
], style={'border-bottom': 'double', **style_defaults})


##############################
# Transfer_coefficients inputs
##############################
TC_row_1 = dbc.Row(
    children=[
        dbc.Col(
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
        dbc.Col(
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

TC_row_2 = dbc.Row(
    children=[
        dbc.Col(
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
        dbc.Col(
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


transfer_coefficients = html.Div([
    html.H5(
        ['Fraction of harvested biomass trasnferred to different pools'],
        id='transfer-coefficient-hovertext'),
    dbc.Tooltip(
        'When trees are harvested, biomass \
        is transferred to different \'pools\' including harvest residues\
        and products like paper and lumber used in buildings.'
        'The values below represent the ratio of harvested biomass \
            transferred to different \'pools\'. Increasing the ratio \
            of biomass transferred to longer storage pools helps to \
            reduce the amount of carbon emitted to the atmosphere.',
        target='transfer-coefficient-hovertext',
        style={}
    ),
    html.P('Values below must sum to 1. Hit the "Update" button below when all \
            transfer coefficients have been updated.',
            style={'padding-top': '1em'}),
    TC_row_1,
    TC_row_2,
    html.Div([
        html.Button(
            'Update', id='update-TCs-button',
            n_clicks=0,
            style={'background-color': 'black', 'color': 'white'}
            ),
        ],
        style={'padding-top': '1em', 'text-align': 'center'}),
    html.P(id='validation-text', style={'color': 'red'}),
    dcc.ConfirmDialog(
        id='validate-dialog',
        message='Input Error.  Transfer coefficients must sum to 1.')
]
)

user_inputs = dbc.Col(
    width=5,
    style={'border-right': 'double', 'padding-right': '0.5em'},
    children=[
        html.H4(
            "Explore how human activities can affect carbon emissions and removals.",
            ),
        half_lives,
        transfer_coefficients
        ],

)


####################################
# About tab
####################################


about = html.Div(
    style={'padding-left': '25%', 'padding-right': '25%'},
    children=[
        html.H1("About"),
        html.P("Humans have been dependent on biomass resources throughout our \
            existence for food, clothing, shelter, communication (paper), \
            sanitary products and more. This relationship has changed dramatically \
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

    html.H1(
        "Above ground forest carbon dynamics after harvesting",
        style=style_defaults),
    html.P(
        'When trees are harvested, biomass is transferred to different\
         \'pools\' including harvest residues, products like paper\
         and lumber used in buildings, and to the atmosphere (burning\
         harvest slash).'
        ' Harvesting also creates an opening in\
         the forest canopy providing light that stimulates new growth.\
         Human activities that, for example, change the ratio of carbon\
         stored in long-lived products or alter the growth-rate of forests\
         can affect the carbon balance between the atmosphere, forest\
         ecosystems and the anthroposphere.  This page allows you to\
         interactively explore how human activities can influence carbon\
         dynamics for managed forests and different climate metrics.',
        style=style_defaults),
    dbc.Row([
        user_inputs,
        figures_div,
        ], style=style_defaults),
    html.Br(),
    # Hidden divs for caching data.
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
