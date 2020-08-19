import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from forest_carbon import CarbonFlux, CarbonModel

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
                'Mean age of managed forest:',
                dcc.Slider(
                    id='regrowth-slider',
                    min=20,
                    max=100,
                    step=1,
                    value=40,
                    marks=make_slider_makers(20, 100, 20)
                ),
                html.Div(id='regrowth-selection'),
                html.Br(),
                'Biomass decay half-life:',
                dcc.Slider(
                    id='biomass-decay',
                    min=10,
                    max=40,
                    step=1,
                    value=20,
                    marks=make_slider_makers(10, 40, 10)
                ),
                html.Div(id='decay-selection'),
                html.Br(),
                'short-lived product half-life:',
                dcc.Slider(
                    id='short-lived',
                    min=10,
                    max=40,
                    step=1,
                    marks=make_slider_makers(10, 40, 10),
                    value=20
                ),
                html.Div(id='short-selection'),
                html.Br(),
                'long-lived product half-life:',
                dcc.Slider(
                    id='long-lived',
                    min=30,
                    max=90,
                    step=1,
                    value=60,
                    marks=make_slider_makers(30, 90, 20)
                ),
                html.Div(id='long-selection'),
                html.Br(),
                ])

GWP_calculation = html.Div(
    ["Holding place for GWP result"],
    id='dynamic-GWP-result',
    style={'text-align': 'center'})

carbon_balance_figure = html.Div(
            id='right-column',
            className="eight columns",
            children=[
                html.H5("Cumulative carbon emissions and removals."),
                dcc.Graph(id='carbon-balance-figure'),
                GWP_calculation
                    ])

app.layout = html.Div([
    html.H1("Above ground forest carbon dynamics from harvesting."),
    html.Div([
        distribution_selections,
        carbon_balance_figure
        ], className='row'),
    html.Div(id='carbon-model', style={'display': 'none'})
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
    Output(component_id='carbon-balance-figure', component_property='figure'),
    [
        Input(component_id='regrowth-slider', component_property='value'),
        Input(component_id='biomass-decay', component_property='value'),
        Input(component_id='short-lived', component_property='value'),
        Input(component_id='long-lived', component_property='value'),
    ]
)
def update_figure(mean_forest, mean_decay, mean_short, mean_long):
    forest_regrowth = CarbonFlux(mean_forest, 1.7, 1000, 'forest regrowth', 1, emission=False)
    decay = CarbonFlux(mean_decay, 2, 1000, 'biomass decay', 0.5)
    energy = CarbonFlux(1, 1.05, 1000, 'energy', 0.5*0.1)
    short_lived = CarbonFlux(mean_short, 1.5, 1000, 'short-lived products', 0.5 * 0.4)
    long_lived = CarbonFlux(mean_long, 1.5, 1000, 'long-lived products', 0.5 * 0.5)

    data = {
        'forest_regrowth': forest_regrowth,
        'biomass_decay': decay,
        'energy': energy,
        'short_lived_products': short_lived,
        'long_lived_products': long_lived}

    carbon_model = CarbonModel(data, 'harvest')
    fig = carbon_model.plot_carbon_balance()
    return fig

"""
@app.callback(
    Output(component_id='dynamic-GWP-result', component_property='children'),
    [
        Input(component_id='regrowth-slider', component_property='value'),
        Input(component_id='biomass-decay', component_property='value'),
        Input(component_id='short-lived', component_property='value'),
        Input(component_id='long-lived', component_property='value'),
        Input(component_id='carbon-model', component_property='children'),
    ]
)
def update_GWP(mean_forest, mean_decay, mean_short, mean_long, carbon_model):
    print(carbon_model.net_annual_carbon_flux[0:10])
    pass
"""


if __name__ == '__main__':
    app.run_server(debug=True)
