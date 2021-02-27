from plotly import express as px
from plotly import graph_objects as go


def radiative_forcing_plot(radiative_forcing, time_horizon):
    fig = px.line(x=time_horizon, y=radiative_forcing, template='simple_white')
    fig.add_hline(y=0.0)
    fig.update_yaxes(
        title={'text': 'Radiative forcing (W per m2)'},
        tickformat='e')
    fig.update_xaxes(title={'text': 'Time since harvest (years)'},)
    fig.add_trace(go.Scatter(x=[0, 120], y=[0, 0], mode='lines', marker_color='black'))
    fig.update_layout(
        showlegend=False,)
    return fig


def temperature_response_plot(temp_response, time_horizon):
    fig = px.line(x=time_horizon, y=temp_response, template='simple_white')
    fig.update_yaxes(
        title={'text': 'Temperature response (k)'},
        tickformat='e')
    fig.update_xaxes(title={'text': 'Time since harvest (years)'},)
    fig.add_trace(go.Scatter(x=[0, 120], y=[0, 0], mode='lines', marker_color='black'))
    fig.update_layout(
        showlegend=False)
    return fig
