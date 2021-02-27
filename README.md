This is a repository of tools for evaluating and exploring climate effects linked to forest products.

climate_metrics.py: an implementation of the IPCC GWP method.
forest_carbon.py: A class for modelling carbon dynamics from forest disturbance.
web_app_demo/forest_carbon_web_app.py: a [web application](https://forest-carbon-app.herokuapp.com/) to explore strategies for reducing the climate effects of using forest products.

## Development
Tests can be run from the top level directory with `python -m pytest tests`. The [-m flag](https://docs.pytest.org/en/stable/pythonpath.html#invoking-pytest-versus-python-m-pytest) adds the current directory to sys.path.

## Deployment
The web_app_demo is deployed from Github using Heroku.  The Procfile points to the application from the main directory.
