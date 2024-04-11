# Project 1

## Prerequisites

You will need python3 and pip3 installed for this as well as git installed. I am working on a Linux machine so the environments will differ.

## Getting started

First create a virtual environment in the project repo, this is best practice when working with Python environments.

Navigate to the repository directory in a terminal and execute the following commands to setup the environment.

These commands will change if you are on Windows:

```bash
python3 -m venv ./venv
```

```bash
source activate venv/bin/activate
```

```bash
pip3 install -e ".[dev]"
```

Dagster is a Python package I used to save intermediary artifacts with a cache to speed up the process. It creates dependencies between software artifacts and runs them in order. It is not necessary to use the software but does make it easier. To use the software start the Dagster UI web server with the following command:

```bash
dagster dev
```

The UI will be served at http://localhost:3000

To run all the intermediate artifacts use the Materialize All button and it will re-generate all of the data files. The notebooks will not run even though they show up in the graph. To run a notebook load it up and run the first cell to pull in the data needed for that notebook and then run the cells in order.
