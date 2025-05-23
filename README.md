# Buffer Reshuffling and Retrieval (BRR)

This repository contains the implementation for solving the Buffer Reshuffling and Retrieval (BRR) problem, focusing on integer programming models for efficient warehouse operations. It includes functionalities for instance generation, solving with Gurobi, checking results, and visualizing the buffer state over time.

## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
    * [Running Experiments](#running-experiments)
    * [Checking Results](#checking-results)
    * [Visualizing Steps](#visualizing-steps)
* [Models](#models)

## Introduction

The Buffer Reshuffling and Retrieval (BRR) problem addresses the optimization of unit load movements within a warehouse buffer, involving tasks such as retrieval, storage, and internal reshuffling. This project provides integer programming models to tackle this complex logistics challenge, aiming to minimize travel distances.

## Features

* **Instance Generation**: Dynamically create various warehouse instances with configurable parameters like layout, fill level, access directions, and unit load characteristics.
* **Integer Programming Models**: Includes both static and dynamic multiple-vehicle models for solving the BRR problem using Gurobi.
* **Solution Validation**: A dedicated script to check the feasibility and validity of generated solutions.
* **Visualization**: Generate visual representations of warehouse states and AMR movements over time, including GIF animations and combined subplot figures.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <repository_url>
    cd buffer_reshuffling_and_retrieval_ip
    ```
2.  **Install Gurobi:** This project uses Gurobi as the optimization solver. Ensure you have Gurobi installed and a valid license configured on your system. Refer to the [Gurobi Documentation](https://www.gurobi.com/documentation/) for installation instructions.
3.  **Install Python Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
    The required packages are:
    * `gurobipy==12.0.2`
    * `imageio==2.36.1`
    * `matplotlib==3.10.3`
    * `networkx==3.4.2`
    * `numpy==2.2.6`
    * `ortools==9.12.4544`
    * `pandas==2.2.3`
    * `tupledict==1.1`

## Usage

### Running Experiments

Experiments can be run using `run_BRR_experiment.py`. This script can either generate new instances or process existing ones.

* **Generate and Solve New Instances**:
    The `generate_instances()` function in `run_BRR_experiment.py` defines the parameters for new instances (e.g., `Size_3x3_Layout_1x1_sink_source`, `fill_levels=[0.1]`, `time_window_lengths=[30]`, `fleet_sizes=[1]`, etc.).
    ```bash
    python run_BRR_experiment.py --verbose
    ```
    The `--verbose` flag provides more detailed output during the solving process.

* **Process Existing Instances**:
    To re-process instances already present in `experiments/inputsBRR/`:
    ```bash
    python run_BRR_experiment.py --check-existing --verbose
    ```
* **Solve a Specific Instance**:
    You can specify a path to an existing JSON instance file:
    ```bash
    python run_BRR_experiment.py --instance experiments/inputsBRR/Size_3x3_Layout_1x1_sink_source/handling_time_1/unit_loads_1/access_directions_1111/rs_max_100/as_max_50/tw_length_30/speed_1/fleet_size_1/1.json --verbose
    ```
    This will solve the particular instance located at the given path.

Solutions and feasibility statuses are saved in `experiments/resultsBRR/` and `experiments/feasibleBRR/` respectively, following a structured path based on instance parameters.

### Checking Results

The `result_checker_BRR.py` script allows you to load a solution and interactively modify or inspect the decisions.

```bash
python result_checker_BRR.py --solution_path <path_to_solution_json_file>
```
After loading, you can used commands like `add <new_decision>`, `change <index> <new_decision>`, `delete <index>`, `help` and `done` to interact with the decisions. 

### Visualizing Steps 

The `visualize_BRR_steps.py` script generates visualizations of the warehouse state over time based on a solved instance's JSON output. 

* **Visualize a single experiment and create a GIF**:
    ```bash 
    python visualize_BRR_steps.py --all
    ``` 
    This processes all solution files listed in `experiments/hashesBRR/results.txt`. 
* **Generate a subplot image for specific timesteps**:
    You can specify a comma-seperated list of timestamps to include in a single combined image: 
    ```bash
    python visualize_BRR_steps.py --file <path_to_solution_json_file> --subplot-timesteps "0, 5, 10, 15"
    ```
    This will create a `subplots_<timesteps>.png` file containing the specified states. 
* **Additional Visualization Flags**:
    * `--no-source`: Excludes source points from the visualization. 
    * `--color-vls`: Colors virtual lanes in the visualization. 
    * `--debug`: Enables debug information in the visualization 

## Models

The core of the solution lies in the integer programming models implemented using Gurobi.

* **Static Model (`static_model.py`)**: This model likely represents a simplified version or a baseline for the problem, possibly assuming a fixed state or less dynamic behavior.
* **Dynamic Multiple Model (`dynamic_multiple_model.py`)**: This model addresses the dynamic nature of the problem with multiple autonomous mobile robots (AMRs), considering their movements and interactions over time. It includes parameters for time limit, thread usage, MIP gap, and numerical accuracy.

Both models define various binary variables (e.g., `b_vars` for unit load presence, `x_vars` for relocations, `y_vars` for retrievals, `e_vars` for empty repositioning, `c_vars` for vehicle location, `g_vars` for retrieved unit loads, `z_vars` for stored unit loads, `s_vars` for stored unit loads) and a comprehensive set of constraints to ensure feasible and optimal solutions. The objective functions aim to minimize travel distances.