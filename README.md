# Buffer Reshuffling and Retrieval (BRR) Optimization

This repository addresses the Buffer Reshuffling and Retrieval (BRR) problem. It implements a Mixed Integer Programming (MIP) model for exact solutions and a scalable heuristic approach designed for larger-scale automated warehouse operations.

<div align="center">
  <video src="https://github.com/user-attachments/assets/c28fd613-cb21-4f2a-8509-ba2d8647413a" width="100%" controls autoplay loop muted></video>
  <br>
  <em>Visualization of the Dynamic Buffer Reshuffling and Retrieval Problem with mutliple Autonomous Mobile Robots.  </em>
</div>

**Note on Visualization & Kinematic Assumptions:** The AMR driving trajectories shown in the video above are simplified for visualization purposes. As detailed in our paper, the optimization model operates under specific kinematic abstractions (e.g., constant velocity, negligible acceleration/deceleration phases). The visualization does not reflect continuous, high-fidelity kinematic constraints. For real-world deployment, we assume that the AMRs are equipped with onboard safety sensors (e.g., LIDAR) and that a dedicated, lower-level fleet management system handles continuous path smoothing and physical collision avoidance. This abstraction allows our models to focus entirely on solving the complex logical coordination, deadlock prevention, and temporal synchronization of the fleet.

## Optimization Methods

The system tackles the problem using two distinct approaches:

### 1. Exact Formulation (MIP)
Uses **Gurobi** to solve a Mixed Integer Programming model. This approach guarantees optimal solutions but is limited to smaller instances due to computational complexity.
- **Source**: `src/integer_programming/`

### 2. Heuristic Solver
Designed for large-scale instances. The heuristic pipeline consists of four intermediate steps:
1.  **Priority Mapping**: Creates a chronological task queue that prioritizes urgent storage and retrieval requests based on time windows.
2.  **A\* Search**: Determines the optimal logical sequence of moves (storage, retrieval, reshuffling) required to fulfill the tasks.
3.  **Scheduling**: Assigns these moves to available vehicles (AMRs) using a VRP solver, respecting time windows and precedence constraints.
4.  **Repair Functions**: Post-processes the schedule to resolve complex vehicle interactions, collisions, or deadlocks.

- **Source**: `src/heuristics/`

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments

**1. Generate Instances & Run MIP (Gurobi)**
```bash
python run_BRR_experiment.py
```

**2. Run Heuristic Solver**
```bash
# Solve a single instance
python run_heuristic_BRR.py --instance examples/Size_3x3_Layout_1x1.csv

# Solve all instances in a directory
python run_heuristic_BRR.py --directory experiments/inputsBRR/Size_3x3_Layout_1x1/

# Auto-solve all instances that have Gurobi results (for comparison)
python run_heuristic_BRR.py --auto-solve
```

**3. Generate Layouts**
```bash
python examples/layout_generator.py
```

## Visualization

The system provides tools to visualize the optimization results:

- **`visualize_BRR_steps.py`**: Visualizes the warehouse state evolution timestep by timestep.
- **`visualize_astar_moves.py`**: Visualizes the logical sequence of A* moves.
- **`visualize_slot_heatmap.py`**: Generates heatmaps showing slot occupancy and usage.

```bash
# Example usage
python visualization/visualize_BRR_steps.py --file experiments/resultsBRR/.../result.json
```

## Project Structure

```
src/
  heuristics/            # A*, Scheduling, Priority Map, Repair logic
  integer_programming/   # Gurobi MIP models
  instance/              # Problem instance loaders
  test_cases/            # Validation and solution checking
  visualization/         # Visualization scripts
experiments/             # Input instances and results
examples/                # Sample layouts and generators
```

## License
MIT License
