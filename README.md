# Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP) Optimization

This repository addresses the Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP). It implements a Mixed Integer Programming (MIP) model for exact solutions and a scalable heuristic approach designed for larger-scale automated warehouse operations.

<div align="center">
  <video src="https://github.com/user-attachments/assets/ad030e7a-47ce-4a87-a2b1-76a5db733046" width="100%" controls autoplay loop muted></video>
  <br>
  <em>Visualization of the Dynamic Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP) with multiple Autonomous Mobile Robots.  </em>
</div>

**Note on Visualization & Kinematic Assumptions:** The AMR driving trajectories shown in the video above are simplified for visualization purposes. As detailed in our paper, the optimization model operates under specific kinematic abstractions (e.g., constant velocity, negligible acceleration/deceleration phases). The visualization does not reflect continuous, high-fidelity kinematic constraints. For real-world deployment, we assume that the AMRs are equipped with onboard safety sensors (e.g., LIDAR) and that a dedicated, lower-level fleet management system handles continuous path smoothing and physical collision avoidance. This abstraction allows our models to focus entirely on solving the complex logical coordination, deadlock prevention, and temporal synchronization of the fleet.

## Optimization Methods

## 🚀 Quick Start

### Installation
```bash
# Install as a package (recommended for use as a submodule)
pip install -e .

# Or just install required packages
pip install -r requirements.txt
```

### Running Experiments

**1. Generate Instances & Run MIP (Gurobi)**
```bash
# Generate new instances and solve with Gurobi
python run_BSRRP_experiment.py
```

**2. Run Heuristic Solver**
```bash
# Solve a single instance
python run_heuristic_BSRRP.py --instance examples/Size_3x3_Layout_1x1.csv

# Solve all instances in a directory
python run_heuristic_BSRRP.py --directory experiments/inputsBSRRP/Size_3x3_Layout_1x1/

# Auto-solve all instances that have Gurobi results (for comparison)
python run_heuristic_BSRRP.py --auto-solve
```

**3. Generate Layouts**
```bash
# Generate new warehouse layouts
python examples/layout_generator.py
```

## 🔧 Core BSRRP System

### Optimization Methods

The system tackles the problem using two distinct approaches:

#### 1. Exact Formulation (MIP)
Uses **Gurobi** to solve a Mixed Integer Programming model. This approach guarantees optimal solutions but is limited to smaller instances due to computational complexity.
- **Source**: `src/bsrrp/integer_programming/`

#### 2. Heuristic Solver
Designed for large-scale instances. The heuristic pipeline consists of four intermediate steps:
1.  **Priority Mapping**: Creates a chronological task queue that prioritizes urgent storage and retrieval requests based on time windows.
2.  **A\* Search**: Determines the optimal logical sequence of moves (storage, retrieval, reshuffling) required to fulfill the tasks.
3.  **Scheduling**: Assigns these moves to available vehicles (AMRs) using a VRP solver (OR-Tools CP-SAT), respecting time windows and precedence constraints.
4.  **Repair Functions**: Post-processes the schedule to resolve complex vehicle interactions, collisions, or deadlocks.
- **Source**: `src/bsrrp/heuristics/`

### Key Features

- **Multi-objective optimization**: Minimizes total cost while respecting time windows
- **Constraint handling**: Source blocking prevention, vehicle capacity limits
- **Performance optimization**: O(1) heuristics, efficient data structures
- **Comprehensive validation**: Automated solution checking and comparison

## 🛠️ System Components

### 1. Heuristic Solver (`run_heuristic_BSRRP.py`)

Fast near-optimal solutions using A* search and vehicle routing optimization.

**Features:**
- **Comparison with Gurobi results** when available
- **Batch processing** for multiple instances
- **Auto-solve mode** to process all instances with Gurobi results

**Usage:**
```bash
# With custom time limits
python run_heuristic_BSRRP.py --instance instance.json --astar-time-limit 600 --vrp-time-limit 300

# Compare with specific Gurobi result
python run_heuristic_BSRRP.py --instance instance.json --gurobi-result gurobi_result.json

# Disable automatic visualization (enabled by default)
python run_heuristic_BSRRP.py --instance instance.json --no-visualize
```

### 2. Visualization Tools

This repository provides complementary visualization scripts plus automatic visualization.

#### Main Visualization Scripts

1. **`visualize_BSRRP_steps.py`** - Timestep visualization
   - Visualizes warehouse state per **timestep** using decision variables
   - Works with Gurobi optimization results and heuristic results
   - Shows warehouse evolution over time

2. **`visualize_astar_moves.py`** - A* move visualization  
   - Visualizes warehouse state after each **A* move**
   - Works specifically with heuristic results containing A* data
   - Shows the logical sequence of warehouse operations

3. **Heatmap Analysis Tools**
   - **`visualize_slot_heatmap.py`**: Visualizes average slot occupancy duration.
   - **`visualize_slot_heatmap_aisles.py`**: Visualizes combined occupancy and vehicle travel time in aisles.
   - **`visualize_ap_usage.py`**: Visualizes Access Point usage frequency using color-coded circles.

#### Automatic Visualization System

The `bsrrp.visualization.auto_visualize` module provides automatic, multiprocessing-enabled visualization:

**Features:**
- **Multiprocessing support** for concurrent visualization creation
- **Automatic script detection** and appropriate visualization selection
- **Background processing** to avoid blocking main computations
- **Overwrite protection** with configurable behavior
- **Batch processing** capabilities for multiple result files

**Integration:**
- **Automatically used** by `run_heuristic_BSRRP.py` after solving
- **Configurable** via command-line arguments (`--auto-visualize`, `--no-visualize`)
- **Efficient** for large experiment batches with parallel processing

#### Usage Examples

**Timestep Visualization**
```bash
# Basic timestep visualization
python visualization/visualize_BSRRP_steps.py --file <result_file.json>

# With virtual lane coloring
python visualization/visualize_BSRRP_steps.py --file <result_file.json> --color

# Create subplot for specific timesteps
python visualization/visualize_BSRRP_steps.py --file <result_file.json> --subplot-timesteps "0,25,50"
```

**A* Move Visualization**
```bash
# Basic A* move visualization (with coloring by default)
python visualization/visualize_astar_moves.py --file <heuristic_result_file.json>

# Without virtual lane coloring
python visualization/visualize_astar_moves.py --file <heuristic_result_file.json> --no-color
```

**Heatmap Utilization Analysis**
These scripts generate analytical heatmaps in the `heatmaps/` directory.

```bash
# Analyze slot occupancy duration (average over an instance type)
python visualization/visualize_slot_heatmap.py --instance-type manual2

# Analyze combined aisle travel and slot occupancy
python visualization/visualize_slot_heatmap_aisles.py --instance-type manual2

# Analyze Access Point usage frequency
python visualization/visualize_ap_usage.py --instance-type manual2

# Analyze a single specific result file
python visualization/visualize_slot_heatmap.py --file <path_to_result_file.json>
```

## ⚙️ Algorithm Comparison

### Gurobi Optimization vs. Heuristic Approach

| Aspect | Gurobi (Optimization) | Heuristic (A* + VRP) |
|--------|----------------------|----------------------|
| **Optimality** | Guaranteed optimal (within time limit) | Near-optimal solutions |
| **Scalability** | Limited for large instances | Better scalability |
| **Speed** | Can be slow for complex instances | Generally faster |
| **Output file** | `*.json` | `*_heuristic.json` |
| **Move sequence** | Decision variables by timestep | Logical A* moves + VRP assignment |
| **Use case** | Small to medium instances, benchmarking | Large instances, real-time applications |

### When to Use Each Approach

**Use Gurobi when:**
- Instance size is manageable (small to medium)
- Optimal solutions are required
- You have sufficient computational time
- Benchmarking heuristic approaches

**Use Heuristic when:**
- Large instances that Gurobi struggles with
- Fast solutions are needed
- Near-optimal solutions are acceptable
- Real-time or online scenarios

## 🛠️ System Requirements

### Prerequisites
- Python 3.8+
- Gurobi Optimizer (with valid license for exact optimization)
- Required Python packages: `numpy`, `matplotlib`, `imageio`, `argparse`, `gurobipy`, `pandas`, `seaborn`, `ortools`

### Gurobi Setup
Ensure Gurobi is properly installed and licensed on your system for the exact optimization solver.

## 📁 Directory Structure

The system organizes experiments in a structured hierarchy:

```
buffer_reshuffling_and_retrieval_ip/
├── src/                                # Source code
│   └── bsrrp/                          # Root package
│       ├── heuristics/                 # A* search and VRP algorithms
│       ├── integer_programming/        # Gurobi MIP models
│       ├── instance/                   # Problem instance loaders
│       ├── test_cases/                 # Validation functions
│       └── visualization/              # Auto-visualization modules
├── examples/                           # Layout generators and sample instances
├── experiments/
│   ├── inputsBSRRP/                    # Instance files
│   ├── resultsBSRRP/                   # Solution files  
│   ├── hashesBSRRP/                    # Hash tracking
│   └── feasibleBSRRP/                  # Feasibility information
├── heatmaps/                           # Generated heatmaps
├── run_BSRRP_experiment.py             # Main experiment runner
├── run_heuristic_BSRRP.py              # Heuristic solver
├── visualization/                      # Visualization scripts
│   ├── visualize_BSRRP_steps.py        # Timestep visualization
│   ├── visualize_astar_moves.py        # A* move visualization
│   └── ...                             # Heatmap scripts
└── README.md                           # This file
```

## 🤝 Contributing

The system is designed for extensibility:
- Add new heuristics in `src/bsrrp/heuristics/`
- Extend validation in `src/bsrrp/test_cases/`
- Create new instance generators in `examples/`
- Enhance visualization in `src/bsrrp/visualization/` and `visualization/`

## 📝 Citation

If you use this code in your research, please cite our repository and the corresponding paper. 

**Code Repository:**
```bibtex
@misc{disselnmeyer2025bsrrp_code,
  author       = {Disselnmeyer, Max},
  title        = {Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP) Optimization},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{[https://github.com/mdisse/bsrrp_ip_heuristic](https://github.com/mdisse/bsrrp_ip_heuristic)}},
}
```

## 💳 Funding

This work was supported by the **Federal Ministry for Economic Affairs and Climate Action (BMWK)** [grant number 13IK0321] and the **European Union – NextGenerationEU**.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📦 Using as a Submodule

To use this repository as a submodule in another project:

1. Add the submodule to your project:
   ```bash
   git submodule add https://github.com/yourusername/buffer_reshuffling_and_retrieval_ip.git
   ```
2. Install the package in editable mode from your project's root:
   ```bash
   pip install -e ./buffer_reshuffling_and_retrieval_ip
   ```
3. You can now import the modules in your code:
   ```python
   from bsrrp.heuristics.astar import AStarSolver
   from bsrrp.test_cases.test_case_bsrrp import TestCaseBsrrp
   ```
