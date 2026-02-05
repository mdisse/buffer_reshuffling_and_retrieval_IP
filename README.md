# Buffer Reshuffling and Retrieval (BRR) Optimization System

This repository contains the Buffer Reshuffling and Retrieval (BRR) optimization system for automated warehouse operations. The system provides both exact optimization (Gurobi) and fast heuristic approaches for solving complex vehicle routing and unit load management problems.

## 🚀 Quick Start

```bash
# Run heuristic approach on a sample instance
python run_heuristic_BRR.py --instance examples/Size_3x3_Layout_1x1.csv

# Generate new instances
python examples/layout_generator.py

# Validate solution quality
python result_checker_BRR.py --solution_path experiments/resultsBRR/.../solution.json

# Visualize results
python visualize_BRR_steps.py --file experiments/resultsBRR/.../result.json
```

## 🔧 Core BRR System

### Core Components

The BRR system consists of several integrated components:

1. **A* Search Algorithm** (`src/heuristics/astar.py`)
   - Optimal pathfinding for vehicle movements
   - Sophisticated urgency-based task prioritization
   - Direct retrieval optimization

2. **VRP Solver** (`src/heuristics/twvrp_ortools.py`)
   - Time-windowed vehicle routing optimization
   - Source blocking prevention
   - Constraint violation handling

3. **Instance Generation** (`examples/layout_generator.py`)
   - Automated test case generation
   - Configurable warehouse layouts
   - Unit load scheduling

4. **Validation System** (`src/test_cases/writer_functions.py`)
   - Solution verification
   - Constraint checking
   - Performance comparison with Gurobi IP solver

### Key Features

- **Multi-objective optimization**: Minimizes total cost while respecting time windows
- **Constraint handling**: Source blocking prevention, vehicle capacity limits
- **Performance optimization**: O(1) heuristics, efficient data structures
- **Comprehensive validation**: Automated solution checking and comparison

## 🛠️ System Components

### 1. Heuristic Solver (`run_heuristic_BRR.py`)

Fast near-optimal solutions using A* search and vehicle routing optimization.

**Features:**
- **Comparison with Gurobi results** when available
- **Batch processing** for multiple instances
- **Auto-solve mode** to process all instances with Gurobi results

**Usage:**
```bash
# Solve a single instance
python run_heuristic_BRR.py --instance experiments/inputsBRR/.../instance.json

# Solve all instances in a directory
python run_heuristic_BRR.py --directory experiments/inputsBRR/Size_3x3_Layout_1x1/

# Auto-solve all instances that have Gurobi results but no heuristic results
python run_heuristic_BRR.py --auto-solve

# With custom time limits
python run_heuristic_BRR.py --instance instance.json --astar-time-limit 600 --vrp-time-limit 300

# Compare with specific Gurobi result
python run_heuristic_BRR.py --instance instance.json --gurobi-result gurobi_result.json

# Disable automatic visualization (enabled by default)
python run_heuristic_BRR.py --instance instance.json --no-visualize
```

**Output:**
- Creates `*_heuristic.json` files with A* moves and VRP assignments
- Includes comparison metrics when Gurobi results are available
- **Automatically creates visualizations** after solving (can be disabled with `--no-visualize`)

#### TWVRP Solver Options

The heuristic solver includes two different Time-Window VRP implementations:

**PyVRP-based Solver** (`twvrp_pyvrp.py`):
- Uses the PyVRP library for vehicle routing
- Generally faster for larger problems
- Good optimization quality
- Default solver

**OR-Tools-based Solver** (`twvrp_ortools.py`):
- Uses Google OR-Tools constraint solver
- More flexible constraint modeling
- Better for complex time window constraints
- Alternative implementation

**Usage:**
```python
# Use PyVRP solver (default)
from src.heuristics.twvrp import solve_twvrp_with_pyvrp
solution = solve_twvrp_with_pyvrp(warehouse, moves, num_vehicles)

# Use OR-Tools solver
from src.heuristics.twvrp import solve_twvrp_with_ortools
solution = solve_twvrp_with_ortools(warehouse, moves, num_vehicles)

# Unified interface (specify solver)
from src.heuristics.twvrp import solve_twvrp
solution = solve_twvrp(warehouse, moves, num_vehicles, solver='pyvrp')
solution = solve_twvrp(warehouse, moves, num_vehicles, solver='ortools')
```

### 2. Result Validator (`result_checker_BRR.py`)

Validates solution files to ensure feasibility and correctness.

**Features:**
- **Constraint validation** for warehouse operations
- **Decision variable verification**
- **Interactive decision editing** for manual corrections
- **Automatic instance file detection**

**Usage:**
```bash
# Validate a solution file
python result_checker_BRR.py --solution_path experiments/resultsBRR/.../solution.json

# Validate heuristic results
python result_checker_BRR.py --solution_path experiments/resultsBRR/.../solution_heuristic.json
```

**Interactive Features:**
- View and modify decision sequences
- Add, remove, or edit individual decisions
- Real-time validation feedback

### 3. Visualization Tools

This repository provides two complementary visualization scripts plus automatic visualization.

#### Main Visualization Scripts

1. **`visualize_BRR_steps.py`** - Timestep visualization
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

The `src.visualization.auto_visualize` module provides automatic, multiprocessing-enabled visualization:

**Features:**
- **Multiprocessing support** for concurrent visualization creation
- **Automatic script detection** and appropriate visualization selection
- **Background processing** to avoid blocking main computations
- **Overwrite protection** with configurable behavior
- **Batch processing** capabilities for multiple result files

**Integration:**
- **Automatically used** by `run_heuristic_BRR.py` after solving
- **Configurable** via command-line arguments (`--auto-visualize`, `--no-visualize`)
- **Efficient** for large experiment batches with parallel processing

### Usage Examples

#### Timestep Visualization
```bash
# Basic timestep visualization
python visualize_BRR_steps.py --file <result_file.json>

# With virtual lane coloring
python visualize_BRR_steps.py --file <result_file.json> --color

# Create subplot for specific timesteps
python visualize_BRR_steps.py --file <result_file.json> --subplot-timesteps "0,25,50"
```

#### A* Move Visualization
```bash
# Basic A* move visualization (with coloring by default)
python visualize_astar_moves.py --file <heuristic_result_file.json>

# Without virtual lane coloring
python visualize_astar_moves.py --file <heuristic_result_file.json> --no-color
```

#### Heatmap Utilization Analysis
These scripts generate analytical heatmaps in the `heatmaps/` directory.

```bash
# Analyze slot occupancy duration (average over an instance type)
python visualize_slot_heatmap.py --instance-type manual2

# Analyze combined aisle travel and slot occupancy
python visualize_slot_heatmap_aisles.py --instance-type manual2

# Analyze Access Point usage frequency
python visualize_ap_usage.py --instance-type manual2

# Analyze a single specific result file
python visualize_slot_heatmap.py --file <path_to_result_file.json>
```

## 🔄 Complete Workflow

Here's a typical workflow for running experiments and analyzing results:

### Step 1: Generate Instances and Run Gurobi Optimization
```bash
# Generate new instances and solve with Gurobi
python run_BRR_experiment.py
```

### Step 2: Solve with Heuristic Approach
```bash
# Auto-solve all instances that have Gurobi results
python run_heuristic_BRR.py --auto-solve --verbose

# Or solve specific instances
python run_heuristic_BRR.py --directory experiments/inputsBRR/Size_3x3_Layout_1x1_sink_source/
```

### Step 3: Validate Results
```bash
# Validate Gurobi results
python result_checker_BRR.py --solution_path experiments/resultsBRR/.../1.json

# Validate heuristic results  
python result_checker_BRR.py --solution_path experiments/resultsBRR/.../1_heuristic.json
```

### Step 4: Visualize Solutions
```bash
# Visualize timestep evolution (Gurobi results)
python visualize_BRR_steps.py --file experiments/resultsBRR/.../1.json

# Visualize A* moves (heuristic results)
python visualize_astar_moves.py --file experiments/resultsBRR/.../1_heuristic.json

# Create timestep visualization of heuristic results
python visualize_BRR_steps.py --file experiments/resultsBRR/.../1_heuristic.json --color
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

### Installation
```bash
# Install required packages
pip install numpy matplotlib imageio gurobipy pandas seaborn

# Or use the requirements file
pip install -r requirements.txt
```

### Prerequisites
- Python 3.8+
- Gurobi Optimizer (with valid license for exact optimization)
- Required Python packages: `numpy`, `matplotlib`, `imageio`, `argparse`, `gurobipy`, `pandas`, `seaborn`

### Gurobi Setup
Ensure Gurobi is properly installed and licensed on your system for the exact optimization solver.

## 📁 Directory Structure

The system organizes experiments in a structured hierarchy:

```
buffer_reshuffling_and_retrieval_ip/
├── src/                                # Source code
│   ├── heuristics/                     # A* search and VRP algorithms
│   ├── test_cases/                     # Validation functions
│   └── visualization/                  # Auto-visualization modules
├── examples/                           # Layout generators and sample instances
├── experiments/
│   ├── inputsBRR/                      # Instance files
│   │   └── Size_3x3_Layout_1x1_sink_source/
│   │       └── handling_time_1/
│   │           └── unit_loads_3/
│   │               └── access_directions_1111/
│   │                   └── rs_max_100/
│   │                       └── as_max_50/
│   │                           └── tw_length_30/
│   │                               └── speed_1/
│   │                                   └── instance.json
│   ├── resultsBRR/                     # Solution files  
│   │   └── [same structure]/
│   │       └── fleet_size_1/
│   │           ├── 1.json              # Gurobi result
│   │           ├── 1_heuristic.json    # Heuristic result
│   │           ├── 1/                  # Timestep visualization
│   │           └── 1_heuristic_astar/  # A* move visualization
│   ├── hashesBRR/                      # Hash tracking
│   └── feasibleBRR/                    # Feasibility information
├── run_BRR_experiment.py              # Main experiment runner
├── run_heuristic_BRR.py               # Heuristic solver
├── result_checker_BRR.py              # Solution validator
├── visualize_BRR_steps.py             # Timestep visualization
├── visualize_astar_moves.py           # A* move visualization
├── evaluate_experiments.py            # Experiment analysis (see below)
└── README.md                          # This file
```

## 🤝 Contributing

The system is designed for extensibility:
- Add new heuristics in `src/heuristics/`
- Extend validation in `src/test_cases/`
- Create new instance generators in `examples/`
- Enhance visualization in `src/visualization/`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

## 📊 Experiment Analysis

For researchers and developers who want to analyze experimental results, this repository includes comprehensive evaluation capabilities.

### Experiment Evaluation Script

The `evaluate_experiments.py` script provides analysis of both IP (exact) and heuristic approaches across all generated instances.

#### Quick Start - Experiment Analysis

```bash
# Analyze all experiments with visualizations
python evaluate_experiments.py

# Filter by fleet size
python evaluate_experiments.py --fleet 3

# Quick summary only
python evaluate_experiments.py --summary-only
```

#### Usage Examples

```bash
# Basic usage
python evaluate_experiments.py                    # Analyze all experiments
python evaluate_experiments.py --fleet 3          # Only fleet size 3  
python evaluate_experiments.py --fleet 1 2        # Fleet sizes 1 and 2

# Output control
python evaluate_experiments.py --no-plots         # Skip visualization generation
python evaluate_experiments.py --summary-only     # Just print summary, no files
python evaluate_experiments.py --no-csv          # Skip CSV export

# Advanced
python evaluate_experiments.py --base_path /custom/path  # Custom experiment directory
```

#### Command Line Options

- `--fleet SIZE [SIZE ...]` - Filter by fleet size(s)
- `--no-plots` - Skip plot generation (faster execution)
- `--no-csv` - Skip CSV export
- `--summary-only` - Only print summary (no CSV or plots)
- `--base_path PATH` - Base path to experiment directory (default: current directory)

### Output Files

#### 1. Console Summary
Real-time analysis with key metrics:
```
📊 OVERALL STATISTICS:
   Total unique instances: 771
   IP solved: 607 (78.7%)
   Heuristic solved: 164 (21.3%)
   Both approaches solved: 0 (0.0%)

🚛 FLEET SIZE ANALYSIS:
   Fleet size 1: 108 instances, IP: 94.4%, Heuristic: 5.6%
   Fleet size 2: 330 instances, IP: 90.0%, Heuristic: 10.0%
   Fleet size 3: 332 instances, IP: 62.3%, Heuristic: 37.7%
```

#### 2. CSV Export: `detailed_results.csv`
Comprehensive data for each instance with columns:
- **Instance Parameters**: `warehouse_size`, `unit_loads`, `fleet_size`, etc.
- **IP Results**: `ip_execution_time`, `ip_objective`, `ip_status`, `ip_gap`
- **Heuristic Results**: `heuristic_execution_time`, `heuristic_objective`, `heuristic_violations`, `heuristic_tardiness`
- **Comparison**: `optimality_gap` (heuristic vs IP performance)

#### 3. Visualizations: `experiment_analysis/`
- `solve_rates.png` - Success rates for IP vs Heuristic
- `execution_times.png` - Runtime comparison (log scale)
- `optimality_gaps.png` - Quality comparison distribution
- `performance_by_fleet_size.png` - Analysis by fleet size
- `performance_by_unit_loads.png` - Analysis by number of unit loads

### Key Metrics Explained

#### Solve Rates
- **IP Solve Rate**: Percentage of instances solved by exact method
- **Heuristic Solve Rate**: Percentage solved by heuristic approach
- **Both Solved**: Instances solved by both approaches (enables direct comparison)

#### Performance Analysis
- **Execution Time**: Runtime comparison between approaches
- **Optimality Gap**: `(heuristic_obj - ip_obj) / ip_obj * 100`
- **Success by Parameters**: Performance broken down by fleet size, unit loads, warehouse size, layout type

#### Understanding Results

**Missing "Both Solved" Instances**
If "Both approaches solved: 0", this indicates:
- IP and heuristic were run on different instance sets
- Different experimental conditions or time limits
- This is normal for large-scale experimentation

**Performance Insights**
- **IP** provides optimal solutions but with longer runtimes
- **Heuristic** offers faster solutions with potential optimality gaps
- Fleet size significantly impacts solvability (smaller fleets → easier problems)
- Warehouse size affects complexity (larger warehouses → more challenging)

### Expected File Structure for Analysis

```
buffer_reshuffling_and_retrieval_ip/
├── experiments/
│   └── resultsBRR/
│       ├── Size_3x3_Layout_1x1_*/
│       │   ├── *_heuristic.json  # Heuristic results
│       │   └── *.json            # IP results (without _heuristic suffix)
│       └── Size_4x4_Layout_*/
│           └── ...
├── src/                          # Source code
├── examples/                     # Layout examples
├── evaluate_experiments.py      # Main analysis script
└── README.md                    # This file
```

### Analysis Tips

1. **Filter by fleet size** for focused analysis of specific scenarios
2. **Use CSV export** for detailed statistical analysis in Excel/R/Python
3. **Check visualizations** for performance trends and outliers
4. **Compare execution times** to understand efficiency trade-offs
5. **Analyze optimality gaps** where both approaches solved same instances
6. **Group by parameters** to identify which problem characteristics affect solvability
