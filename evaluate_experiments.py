#!/usr/bin/env python3
"""
Experiment Evaluation Script for Buffer Reshuffling and Retrieval

This script analyzes the performance of both IP (exact) and heuristic approaches
across all generated instances, providing comprehensive statistics and comparisons.

Usage Examples:
    python evaluate_experiments.py                    # Analyze all experiments
    python evaluate_experiments.py --fleet 3          # Only fleet size 3
    python evaluate_experiments.py --fleet 1 2        # Fleet sizes 1 and 2
    python evaluate_experiments.py --no-plots         # Skip visualization generation
    python evaluate_experiments.py --summary-only     # Just print summary, no files
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Tuple, Any
import re

class ExperimentEvaluator:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.results_base = self.base_path / "experiments" / "resultsBRR"
        self.hashes_base = self.base_path / "experiments" / "hashesBRR"
        
        # Initialize data containers
        self.ip_results = []
        self.heuristic_results = []
        self.instance_metadata = {}
        
    def parse_instance_path(self, filepath: str) -> Dict[str, Any]:
        """Extract instance parameters from the file path."""
        # Example path: Size_3x3_Layout_1x1_sink_source/handling_time_1/unit_loads_6/access_directions_0010/rs_max_200/as_max_150/tw_length_30/speed_1/fleet_size_3/7_heuristic.json
        parts = Path(filepath).parts
        
        params = {}
        for part in parts:
            if part.startswith("Size_"):
                # Extract warehouse size
                size_match = re.search(r"Size_(\d+)x(\d+)_Layout_(\d+)x(\d+)", part)
                if size_match:
                    params["warehouse_size"] = f"{size_match.group(1)}x{size_match.group(2)}"
                    params["layout_size"] = f"{size_match.group(3)}x{size_match.group(4)}"
                
                # Extract layout type
                if "sink_source" in part:
                    params["layout_type"] = "sink_source"
                elif "sink" in part:
                    params["layout_type"] = "sink"
                elif "wide" in part:
                    params["layout_type"] = "wide"
                else:
                    params["layout_type"] = "standard"
                    
            elif part.startswith("handling_time_"):
                params["handling_time"] = int(part.split("_")[-1])
            elif part.startswith("unit_loads_"):
                params["unit_loads"] = int(part.split("_")[-1])
            elif part.startswith("access_directions_"):
                params["access_directions"] = part.split("_")[-1]
            elif part.startswith("rs_max_"):
                params["rs_max"] = int(part.split("_")[-1])
            elif part.startswith("as_max_"):
                params["as_max"] = int(part.split("_")[-1])
            elif part.startswith("tw_length_"):
                params["tw_length"] = int(part.split("_")[-1])
            elif part.startswith("speed_"):
                params["speed"] = int(part.split("_")[-1])
            elif part.startswith("fleet_size_"):
                params["fleet_size"] = int(part.split("_")[-1])
        
        # Extract instance number from filename
        filename = Path(filepath).name
        if filename.endswith("_heuristic.json"):
            instance_match = re.search(r"(\d+)_heuristic\.json", filename)
            if instance_match:
                params["instance_id"] = int(instance_match.group(1))
                params["approach"] = "heuristic"
        elif filename.endswith(".json") and not filename.endswith("_heuristic.json"):
            instance_match = re.search(r"(\d+)\.json", filename)
            if instance_match:
                params["instance_id"] = int(instance_match.group(1))
                params["approach"] = "ip"
        
        return params
    
    def load_result_file(self, filepath: Path) -> Dict[str, Any]:
        """Load and parse a result JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def collect_all_results(self, fleet_size_filter: List[int] = None):
        """Collect all IP and heuristic results."""
        print("Collecting experiment results...")
        
        # Collect all result files
        if self.results_base.exists():
            for result_file in self.results_base.rglob("*.json"):
                rel_path = result_file.relative_to(self.results_base)
                params = self.parse_instance_path(str(rel_path))
                
                # Apply fleet size filter if specified
                if fleet_size_filter and params.get("fleet_size") not in fleet_size_filter:
                    continue
                
                # Load result data
                result_data = self.load_result_file(result_file)
                if result_data is None:
                    continue
                
                # Combine metadata with results
                combined_data = {**params, **result_data}
                combined_data["filepath"] = str(result_file)
                
                # Separate by approach
                if params.get("approach") == "ip":
                    self.ip_results.append(combined_data)
                elif params.get("approach") == "heuristic":
                    self.heuristic_results.append(combined_data)
        
        print(f"Loaded {len(self.ip_results)} IP results and {len(self.heuristic_results)} heuristic results")
    
    def identify_unique_instances(self) -> Dict[str, Dict]:
        """Identify unique problem instances based on parameters."""
        unique_instances = {}
        
        for result in self.ip_results + self.heuristic_results:
            # Create instance key (exclude approach and filepath)
            key_params = {k: v for k, v in result.items() 
                         if k not in ["approach", "filepath", "execution_time", "objective_value", 
                                    "status", "gap", "nodes_explored", "violations", "tardiness_info",
                                    "total_tardiness", "num_late_unit_loads", "num_late_moves", 
                                    "decisions_count", "decisions", "vehicles", "message"]}
            
            # Create a hashable key from sorted items
            key_items = []
            for k, v in sorted(key_params.items()):
                if isinstance(v, (dict, list)):
                    # Convert unhashable types to strings
                    key_items.append((k, str(v)))
                else:
                    key_items.append((k, v))
            
            instance_key = tuple(key_items)
            
            if instance_key not in unique_instances:
                unique_instances[instance_key] = {
                    "params": key_params,
                    "ip_solved": False,
                    "heuristic_solved": False,
                    "ip_data": None,
                    "heuristic_data": None
                }
            
            # Update with approach-specific data
            if result.get("approach") == "ip":
                unique_instances[instance_key]["ip_solved"] = True
                unique_instances[instance_key]["ip_data"] = result
            elif result.get("approach") == "heuristic":
                unique_instances[instance_key]["heuristic_solved"] = True
                unique_instances[instance_key]["heuristic_data"] = result
        
        return unique_instances
    
    def analyze_solve_rates(self, unique_instances: Dict) -> Dict[str, Any]:
        """Analyze solving success rates for both approaches."""
        total_instances = len(unique_instances)
        ip_solved = sum(1 for inst in unique_instances.values() if inst["ip_solved"])
        heuristic_solved = sum(1 for inst in unique_instances.values() if inst["heuristic_solved"])
        both_solved = sum(1 for inst in unique_instances.values() 
                         if inst["ip_solved"] and inst["heuristic_solved"])
        
        stats = {
            "total_instances": total_instances,
            "ip_solved": ip_solved,
            "heuristic_solved": heuristic_solved,
            "both_solved": both_solved,
            "ip_solve_rate": ip_solved / total_instances if total_instances > 0 else 0,
            "heuristic_solve_rate": heuristic_solved / total_instances if total_instances > 0 else 0,
            "both_solve_rate": both_solved / total_instances if total_instances > 0 else 0
        }
        
        return stats
    
    def analyze_performance(self, unique_instances: Dict) -> Dict[str, Any]:
        """Analyze performance metrics for solved instances."""
        ip_times = []
        heuristic_times = []
        ip_objectives = []
        heuristic_objectives = []
        gaps = []
        
        for instance in unique_instances.values():
            if instance["ip_solved"] and instance["ip_data"]:
                ip_data = instance["ip_data"]
                if "execution_time" in ip_data:
                    ip_times.append(ip_data["execution_time"])
                if "objective_value" in ip_data:
                    ip_objectives.append(ip_data["objective_value"])
            
            if instance["heuristic_solved"] and instance["heuristic_data"]:
                heur_data = instance["heuristic_data"]
                if "execution_time" in heur_data:
                    heuristic_times.append(heur_data["execution_time"])
                if "objective_value" in heur_data:
                    heuristic_objectives.append(heur_data["objective_value"])
            
            # Calculate gap for instances solved by both
            if (instance["ip_solved"] and instance["heuristic_solved"] and 
                instance["ip_data"] and instance["heuristic_data"]):
                ip_obj = instance["ip_data"].get("objective_value")
                heur_obj = instance["heuristic_data"].get("objective_value")
                
                if ip_obj is not None and heur_obj is not None and ip_obj > 0:
                    gap = ((heur_obj - ip_obj) / ip_obj) * 100
                    gaps.append(gap)
        
        stats = {
            "ip_execution_times": ip_times,
            "heuristic_execution_times": heuristic_times,
            "ip_objectives": ip_objectives,
            "heuristic_objectives": heuristic_objectives,
            "optimality_gaps": gaps
        }
        
        # Calculate summary statistics
        for key in ["ip_execution_times", "heuristic_execution_times", "optimality_gaps"]:
            if stats[key]:
                values = stats[key]
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_median"] = np.median(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_min"] = np.min(values)
                stats[f"{key}_max"] = np.max(values)
        
        return stats
    
    def analyze_by_parameters(self, unique_instances: Dict) -> Dict[str, Any]:
        """Analyze performance grouped by problem parameters."""
        # Group by different parameters
        param_analysis = {}
        
        # Group by fleet size
        fleet_size_groups = defaultdict(list)
        for instance in unique_instances.values():
            fleet_size = instance["params"].get("fleet_size")
            if fleet_size and not isinstance(fleet_size, (list, dict)):
                fleet_size_groups[fleet_size].append(instance)
        
        param_analysis["fleet_size"] = {}
        for fleet_size, instances in fleet_size_groups.items():
            ip_solved = sum(1 for inst in instances if inst["ip_solved"])
            heur_solved = sum(1 for inst in instances if inst["heuristic_solved"])
            param_analysis["fleet_size"][fleet_size] = {
                "total": len(instances),
                "ip_solved": ip_solved,
                "heuristic_solved": heur_solved,
                "ip_rate": ip_solved / len(instances),
                "heuristic_rate": heur_solved / len(instances)
            }
        
        # Group by unit loads
        ul_groups = defaultdict(list)
        for instance in unique_instances.values():
            unit_loads = instance["params"].get("unit_loads")
            if unit_loads and not isinstance(unit_loads, (list, dict)):
                ul_groups[unit_loads].append(instance)
        
        param_analysis["unit_loads"] = {}
        for ul_count, instances in ul_groups.items():
            ip_solved = sum(1 for inst in instances if inst["ip_solved"])
            heur_solved = sum(1 for inst in instances if inst["heuristic_solved"])
            param_analysis["unit_loads"][ul_count] = {
                "total": len(instances),
                "ip_solved": ip_solved,
                "heuristic_solved": heur_solved,
                "ip_rate": ip_solved / len(instances),
                "heuristic_rate": heur_solved / len(instances)
            }
        
        # Group by warehouse size
        size_groups = defaultdict(list)
        for instance in unique_instances.values():
            warehouse_size = instance["params"].get("warehouse_size")
            if warehouse_size and not isinstance(warehouse_size, (list, dict)):
                size_groups[warehouse_size].append(instance)
        
        param_analysis["warehouse_size"] = {}
        for size, instances in size_groups.items():
            ip_solved = sum(1 for inst in instances if inst["ip_solved"])
            heur_solved = sum(1 for inst in instances if inst["heuristic_solved"])
            param_analysis["warehouse_size"][size] = {
                "total": len(instances),
                "ip_solved": ip_solved,
                "heuristic_solved": heur_solved,
                "ip_rate": ip_solved / len(instances),
                "heuristic_rate": heur_solved / len(instances)
            }
        
        return param_analysis
    
    def create_visualizations(self, unique_instances: Dict, performance_stats: Dict, 
                            param_analysis: Dict, output_dir: str = "experiment_analysis"):
        """Create visualization plots for the analysis."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        try:
            sns.set_palette("husl")
        except:
            # Fallback if seaborn style doesn't work
            pass
        
        # 1. Solve rates comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        solve_stats = self.analyze_solve_rates(unique_instances)
        
        approaches = ['IP', 'Heuristic', 'Both']
        rates = [solve_stats['ip_solve_rate'], solve_stats['heuristic_solve_rate'], 
                solve_stats['both_solve_rate']]
        counts = [solve_stats['ip_solved'], solve_stats['heuristic_solved'], 
                 solve_stats['both_solved']]
        
        bars = ax.bar(approaches, rates, alpha=0.7)
        ax.set_ylabel('Solve Rate')
        ax.set_title(f'Instance Solve Rates (Total: {solve_stats["total_instances"]} instances)')
        ax.set_ylim(0, 1)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/solve_rates.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Execution time comparison
        if performance_stats.get("ip_execution_times") and performance_stats.get("heuristic_execution_times"):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            data_to_plot = [performance_stats["ip_execution_times"], 
                           performance_stats["heuristic_execution_times"]]
            labels = ['IP', 'Heuristic']
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Execution Time Comparison')
            ax.set_yscale('log')  # Log scale for better visualization
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/execution_times.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Optimality gap distribution
        if performance_stats.get("optimality_gaps"):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            gaps = performance_stats["optimality_gaps"]
            ax.hist(gaps, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Optimality Gap (%)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Optimality Gap Distribution (n={len(gaps)})')
            ax.axvline(np.mean(gaps), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(gaps):.2f}%')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/optimality_gaps.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Performance by fleet size
        if "fleet_size" in param_analysis:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            fleet_sizes = sorted(param_analysis["fleet_size"].keys())
            ip_rates = [param_analysis["fleet_size"][fs]["ip_rate"] for fs in fleet_sizes]
            heur_rates = [param_analysis["fleet_size"][fs]["heuristic_rate"] for fs in fleet_sizes]
            
            x = np.arange(len(fleet_sizes))
            width = 0.35
            
            ax1.bar(x - width/2, ip_rates, width, label='IP', alpha=0.7)
            ax1.bar(x + width/2, heur_rates, width, label='Heuristic', alpha=0.7)
            ax1.set_xlabel('Fleet Size')
            ax1.set_ylabel('Solve Rate')
            ax1.set_title('Solve Rates by Fleet Size')
            ax1.set_xticks(x)
            ax1.set_xticklabels(fleet_sizes)
            ax1.legend()
            ax1.set_ylim(0, 1)
            
            # Instance counts
            totals = [param_analysis["fleet_size"][fs]["total"] for fs in fleet_sizes]
            ax2.bar(fleet_sizes, totals, alpha=0.7)
            ax2.set_xlabel('Fleet Size')
            ax2.set_ylabel('Number of Instances')
            ax2.set_title('Instance Count by Fleet Size')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_by_fleet_size.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Performance by unit loads
        if "unit_loads" in param_analysis:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ul_counts = sorted(param_analysis["unit_loads"].keys())
            ip_rates = [param_analysis["unit_loads"][ul]["ip_rate"] for ul in ul_counts]
            heur_rates = [param_analysis["unit_loads"][ul]["heuristic_rate"] for ul in ul_counts]
            
            x = np.arange(len(ul_counts))
            width = 0.35
            
            ax.bar(x - width/2, ip_rates, width, label='IP', alpha=0.7)
            ax.bar(x + width/2, heur_rates, width, label='Heuristic', alpha=0.7)
            ax.set_xlabel('Number of Unit Loads')
            ax.set_ylabel('Solve Rate')
            ax.set_title('Solve Rates by Number of Unit Loads')
            ax.set_xticks(x)
            ax.set_xticklabels(ul_counts)
            ax.legend()
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_by_unit_loads.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")
    
    def print_summary_report(self, unique_instances: Dict, performance_stats: Dict, 
                           param_analysis: Dict):
        """Print a comprehensive summary report."""
        print("\n" + "="*80)
        print("EXPERIMENT EVALUATION SUMMARY REPORT")
        print("="*80)
        
        # Overall statistics
        solve_stats = self.analyze_solve_rates(unique_instances)
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total unique instances: {solve_stats['total_instances']}")
        print(f"   IP solved: {solve_stats['ip_solved']} ({solve_stats['ip_solve_rate']:.1%})")
        print(f"   Heuristic solved: {solve_stats['heuristic_solved']} ({solve_stats['heuristic_solve_rate']:.1%})")
        print(f"   Both approaches solved: {solve_stats['both_solved']} ({solve_stats['both_solve_rate']:.1%})")
        
        # Performance comparison
        if performance_stats.get("ip_execution_times") and performance_stats.get("heuristic_execution_times"):
            print(f"\n⏱️  EXECUTION TIME COMPARISON:")
            print(f"   IP average time: {performance_stats.get('ip_execution_times_mean', 0):.2f}s")
            print(f"   Heuristic average time: {performance_stats.get('heuristic_execution_times_mean', 0):.2f}s")
            
            if performance_stats.get('ip_execution_times_mean', 0) > 0:
                speedup = performance_stats.get('ip_execution_times_mean', 0) / performance_stats.get('heuristic_execution_times_mean', 1)
                print(f"   Heuristic speedup: {speedup:.1f}x faster")
        
        # Optimality gaps
        if performance_stats.get("optimality_gaps"):
            print(f"\n🎯 OPTIMALITY ANALYSIS:")
            print(f"   Instances compared: {len(performance_stats['optimality_gaps'])}")
            print(f"   Average gap: {performance_stats.get('optimality_gaps_mean', 0):.2f}%")
            print(f"   Median gap: {performance_stats.get('optimality_gaps_median', 0):.2f}%")
            print(f"   Best case: {performance_stats.get('optimality_gaps_min', 0):.2f}%")
            print(f"   Worst case: {performance_stats.get('optimality_gaps_max', 0):.2f}%")
        
        # Fleet size analysis
        if "fleet_size" in param_analysis:
            print(f"\n🚛 FLEET SIZE ANALYSIS:")
            for fs in sorted(param_analysis["fleet_size"].keys()):
                data = param_analysis["fleet_size"][fs]
                print(f"   Fleet size {fs}: {data['total']} instances, "
                      f"IP: {data['ip_rate']:.1%}, Heuristic: {data['heuristic_rate']:.1%}")
        
        # Unit loads analysis
        if "unit_loads" in param_analysis:
            print(f"\n📦 UNIT LOADS ANALYSIS:")
            for ul in sorted(param_analysis["unit_loads"].keys()):
                data = param_analysis["unit_loads"][ul]
                print(f"   {ul} unit loads: {data['total']} instances, "
                      f"IP: {data['ip_rate']:.1%}, Heuristic: {data['heuristic_rate']:.1%}")
        
        # Warehouse size analysis
        if "warehouse_size" in param_analysis:
            print(f"\n🏗️  WAREHOUSE SIZE ANALYSIS:")
            for size in sorted(param_analysis["warehouse_size"].keys()):
                data = param_analysis["warehouse_size"][size]
                print(f"   {size}: {data['total']} instances, "
                      f"IP: {data['ip_rate']:.1%}, Heuristic: {data['heuristic_rate']:.1%}")
        
        print("\n" + "="*80)
    
    def save_detailed_results(self, unique_instances: Dict, output_file: str = "detailed_results.csv"):
        """Save detailed results to CSV for further analysis."""
        rows = []
        
        for instance in unique_instances.values():
            row = instance["params"].copy()
            
            # Add IP results
            if instance["ip_solved"] and instance["ip_data"]:
                ip_data = instance["ip_data"]
                row["ip_solved"] = True
                row["ip_execution_time"] = ip_data.get("execution_time")
                row["ip_objective"] = ip_data.get("objective_value")
                row["ip_status"] = ip_data.get("status")
                row["ip_gap"] = ip_data.get("gap")
            else:
                row["ip_solved"] = False
                row["ip_execution_time"] = None
                row["ip_objective"] = None
                row["ip_status"] = None
                row["ip_gap"] = None
            
            # Add heuristic results
            if instance["heuristic_solved"] and instance["heuristic_data"]:
                heur_data = instance["heuristic_data"]
                row["heuristic_solved"] = True
                row["heuristic_execution_time"] = heur_data.get("execution_time")
                row["heuristic_objective"] = heur_data.get("objective_value")
                row["heuristic_violations"] = len(heur_data.get("violations", []))
                row["heuristic_tardiness"] = heur_data.get("total_tardiness", 0)
            else:
                row["heuristic_solved"] = False
                row["heuristic_execution_time"] = None
                row["heuristic_objective"] = None
                row["heuristic_violations"] = None
                row["heuristic_tardiness"] = None
            
            # Calculate optimality gap
            if (row["ip_objective"] is not None and row["heuristic_objective"] is not None 
                and row["ip_objective"] > 0):
                row["optimality_gap"] = ((row["heuristic_objective"] - row["ip_objective"]) 
                                       / row["ip_objective"]) * 100
            else:
                row["optimality_gap"] = None
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"Detailed results saved to {output_file}")
        
        return df
    
    def run_evaluation(self, fleet_size_filter: List[int] = None, 
                      create_plots: bool = True, save_csv: bool = True):
        """Run the complete evaluation process."""
        print("Starting experiment evaluation...")
        
        # Collect all results
        self.collect_all_results(fleet_size_filter)
        
        # Identify unique instances
        unique_instances = self.identify_unique_instances()
        print(f"Identified {len(unique_instances)} unique problem instances")
        
        # Analyze performance
        performance_stats = self.analyze_performance(unique_instances)
        param_analysis = self.analyze_by_parameters(unique_instances)
        
        # Print summary report
        self.print_summary_report(unique_instances, performance_stats, param_analysis)
        
        # Create visualizations
        if create_plots:
            self.create_visualizations(unique_instances, performance_stats, param_analysis)
        
        # Save detailed results
        if save_csv:
            df = self.save_detailed_results(unique_instances)
            return df
        
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Buffer Reshuffling and Retrieval experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_experiments.py                    # Analyze all experiments
    python evaluate_experiments.py --fleet 3          # Only fleet size 3
    python evaluate_experiments.py --fleet 1 2        # Fleet sizes 1 and 2
    python evaluate_experiments.py --no-plots         # Skip visualization generation
    python evaluate_experiments.py --summary-only     # Just print summary, no files
        """
    )
    
    parser.add_argument("--base_path", default=".", 
                       help="Base path to the experiment directory")
    parser.add_argument("--fleet", "--fleet_size", nargs="+", type=int, 
                       metavar="SIZE", dest="fleet_size",
                       help="Filter by fleet size(s) (e.g., --fleet 3 or --fleet 1 2)")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip plot generation")
    parser.add_argument("--no-csv", action="store_true", 
                       help="Skip CSV export")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary (no CSV or plots)")
    
    args = parser.parse_args()
    
    # User-friendly header
    print("🔍 Analyzing Buffer Reshuffling and Retrieval Experiments")
    print("=" * 60)
    
    if args.fleet_size:
        print(f"📊 Fleet size filter: {args.fleet_size}")
    else:
        print("📊 Analyzing all fleet sizes")
    
    evaluator = ExperimentEvaluator(args.base_path)
    
    # Override args if summary-only is requested
    create_plots = not args.no_plots and not args.summary_only
    save_csv = not args.no_csv and not args.summary_only
    
    df = evaluator.run_evaluation(
        fleet_size_filter=args.fleet_size,
        create_plots=create_plots,
        save_csv=save_csv
    )
    
    # User-friendly completion message
    if args.summary_only:
        print("\n✅ Summary analysis complete!")
    else:
        if df is not None:
            print(f"\n✅ Analysis complete! Generated:")
            print(f"   📋 detailed_results.csv ({df.shape[0]} instances)")
            if create_plots:
                print(f"   📊 Visualizations in experiment_analysis/")
        
        print(f"\n💡 Tip: Open detailed_results.csv in Excel/LibreOffice for detailed analysis")


if __name__ == "__main__":
    main()
