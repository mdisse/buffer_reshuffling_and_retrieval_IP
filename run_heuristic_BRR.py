import argparse
import os
import sys
import glob
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.instance.instance_loader import InstanceLoader
from src.instance.instance import Instance
from src.test_cases.test_case_brr import TestCaseBrr
from src.test_cases.writer_functions import generate_heuristic_result_path
from src.heuristics.map_tw_prio import create_task_queue 
from src.visualization.auto_visualize import auto_visualize 

def find_solved_gurobi_instances(experiments_dir="experiments"):
    """
    Find all instances that have been solved by Gurobi by looking for result files.
    Returns a list of (instance_path, result_path) tuples.
    """
    solved_instances = []
    
    # Look for result files in resultsBRR directory
    results_pattern = os.path.join(experiments_dir, "resultsBRR", "**", "*.json")
    result_files = glob.glob(results_pattern, recursive=True)
    
    for result_file in result_files:
        # Skip heuristic result files
        if '_heuristic.json' in result_file:
            continue
            
        # Convert result path to corresponding instance path
        # Simply replace 'resultsBRR' with 'inputsBRR' - both have fleet_size directories
        instance_file = result_file.replace('resultsBRR', 'inputsBRR')
        
        # Check if the instance file exists
        if os.path.exists(instance_file):
            # Verify it's a valid Gurobi result by checking if it contains solver output
            try:
                with open(result_file, 'r') as f:
                    content = f.read()
                    # Check if it's a proper result file (not just an instance copy)
                    if 'results' in content or 'objective' in content or 'runtime' in content:
                        solved_instances.append((instance_file, result_file))
            except:
                continue

    return solved_instances


def find_instances_without_heuristic_results(experiments_dir="experiments"):
    """
    Find all solved Gurobi instances that don't have corresponding heuristic results.
    Returns a list of (instance_path, fleet_size) tuples that need heuristic solving.
    """
    solved_instances = find_solved_gurobi_instances(experiments_dir)
    instances_needing_heuristic = []
    
    for instance_file, result_file in solved_instances:
        # Generate expected heuristic result filename
        heuristic_result_file = result_file.replace('.json', '_heuristic.json')
        
        # Check if heuristic result doesn't exist
        if not os.path.exists(heuristic_result_file):
            # Extract fleet size from result path
            fleet_size = 1  # default
            if 'fleet_size_' in result_file:
                try:
                    fleet_size = int(result_file.split('fleet_size_')[1].split('/')[0])
                except:
                    fleet_size = 1
            
            instances_needing_heuristic.append((instance_file, fleet_size))
    
    return instances_needing_heuristic


def solve_instance(instance, verbose, instance_file_path, gurobi_result_path=None, astar_time_limit=None, vrp_time_limit=None, enable_visualization=True, vrp_solver='scheduling', overwrite=False, **kwargs): 
    """
    Solve the given instance using the BRR heuristic with optional comparison.
    """
    # Check if result checks existence (if overwrite is False)
    if not overwrite:
        try:
            fleet_size = instance.get_fleet_size()
            result_path = generate_heuristic_result_path(instance_file_path, fleet_size)
            if os.path.exists(result_path):
                msg = f"Skipping {os.path.basename(instance_file_path)} (result exists)"
                if verbose:
                    print(msg + f": {result_path}")
                else:
                    print(msg)
                
                return {
                    'instance': os.path.basename(instance_file_path),
                    'heuristic_feasible': False,
                    'skipped': True
                }
        except Exception as e:
            if verbose: print(f"Warning: Could not check for existing result: {e}")

    result = solve_with_comparison(instance, instance_file_path, verbose, gurobi_result_path, astar_time_limit, vrp_time_limit, vrp_solver)
    test_case = result['test_case']

    # Check if A* failed
    if result.get('astar_failed', False):
        if verbose:
            print("="*85)
            print("HEURISTIC SOLUTION")
            print("="*85)
            print("  ❌ A* failed to find a solution")
            print("  Skipping VRP, validation, and visualization")
            print(f" Instance {instance_file_path} could not be solved due to A* failure.")
        else:
            print(f"  ❌ A* failed to find a solution")
            print(f" Instance {instance_file_path} could not be solved due to A* failure.")
        return result

    # Load Gurobi result for comparison BEFORE saving (so MIP gap can be stored)
    fleet_size = instance.get_fleet_size()
    gurobi_result = None
    if gurobi_result_path:
        gurobi_result = load_gurobi_result(gurobi_result_path)
    else:
        # Try to find the corresponding result file automatically
        auto_result_path = find_gurobi_result_file(instance_file_path, fleet_size)
        if auto_result_path and verbose:
            print(f"  Found Gurobi result file: {auto_result_path}")
        gurobi_result = load_gurobi_result(auto_result_path) if auto_result_path else None
    
    # Calculate MIP gap BEFORE saving so it can be stored in the file
    if gurobi_result and gurobi_result.get('feasible'):
        gurobi_obj = gurobi_result['objective']
        heuristic_obj = test_case.heuristic_objective
        
        if gurobi_obj > 0 and heuristic_obj is not None:
             # Calculate gap as percentage
             gap = ((heuristic_obj - gurobi_obj) / gurobi_obj) * 100
             test_case.mip_gap = gap
             if verbose:
                 print(f"  Calculated comparison MIP gap before validation: {test_case.mip_gap:.2f}%")

    # If validate_gap is requested (default True), run Gurobi to calculate the true MIP gap against lower bound
    # This will overwrite the comparison gap if it was calculated above, which is what we want
    # (True MIP gap is more accurate than gap vs potentially suboptimal Gurobi solution)
    if kwargs.get('validate_gap', True):
        if verbose:
            print("  Running Gurobi to validate solution and calculate true MIP gap...")
        
        if hasattr(test_case, 'calculate_gurobi_gap'):
            # Extract the solution from the heuristic test case
            from src.test_cases.writer_functions import translate_heuristic_decisions_simple
            translated_decisions = translate_heuristic_decisions_simple(test_case.amr_assignments, test_case.instance)
            
            solution = {}
            for vehicle_decisions in translated_decisions.values():
                for decision_info in vehicle_decisions.values():
                    if 'decision' in decision_info:
                        solution[decision_info['decision']] = 1
            
            # Create validation instance
            val_test_case = TestCaseBrr(instance=instance, variant="dynamic_multiple", solution=solution, verbose=verbose, mode="validate_gap")
            
            # Calculate gap
            true_gap = val_test_case.calculate_gurobi_gap()
            
            if true_gap is not None:
                # Update the original test case's mip_gap so it gets saved
                test_case.mip_gap = true_gap * 100
                if verbose:
                    print(f"  True MIP Gap calculated: {test_case.mip_gap:.2f}%")

    # Save heuristic results, which includes running validation
    validate_results = kwargs.get('validate_gap', True)
    result_file_path = test_case.save_heuristic_results(instance_file_path, fleet_size, validate=validate_results)

    # Auto-generate visualization in background
    if enable_visualization:
        try:
            viz_process = auto_visualize(result_file_path, background=True, overwrite=True)
            if verbose and viz_process:
                print(f"🎨 Started background visualization for {os.path.basename(result_file_path)}")
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not start visualization: {e}")
    elif verbose:
        print("📊 Automatic visualization disabled")

    # Read validation results from the saved file
    validation_results = get_validation_results_from_file(instance_file_path, fleet_size)
    if verbose:
        print(f"DEBUG: validation_results type: {type(validation_results)}")
        if validation_results:
            print(f"DEBUG: is_feasible: {validation_results.get('is_feasible')}")
    
    # Now update result dict with comparison data for output
    if gurobi_result and gurobi_result.get('feasible'):
        gurobi_mipgap = gurobi_result.get('gurobi_mipgap', None)
        
        # If validation provided a better objective and mipgap, use those instead
        if validation_results and validation_results.get('validation_objective') is not None:
            validation_obj = validation_results['validation_objective']
            validation_gap = validation_results.get('validation_mipgap', None)
            if verbose:
                print(f"  Using validation results for comparison: obj={validation_obj}, gap={validation_gap}")
            
            # Use validation gap directly
            mip_gap = validation_gap * 100 if validation_gap is not None else 0.0
            
            result.update({
                'gurobi_obj': validation_obj,
                'gurobi_time': gurobi_result['runtime'],
                'gurobi_mipgap': validation_gap if validation_gap is not None else 0.0,
                'mip_gap': mip_gap,
                'speedup': gurobi_result['runtime'] / test_case.heuristic_runtime if test_case.heuristic_runtime > 0 else 0,
                'has_comparison': True,
                'used_validation': True
            })
        elif validation_results and not validation_results.get('is_feasible', True):
            # If validation failed, set MIP gap to NaN
            if verbose:
                print("  Validation failed (infeasible), setting MIP gap to NaN")
            
            mip_gap = float('nan')
            
            result.update({
                'gurobi_obj': gurobi_result['objective'],
                'gurobi_time': gurobi_result['runtime'],
                'gurobi_mipgap': gurobi_result.get('gurobi_mipgap', 'N/A'),
                'mip_gap': mip_gap,
                'speedup': gurobi_result['runtime'] / test_case.heuristic_runtime if test_case.heuristic_runtime > 0 else 0,
                'has_comparison': True,
                'used_validation': True
            })
        else:
            # Use stored Gurobi result
            heuristic_obj = test_case.heuristic_objective
            gurobi_obj = gurobi_result['objective']
            if gurobi_obj > 0:
                 mip_gap = ((heuristic_obj - gurobi_obj) / gurobi_obj) * 100
            else:
                 mip_gap = float('inf')

            speedup = gurobi_result['runtime'] / test_case.heuristic_runtime if test_case.heuristic_runtime > 0 else 0
            
            result.update({
                'gurobi_obj': gurobi_result['objective'],
                'gurobi_time': gurobi_result['runtime'],
                'gurobi_mipgap': gurobi_mipgap if gurobi_mipgap is not None else 'N/A',
                'mip_gap': mip_gap,
                'speedup': speedup,
                'has_comparison': True,
                'used_validation': False
            })
    else:
        result['has_comparison'] = False

    # Print results
    if verbose:
        print("="*85)
        print("HEURISTIC SOLUTION")
        print("="*85)
        test_case.print_heuristic_solution()

        # Show validation results
        if result['heuristic_feasible'] and validation_results:
            print("\n" + "="*85)
            print("SOLUTION VALIDATION")
            print("="*85)

            is_valid = validation_results.get('is_feasible', False)
            status_code = validation_results.get('status_code', -1)
            validation_msg = validation_results.get('message', 'Unknown')
            violations = validation_results.get('violations', [])

            if is_valid:
                print(f"✅ Solution validation: {validation_msg}")
            else:
                print(f"❌ Solution validation failed: {validation_msg}")
                print(f"   Status code: {status_code}")

                if violations:
                    print(f"\n🔍 Found {len(violations)} constraint violations:")
                    for i, violation in enumerate(violations[:10], 1):  # Show first 10
                        violation_type = violation.get('type', 'unknown')
                        description = violation.get('description', 'No description')
                        print(f"  {i}. {violation_type}: {description}")

                    if len(violations) > 10:
                        print(f"  ... and {len(violations) - 10} more violations")

                    # Show summary of violation types
                    violation_types = {}
                    for violation in violations:
                        vtype = violation.get('type', 'unknown')
                        violation_types[vtype] = violation_types.get(vtype, 0) + 1

                    print(f"\n📊 Violation summary:")
                    for vtype, count in sorted(violation_types.items()):
                        print(f"  • {vtype}: {count}")
                else:
                    print("   No specific violations identified")

        if result['has_comparison']:
            print("\n" + "="*85)
            print("COMPARISON WITH GUROBI RESULT")
            print("="*85)

            print(f"  Gurobi objective: {result['gurobi_obj']:.1f}")
            print(f"  Gurobi runtime: {result['gurobi_time']:.3f} seconds")

            print("\n--- FINAL COMPARISON ---")
            print(f"  Gurobi (exact):      {result['gurobi_obj']:.1f} (in {result['gurobi_time']:.3f}s)")
            print(f"  Heuristic:           {result['heuristic_obj']:.1f} (in {result['heuristic_time']:.3f}s)")
            print(f"  MIP Gap:             {result['mip_gap']:.2f}%")
            print(f"  Runtime speedup:     {result['speedup']:.1f}x faster")

            # Check validation status before claiming optimality/quality
            is_valid = validation_results.get('is_feasible', True) if validation_results else True
            tardiness_info = validation_results.get('tardiness_info', []) if validation_results else []
            total_tardiness = validation_results.get('total_tardiness', 0) if validation_results else 0
            
            # Check if heuristic actually found a solution
            if not result['heuristic_feasible'] or result['heuristic_obj'] <= 0:
                print("  ❌ Heuristic failed to find a valid solution")
            elif not is_valid:
                # Solution found but validation failed
                if tardiness_info:
                    # Has time window violations
                    print(f"  ⚠️  Heuristic found solution with time window violations:")
                    print(f"      • Total tardiness: {total_tardiness}")
                    print(f"      • Late unit loads: {validation_results.get('num_late_unit_loads', 0)}")
                    print(f"      • Late moves: {validation_results.get('num_late_moves', 0)}")
                    
                    # Show first few violations
                    print(f"\n      Tardiness details:")
                    for i, info in enumerate(tardiness_info[:5], 1):
                        ul_id = info.get('ul_id', 'unknown')
                        move_type = info.get('move_type', 'unknown')
                        tardiness = info.get('tardiness', 0)
                        time_window = info.get('time_window', 'unknown')
                        print(f"        {i}. UL {ul_id} ({move_type}): {tardiness} units late (window: {time_window})")
                    
                    if len(tardiness_info) > 5:
                        print(f"        ... and {len(tardiness_info) - 5} more violations")
                else:
                    # Has other constraint violations
                    print(f"  ❌ Heuristic solution violates constraints (validation failed)")
            else:
                # Validation passed - show performance based on MIP gap
                if result['mip_gap'] <= 0:
                    print("  🎉 Heuristic found optimal solution!")
                elif result['mip_gap'] <= 5:
                    print("  ✅ Excellent heuristic performance (≤5% gap)")
                elif result['mip_gap'] <= 10:
                    print("  ✅ Good heuristic performance (≤10% gap)")
                else:
                    print("  ⚠️  Heuristic has room for improvement (>10% gap)")
        else:
            print("  ⚠ No Gurobi result available for comparison")
            print("  💡 Tip: Run the exact solver first or provide --gurobi-result path")
    else:
        # Non-verbose mode: just print summary
        if result['heuristic_feasible'] and result['heuristic_obj'] > 0:
            validation_symbol = "✅"
            validation_msg = ""

            if validation_results:
                is_valid = validation_results.get('is_feasible', False)
                tardiness_info = validation_results.get('tardiness_info', [])
                
                if not is_valid:
                    if tardiness_info:
                        # Time window violations
                        validation_symbol = "⚠️"
                        total_tardiness = validation_results.get('total_tardiness', 0)
                        num_late_ul = validation_results.get('num_late_unit_loads', 0)
                        validation_msg = f"TW violations: {num_late_ul} late ULs, tardiness: {total_tardiness}"
                    else:
                        # Other constraint violations
                        validation_symbol = "❌"
                        violations = validation_results.get('violations', [])
                        if violations:
                            # Count violation types
                            violation_types = {}
                            for violation in violations:
                                vtype = violation.get('type', 'unknown')
                                violation_types[vtype] = violation_types.get(vtype, 0) + 1

                            # Create summary message
                            top_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:2]
                            violation_summary = ", ".join([f"{count} {vtype}" for vtype, count in top_violations])
                            validation_msg = f"Constraint violations: {violation_summary}"
                        else:
                            validation_msg = validation_results.get('message', 'Invalid')

            print(f"  {validation_symbol} Solved: {result['heuristic_obj']:.1f} (in {result['heuristic_time']:.3f}s)")
            if result['has_comparison']:
                gurobi_gap_info = ""
                if 'gurobi_mipgap' in result and result['gurobi_mipgap'] != 'N/A':
                    gurobi_gap = result['gurobi_mipgap']
                    if abs(gurobi_gap) < 1e-6:
                        gurobi_gap_info = " [Gurobi: optimal]"
                    else:
                        gurobi_gap_info = f" [Gurobi gap: {gurobi_gap*100:.2f}%]"
                print(f"     vs Gurobi: {result['gurobi_obj']:.1f} (gap: {result['mip_gap']:.2f}%, speedup: {result['speedup']:.1f}x){gurobi_gap_info}")
            if validation_msg:
                print(f"     {validation_msg}")
        else:
            print(f"  ❌ Failed to solve (possibly infeasible)")

    return result


def solve_all_missing_heuristic_instances(experiments_dir="experiments", verbose=False, astar_time_limit=None, vrp_time_limit=None, enable_visualization=True, vrp_solver='scheduling', validate_gap=True, overwrite=False):
    """
    Find all solved Gurobi instances that don't have heuristic results and solve them.
    """
    print(f"Searching for solved instances in {experiments_dir}...")
    
    if overwrite:
         # Find ALL solved instances, disregarding whether heuristic result exists
         solved_instances = find_solved_gurobi_instances(experiments_dir)
         instances_needing_heuristic = []
         
         for instance_file, result_file in solved_instances:
            # Extract fleet size from result path
            fleet_size = 1
            if 'fleet_size_' in result_file:
                try:
                    fleet_size = int(result_file.split('fleet_size_')[1].split('/')[0])
                except:
                    fleet_size = 1
            
            instances_needing_heuristic.append((instance_file, fleet_size))
            
         if verbose:
             print(f"Overwrite enabled: Processing all {len(instances_needing_heuristic)} found Gurobi instances.")
    else:
         instances_needing_heuristic = find_instances_without_heuristic_results(experiments_dir)
    
    if not instances_needing_heuristic:
        print("No instances found that need heuristic solving.")
        return
    
    print(f"Found {len(instances_needing_heuristic)} instances to process.")
    
    successful_solves = 0
    failed_solves = 0
    skipped_solves = 0
    
    for i, (instance_file, fleet_size) in enumerate(instances_needing_heuristic, 1):
        print(f"\n--- Processing instance {i}/{len(instances_needing_heuristic)}: {os.path.basename(instance_file)} (fleet_size={fleet_size}) ---")
        print(f"    Input file: {instance_file}")
        
        try:
            # Load and solve the instance
            instance_loader = InstanceLoader(instance_file)
            instance = Instance(instanceLoader=instance_loader)
            
            # Override the fleet size to match the result file
            instance.fleet_size = fleet_size
            if verbose:
                print(f"   Fleet size set to: {fleet_size}")
            
            result = solve_instance(instance, verbose, instance_file, None, astar_time_limit, vrp_time_limit, enable_visualization, vrp_solver, validate_gap=validate_gap, overwrite=overwrite)
            
            if result.get('skipped'):
                skipped_solves += 1
                continue

            # Check if we have a valid solution (both A* and VRP succeeded)
            if result and result.get('heuristic_feasible'):
                # Check validation from the file that was just written
                validation_results = get_validation_results_from_file(instance_file, fleet_size)
                solution_valid = False  # Default to not valid if results can't be read
                if validation_results:
                    solution_valid = validation_results.get('is_feasible', False)
                
                if solution_valid:
                    print(f"✅ Successfully solved {os.path.basename(instance_file)}")
                    print(f"   Distance: {result['heuristic_obj']:.1f}, Time: {result['heuristic_time']:.3f}s")
                    successful_solves += 1
                else:
                    print(f"⚠️ Solved with violations {os.path.basename(instance_file)}")
                    print(f"   Distance: {result['heuristic_obj']:.1f}, Time: {result['heuristic_time']:.3f}s")
                    if validation_results and 'violations' in validation_results and validation_results['violations']:
                        # Create a summary of top 2 violation types
                        violation_types = {}
                        for v in validation_results['violations']:
                            vtype = v.get('type', 'unknown')
                            violation_types[vtype] = violation_types.get(vtype, 0) + 1
                        
                        top_violations = sorted(violation_types.items(), key=lambda item: item[1], reverse=True)[:2]
                        violation_summary = ", ".join([f"{count} {vtype}" for vtype, count in top_violations])
                        print(f"   Validation issues: {violation_summary}")
                    elif validation_results:
                         print(f"   Validation failed: {validation_results.get('message', 'No details')}")

                    successful_solves += 1  # Still count as solved, but with warnings
            else:
                print(f"❌ Failed to solve {os.path.basename(instance_file)} - no solution found (possibly infeasible)")
                failed_solves += 1
                
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(instance_file)}: {e}")
            failed_solves += 1
            continue
    
    print(f"\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Total instances processed: {len(instances_needing_heuristic)}")
    print(f"Successfully solved: {successful_solves}")
    print(f"Skipped: {skipped_solves}")
    print(f"Failed to solve: {failed_solves}")
    total_attempts = successful_solves + failed_solves
    if total_attempts > 0:
        print(f"Success rate (of attempted): {100 * successful_solves / total_attempts:.1f}%")



def find_gurobi_result_file(instance_path, fleet_size_override=None):
    """
    Find the corresponding Gurobi result file for an instance.
    Uses the same pattern as heuristic results but without the _heuristic suffix.
    
    Args:
        instance_path: Path to the instance JSON file
        fleet_size_override: Optional fleet size to use instead of reading from instance
        
    Returns:
        Path to the corresponding Gurobi result file or None if not found
    """
    # Convert instance path to result path using the same pattern as heuristic results
    # Replace 'inputsBRR' with 'resultsBRR' (same as heuristic but without _heuristic suffix)
    if 'inputsBRR' in instance_path:
        # Use override fleet size if provided, otherwise read from instance
        if fleet_size_override is not None:
            fleet_size = fleet_size_override
        else:
            try:
                with open(instance_path, 'r') as f:
                    instance_data = json.load(f)
                    fleet_size = instance_data.get('fleet_size', 1)
            except:
                fleet_size = 1
        
        # Replace inputsBRR with resultsBRR
        result_path = instance_path.replace('inputsBRR', 'resultsBRR')
        
        # Insert fleet_size directory before the filename
        path_parts = result_path.split('/')
        filename = path_parts[-1]
        directory = '/'.join(path_parts[:-1])
        result_path = os.path.join(directory, filename)
        
        if os.path.exists(result_path):
            return result_path
    
    return None


def load_gurobi_result(result_path):
    """
    Load Gurobi result from JSON file.
    
    Args:
        result_path: Path to the Gurobi result JSON file
        
    Returns:
        Dictionary with 'objective', 'runtime', and 'mipgap' keys, or None if failed
    """
    try:
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # Check if results are nested under a 'results' key
        if 'results' in result_data:
            result_data = result_data['results']
        
        # Extract objective value
        objective = None
        if 'objective' in result_data:
            objective = float(result_data['objective'])
        elif 'objective_value' in result_data:
            objective = float(result_data['objective_value'])
        elif 'total_distance' in result_data:
            objective = float(result_data['total_distance'])
        
        # Extract runtime
        runtime = None
        if 'runtime' in result_data:
            runtime = float(result_data['runtime'])
        elif 'solve_time' in result_data:
            runtime = float(result_data['solve_time'])
        elif 'computation_time' in result_data:
            runtime = float(result_data['computation_time'])
        
        # Extract Gurobi's MIP gap
        gurobi_mipgap = None
        if 'mipgap' in result_data:
            gurobi_mipgap = float(result_data['mipgap'])
        
        if objective is not None and runtime is not None:
            result = {
                'objective': objective,
                'runtime': runtime,
                'feasible': True
            }
            if gurobi_mipgap is not None:
                result['gurobi_mipgap'] = gurobi_mipgap
            return result
        else:
            return None
            
    except Exception as e:
        return None


def solve_with_comparison(instance, instance_path, verbose=False, gurobi_result_path=None, astar_time_limit=None, vrp_time_limit=None, vrp_solver='scheduling'):
    """
    Solve instance with heuristic and compare against existing Gurobi result.
    
    Args:
        instance: Problem instance
        instance_path: Path to the instance file
        verbose: Enable verbose output
        gurobi_result_path: Path to Gurobi result JSON file (optional)
        astar_time_limit: Maximum time in seconds for A* search (None for no limit)
        vrp_time_limit: Maximum time in seconds for VRP solving (None for no limit)
        vrp_solver: Which VRP solver to use ('ortools' or 'scheduling')
        
    Returns:
        Dictionary with results and comparison data
    """
    
    # Solve with heuristic
    test_case = TestCaseBrr(instance=instance, variant="dynamic_multiple", verbose=verbose, mode="heuristic")
    
    # Start timing
    test_case.start_heuristic_timer()
    
    # Step 1: Create the chronological task queue from Time Windows
    task_queue = create_task_queue(test_case.instance.get_unit_loads(), verbose=verbose)
    test_case.set_task_queue(task_queue)

    # Step 2: Use A* to determine the sequence of moves required to execute the tasks
    test_case.solve_heuristic_astar(time_limit=astar_time_limit)

    # Check if A* found a solution
    if test_case.astar_failed:
        # A* failed to find a solution - skip VRP and validation
        test_case.end_heuristic_timer()
        print("  ⚠️ A* failed to find a solution. Skipping VRP and validation.")
        
        result = {
            'instance': os.path.basename(instance_path),
            'heuristic_obj': float('inf'),
            'heuristic_time': test_case.heuristic_runtime,
            'heuristic_feasible': False,
            'test_case': test_case,
            'has_comparison': False,
            'astar_failed': True
        }
        return result

    # Step 3: Use a VRP heuristic to assign the moves to the available vehicles
    test_case.solve_heuristic_vrp(time_limit=vrp_time_limit, solver=vrp_solver)
    
    # End timing
    test_case.end_heuristic_timer()
    
    # Ensure objective is calculated
    if test_case.heuristic_objective is None:
        test_case.calculate_heuristic_objective()
    
    # Return basic result - comparison with Gurobi will be done in solve_instance before saving
    result = {
        'instance': os.path.basename(instance_path),
        'heuristic_obj': test_case.heuristic_objective,
        'heuristic_time': test_case.heuristic_runtime,
        'heuristic_feasible': test_case.amr_assignments is not None and bool(test_case.amr_assignments),
        'test_case': test_case,
        'has_comparison': False  # Will be updated in solve_instance
    }
    
    return result


def get_validation_results_from_file(instance_file_path, fleet_size_override=None):
    """
    Read validation results from a saved heuristic result file.
    
    Args:
        instance_file_path: Path to the original instance file
        fleet_size_override: Optional fleet size to use instead of reading from instance
        
    Returns:
        Dictionary with validation results or None if not found
    """
    try:
        from src.test_cases.writer_functions import generate_heuristic_result_path
        result_path = generate_heuristic_result_path(instance_file_path, fleet_size_override)
        
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                data = json.load(f)
                return data.get('results', {}).get('validation', None)
    except Exception as e:
        pass
    
    return None


def validate_heuristic_solution(instance, test_case, verbose=False):
    """
    Validate a heuristic solution using the constraint checker.
    
    Returns:
        tuple: (is_valid, status_code, message)
    """
    try:
        # Only validate if we have a solution
        if not test_case.amr_assignments or 'vehicles' not in test_case.amr_assignments:
            return False, -1, "No solution to validate - AMR assignments missing"
        
        # Check if the solution has any vehicles with routes
        has_routes = False
        for vehicle in test_case.amr_assignments.get('vehicles', []):
            if vehicle.get('route') and len(vehicle['route']) > 0:
                has_routes = True
                break
        
        if not has_routes:
            return False, -1, "No valid routes to validate - all vehicles have empty routes"
        
        # Create a solution dictionary from the test case
        # We need to extract the decisions from the test case
        from src.test_cases.writer_functions import translate_heuristic_decisions_simple
        
        translated_decisions = translate_heuristic_decisions_simple(test_case.amr_assignments, test_case.instance)
        
        if not translated_decisions:
            return False, -1, "No decisions to validate"
        
        # Extract decision strings (same format as result checker)
        all_decision_values = []
        for vehicle_id, vehicle_decisions in translated_decisions.items():
            for timestamp, decision_details in vehicle_decisions.items():
                decision_value = decision_details.get("decision")
                if decision_value:
                    all_decision_values.append(decision_value)
        
        if not all_decision_values:
            return False, -1, "No decision values found"
        
        # Create solution dictionary (same format as result checker)
        solution = {}
        for decision in all_decision_values:
            solution[decision] = 1
        
        # Run constraint validation with additional error handling
        validation_test_case = TestCaseBrr(instance=instance, variant="dynamic_multiple", solution=solution, verbose=verbose, mode="check")
        status = validation_test_case.check_solution()
        
        if status == 2:  # Optimal/feasible
            return True, status, "Solution is feasible and optimal"
        elif status == 13:  # Suboptimal but feasible
            return True, status, "Solution is feasible but suboptimal"
        elif status == 9:  # Time limit reached but feasible solution found
            return True, status, "Solution is feasible (time limit reached)"
        elif status == 10:  # Solution limit reached
            return True, status, "Solution is feasible (solution limit reached)"
        elif status == 3:  # Infeasible
            return False, status, "Solution violates constraints"
        else:
            return False, status, f"Validation failed with status {status}"
            
    except Exception as e:
        return False, -1, f"Validation error: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve BRR instances using heuristic approach with efficient comparison")
    parser.add_argument("--instance", type=str, help="Path to the instance json file")
    parser.add_argument("--directory", type=str, help="Path to a directory containing instance json files")
    parser.add_argument("--auto-solve", action='store_true', help="Automatically solve all instances that have Gurobi results but no heuristic results")
    parser.add_argument("--experiments-dir", type=str, default="experiments", help="Path to experiments directory (default: experiments)")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output")
    parser.add_argument("--gurobi-result", type=str, help="Path to Gurobi result JSON file for comparison")
    parser.add_argument("--astar-time-limit", type=float, default=300.0, help="Time limit in seconds for A* search (default: 300s)")
    parser.add_argument("--vrp-time-limit", type=float, default=300.0, help="Time limit in seconds for VRP solving (default: 300s)")
    parser.add_argument("--vrp-solver", type=str, default='scheduling', choices=['ortools', 'scheduling'], 
                        help="VRP solver to use: 'ortools' for routing-based, 'scheduling' for CP-SAT (default: ortools)")
    parser.add_argument("--fleet-size", type=int, help="Override the fleet size from the instance file")
    parser.add_argument("--auto-visualize", action='store_true', default=True, help="Automatically create visualizations (default: enabled)")
    parser.add_argument("--no-visualize", action='store_true', help="Disable automatic visualization")
    parser.add_argument("--no-validate-gap", action='store_true', help="Skip Gurobi validation and MIP gap calculation")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing result files (default: False)")
    args = parser.parse_args()

    if not args.instance and not args.directory and not args.auto_solve:
        print("Error: Either --instance, --directory, or --auto-solve must be provided")
        sys.exit(1)

    verbose = args.verbose
    
    # Determine visualization setting
    enable_visualization = args.auto_visualize and not args.no_visualize

    if args.auto_solve:
        # Auto-solve mode: find all solved Gurobi instances and solve with heuristic
        solve_all_missing_heuristic_instances(args.experiments_dir, verbose, args.astar_time_limit, args.vrp_time_limit, enable_visualization, args.vrp_solver, validate_gap=not args.no_validate_gap, overwrite=args.overwrite)
        
    elif args.instance:
        # Single instance mode
        if not os.path.exists(args.instance):
            print(f"Instance file not found: {args.instance}")
            sys.exit(1)

        instance_loader = InstanceLoader(args.instance)
        instance = Instance(instanceLoader=instance_loader)
        
        # Override fleet size if provided
        if args.fleet_size is not None:
            instance.fleet_size = args.fleet_size
            if verbose:
                print(f"Fleet size overridden to: {args.fleet_size}")
        
        solve_instance(instance, verbose, args.instance, args.gurobi_result, args.astar_time_limit, args.vrp_time_limit, enable_visualization, args.vrp_solver, validate_gap=not args.no_validate_gap, overwrite=args.overwrite)
    
    elif args.directory:
        # Directory mode
        if not os.path.exists(args.directory):
            print(f"Directory not found: {args.directory}")
            sys.exit(1)

        # Find all JSON files in the directory recursively
        json_files = glob.glob(os.path.join(args.directory, "**", "*.json"), recursive=True)
        
        if not json_files:
            print(f"No JSON files found in directory: {args.directory}")
            sys.exit(1)

        print(f"Found {len(json_files)} JSON files in {args.directory}")
        
        successful_solves = 0
        failed_solves = 0
        skipped_solves = 0
        results = []
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n--- Processing file {i}/{len(json_files)}: {os.path.basename(json_file)} ---")
            try:
                instance_loader = InstanceLoader(json_file)
                instance = Instance(instanceLoader=instance_loader)
                
                # Override fleet size if provided
                if args.fleet_size is not None:
                    instance.fleet_size = args.fleet_size
                    if verbose:
                        print(f"Fleet size overridden to: {args.fleet_size}")
                
                result = solve_instance(instance, verbose, json_file, None, args.astar_time_limit, args.vrp_time_limit, enable_visualization, validate_gap=not args.no_validate_gap, overwrite=args.overwrite)
                
                if result.get('skipped'):
                    skipped_solves += 1
                    continue

                # Check if got a result dictionary (from solve_with_comparison) or test_case (legacy)
                if isinstance(result, dict):
                    # New path: check validation feasible status
                    if result.get('heuristic_feasible', False) and result.get('validation_feasible', True):
                        successful_solves += 1
                        print(f"✅ Successfully solved")
                    else:
                        failed_solves += 1
                        if not result.get('heuristic_feasible', False):
                            print(f"❌ Failed to solve (heuristic failed)")
                        else:
                            print(f"❌ Failed validation (solution infeasible)")
                else:
                    # Legacy path: check test_case assignments
                    if result.amr_assignments:
                        successful_solves += 1
                        print(f"✅ Successfully solved")
                    else:
                        failed_solves += 1
                        print(f"❌ Failed to solve (possibly infeasible)")
                    
            except Exception as e:
                print(f"❌ Error processing {json_file}: {e}")
                failed_solves += 1
                continue
        
        print(f"\n=== BATCH PROCESSING SUMMARY ===")
        print(f"Total files processed: {len(json_files)}")
        print(f"Successfully solved: {successful_solves}")
        print(f"Skipped: {skipped_solves}")
        print(f"Failed to solve: {failed_solves}")
        total_attempts = successful_solves + failed_solves
        if total_attempts > 0:
            print(f"Success rate (of attempted): {100 * successful_solves / total_attempts:.1f}%")