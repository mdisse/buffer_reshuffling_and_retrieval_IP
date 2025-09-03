"""
Automatic Visualization Module for BRR Results

This module provides automatic visualization capabilities for both optimization
and heuristic results using multiprocessing for efficiency.
"""

import os
import sys
import subprocess
import multiprocessing as mp
from typing import Optional, List
import json
import time


def run_visualization_script(script_path: str, result_file: str, args: List[str] = None) -> bool:
    """
    Run a visualization script in a subprocess.
    
    Args:
        script_path: Path to the visualization script
        result_file: Path to the result file to visualize
        args: Additional arguments for the script
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Build command
        cmd = [sys.executable, script_path, '--file', result_file, '--color']
        if args:
            cmd.extend(args)
        
        # Run with timeout to prevent hanging
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(script_path) or '.'
        )
        
        if result.returncode == 0:
            return True
        else:
            print(f"Visualization failed for {os.path.basename(result_file)}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"Visualization timeout for {os.path.basename(result_file)}")
        return False
    except Exception as e:
        print(f"Visualization error for {os.path.basename(result_file)}: {e}")
        return False


def visualize_result_async(result_file: str, overwrite: bool = True, 
                          create_timestep: bool = True, create_astar: bool = True) -> None:
    """
    Create visualizations for a result file asynchronously.
    
    Args:
        result_file: Path to the result file
        overwrite: Whether to overwrite existing visualizations
        create_timestep: Whether to create timestep-based visualization
        create_astar: Whether to create A* move visualization (for heuristic results)
    """
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return
    
    # Clean up old visualizations if overwrite is enabled
    if overwrite:
        cleanup_old_visualizations(result_file)
    
    # Get script directory (going up two levels from src/visualization/ to root)
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    timestep_script = os.path.join(script_dir, 'visualize_BRR_steps.py')
    astar_script = os.path.join(script_dir, 'visualize_astar_moves.py')
    
    # Check if scripts exist
    if create_timestep and not os.path.exists(timestep_script):
        print(f"Timestep visualization script not found: {timestep_script}")
        create_timestep = False
    
    if create_astar and not os.path.exists(astar_script):
        print(f"A* visualization script not found: {astar_script}")
        create_astar = False
    
    # Determine result type
    is_heuristic = '_heuristic.json' in result_file
    
    # For heuristic results, check if the A* search found a solution before creating visualizations
    if is_heuristic:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                astar_result = data.get('results', {}).get('astar_result', {})
                has_moves = astar_result.get('move_count', 0) > 0 or len(astar_result.get('move_sequence', [])) > 0
                
                # Skip all visualizations if A* found no moves
                if not has_moves:
                    print(f"⚠️  Skipping visualizations for {result_file} - A* search found no solution")
                    return
        except Exception as e:
            print(f"⚠️  Could not check A* status for {result_file}: {e}")
            # Continue with visualization attempt even if we can't check A* status
    
    visualizations_created = []
    
    # Create timestep visualization
    if create_timestep:
        if run_visualization_script(timestep_script, result_file):
            visualizations_created.append('timestep')
    
    # Create A* visualization for heuristic results
    if create_astar and is_heuristic:
        if run_visualization_script(astar_script, result_file):
            visualizations_created.append('A* moves')
    
    # Print status
    if visualizations_created:
        viz_list = ', '.join(visualizations_created)
        print(f"📊 Created visualizations for {result_file}: {viz_list}")
    else:
        print(f"⚠️  No visualizations created for {result_file}")


def auto_visualize(result_file: str, background: bool = True, overwrite: bool = True) -> Optional[mp.Process]:
    """
    Automatically create visualizations for a result file.
    
    Args:
        result_file: Path to the result file to visualize
        background: Whether to run visualization in background process
        overwrite: Whether to overwrite existing visualizations
        
    Returns:
        Process object if background=True, None otherwise
    """
    if not os.path.exists(result_file):
        print(f"Result file not found: {result_file}")
        return None
    
    if background:
        # Create process for background visualization
        process = mp.Process(
            target=visualize_result_async,
            args=(result_file, overwrite, True, True),
            daemon=True  # Dies when main process exits
        )
        process.start()
        return process
    else:
        # Run visualization synchronously
        visualize_result_async(result_file, overwrite, True, True)
        return None


def auto_visualize_batch(result_files: List[str], max_workers: int = None, 
                        overwrite: bool = True) -> List[mp.Process]:
    """
    Create visualizations for multiple result files using multiprocessing.
    
    Args:
        result_files: List of result file paths
        max_workers: Maximum number of worker processes (default: CPU count)
        overwrite: Whether to overwrite existing visualizations
        
    Returns:
        List of active processes
    """
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(result_files))
    
    print(f"🎨 Starting batch visualization with {max_workers} workers for {len(result_files)} files")
    
    # Use multiprocessing Pool for efficient batch processing
    processes = []
    
    # Process files in batches
    with mp.Pool(max_workers) as pool:
        # Submit all visualization tasks
        results = []
        for result_file in result_files:
            if os.path.exists(result_file):
                result = pool.apply_async(
                    visualize_result_async,
                    args=(result_file, overwrite, True, True)
                )
                results.append((result_file, result))
        
        # Wait for all tasks to complete
        for result_file, result in results:
            try:
                result.get(timeout=600)  # 10 minute timeout per file
            except mp.TimeoutError:
                print(f"⏰ Visualization timeout for {os.path.basename(result_file)}")
            except Exception as e:
                print(f"❌ Visualization error for {os.path.basename(result_file)}: {e}")
    
    print(f"✅ Batch visualization completed")
    return processes


def cleanup_old_visualizations(result_file: str) -> None:
    """
    Remove old visualization directories for a result file.
    
    Args:
        result_file: Path to the result file
    """
    base_name = os.path.splitext(result_file)[0]
    result_dir = os.path.dirname(result_file)
    result_basename = os.path.basename(os.path.splitext(result_file)[0])
    
    # List of possible visualization directories based on the actual patterns used
    viz_dirs = [
        os.path.join(result_dir, result_basename),  # timestep visualization directory
        os.path.join(result_dir, f"{result_basename}_astar")  # A* visualization directory
    ]
    
    for viz_dir in viz_dirs:
        if os.path.exists(viz_dir) and os.path.isdir(viz_dir):
            try:
                import shutil
                shutil.rmtree(viz_dir)
                print(f"🗑️  Cleaned up old visualization: {os.path.basename(viz_dir)}")
            except Exception as e:
                print(f"⚠️  Could not clean up {viz_dir}: {e}")


# Convenience functions for common use cases
def visualize_heuristic_result(result_file: str, background: bool = True) -> Optional[mp.Process]:
    """Visualize a heuristic result file (both timestep and A* visualizations)."""
    return auto_visualize(result_file, background=background, overwrite=True)


def visualize_gurobi_result(result_file: str, background: bool = True) -> Optional[mp.Process]:
    """Visualize a Gurobi result file (timestep visualization only)."""
    if background:
        process = mp.Process(
            target=visualize_result_async,
            args=(result_file, True, True, False),  # Only timestep, no A*
            daemon=True
        )
        process.start()
        return process
    else:
        visualize_result_async(result_file, True, True, False)
        return None


# Example usage functions
if __name__ == "__main__":
    # Example: Auto-visualize a single file
    # auto_visualize("experiments/resultsBRR/some_result.json")
    
    # Example: Batch visualize multiple files
    # result_files = ["file1.json", "file2.json", "file3.json"]
    # auto_visualize_batch(result_files, max_workers=4)
    
    print("Auto-visualization module loaded. Use auto_visualize() or auto_visualize_batch() functions.")
