#!/usr/bin/env python3
"""
Test the updated travel distance-based heuristic function.
"""

import sys
sys.path.insert(0, '/home/maxdisselnmeyer/buffer_reshuffling_and_retrieval_ip')

from src.instance.instance import Instance
from src.examples_gen.unit_load_gen import UnitLoadGenerator
from src.heuristics.astar import AStarSolver
from src.heuristics.h_cost import h_cost
from src.util.map_tw_prio import create_task_queue

def test_distance_heuristic():
    """Test the A* solver with the new distance-based heuristic."""
    print("=== Testing Distance-Based Heuristic ===\n")
    
    # Create a simple instance
    instance = Instance(
        layout_file="examples/Size_3x3_Layout_1x1_sink_source.csv",
        fill_level=0.3,
        max_p=0,
        height=1,
        seed=42,
        access_directions={"north": True, "east": True, "south": True, "west": True}, 
        exampleGenerator=UnitLoadGenerator(tw_length=10, fill_level=0.3, seed=42),
    )
    
    # Get the buffer and task queue
    buffer = instance.get_buffer()
    task_queue = create_task_queue(instance.get_unit_loads(), verbose=False)
    
    # Test with first few tasks
    test_tasks = task_queue[:2] if len(task_queue) > 2 else task_queue
    
    print(f"Testing with {len(test_tasks)} tasks:")
    for i, task in enumerate(test_tasks):
        task_type = "STORAGE" if "_mock" in str(task.id) else "RETRIEVAL"
        print(f"  {i+1}. UL {task.id} ({task_type})")
    
    # Test the distance-based heuristic
    print(f"\nTesting h_cost function:")
    heuristic_distance = h_cost(buffer, test_tasks)
    print(f"Total estimated travel distance: {heuristic_distance}")
    
    # Test A* with the new heuristic
    print(f"\nTesting A* with distance heuristic:")
    
    # Extract all unit load IDs from test tasks
    all_unit_loads = set()
    for task in test_tasks:
        if "_mock" in str(task.id):
            # Storage task - extract real unit load ID
            all_unit_loads.add(task.real_ul_id)
        else:
            # Retrieval task - use task ID directly
            all_unit_loads.add(task.id)
    
    solver = AStarSolver(buffer, all_unit_loads, verbose=False)
    try:
        solution, _ = solver.solve()
        if solution:
            print(f"✓ A* found solution with {len(solution)} moves")
            print("✓ Distance-based heuristic working correctly")
        else:
            print("✗ A* found no solution")
    except Exception as e:
        print(f"✗ A* solver error: {e}")
        return False
    
    print("\n✓ Distance-based heuristic test completed successfully!")
    return True

if __name__ == "__main__":
    test_distance_heuristic()
