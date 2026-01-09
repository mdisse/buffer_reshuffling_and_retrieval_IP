import csv
import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath('./src'))

from src.instance.instance import Instance
from src.examples_gen.multi_robot_constructive_gen import MultiRobotConstructiveGenerator

def create_layout_csv(filename):
    # -5 is path
    # 1 is storage
    # 2 is sink
    # 3 is source
    
    # 8 cols x 3 rows storage
    # Surrounded by paths
    
    storage_w = 8
    storage_h = 3
    
    rows = []
    
    # Pad width: storage_w + 2 (left/right path)
    width = storage_w + 2
    
    # Top path - Row 0
    rows.append([-5] * width)
    
    # Storage rows - Rows 1, 2, 3
    for _ in range(storage_h):
        # 1 col of path, 8 cols of storage, 1 col of path
        row = [-5] + [1] * storage_w + [-5]
        rows.append(row)
        
    # Bottom path (Access from South) - Row 4
    rows.append([-5] * width)
    
    # Source/Sink row - Row 5
    # Let's put Source at bottom left, Sink at bottom right
    # Note: Source/Sink need to be accessible from path.
    # Row 4 is path. Row 5 is below it.
    row_ss = [-5] * width
    row_ss[1] = 3 # Source
    row_ss[-2] = 2 # Sink
    rows.append(row_ss)

    # Bottom padding path to ensure source/sink are fully surrounded/accessible if needed
    # Usually strictly surrounding isn't needed if connected, but let's add one more path row
    rows.append([-5] * width)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Created layout: {filename}")

def main():
    layout_file = "examples/custom_8x3.csv"
    create_layout_csv(layout_file)
    
    output_json = "instance_8x3_south_only.json"
    
    seed = 42
    fill_level = 0.8 
    fleet_size = 2 
    
    # Access: Only South
    access_directions = {
        "north": False,
        "east": False,
        "south": True,
        "west": False
    }

    # Generator for loads
    # Using 2 robots for generation
    gen = MultiRobotConstructiveGenerator(num_robots=fleet_size, seed=seed, fill_level=fill_level)
    
    instance = Instance(
        layout_file=layout_file,
        seed=seed,
        access_directions=access_directions,
        max_p=0, # BRR instance
        fill_level=fill_level,
        fleet_size=fleet_size,
        vehicle_speed=1,
        handling_time=1,
        exampleGenerator=gen,
        rs_max=200,
        as_max=150,
        time_window_length=50
    )
    
    # Save instance
    instance.save_instance(output_json)
    print(f"Created instance: {output_json}")

if __name__ == "__main__":
    main()
