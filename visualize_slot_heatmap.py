import argparse
import os
import json
import re
import numpy as np
import copy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate frequency heatmaps for slot usage.")
    parser.add_argument('--instance-type', type=str, required=True, help='Instance type (e.g. manual, manual2)')
    return parser.parse_args()

def ap_id_to_coords(ap_id, slot, data):
    vl_length = next((lane['n_slots'] for lane in data['virtual_lanes'] if lane['ap_id'] == ap_id), None)
    if vl_length is None: return None
    
    for ap in data['access_points']:
        if ap['ap_id'] == ap_id:
            # Note: Logic copied from visualize_BRR_steps.py
            if ap['direction'] == 'north':
                return ap['global_x'], ap['global_y'] + 1 + vl_length - slot
            elif ap['direction'] == 'south':
                return ap['global_x'], ap['global_y'] - 1 - vl_length + slot
            elif ap['direction'] == 'east':
                return ap['global_x'] - 1 - vl_length + slot, ap['global_y']
            elif ap['direction'] == 'west':
                return ap['global_x'] + 1 + vl_length - slot, ap['global_y']
    return None

def process_file(file_path, heatmap, warehouse_shape):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    bays = data.get("initial_state", {})
    # Initialize ul_positions: map (x, y) -> list of ULs
    ul_positions = {}
    
    # Parse initial state
    for bay_key, bay_config in bays.items():
        bay_rows = len(bay_config)
        bay_cols = len(bay_config[0]) if bay_rows > 0 else 0
        pattern = r"row (\d+), column (\d+)"
        match = re.search(pattern, bay_key)
        if match:
            start_row = int(match.group(1))
            start_col = int(match.group(2))
            for r in range(bay_rows):
                for c in range(bay_cols):
                    # bay_config[r][c] is a list of ULs
                    if bay_config[r][c] and len(bay_config[r][c]) > 0:
                         ul_id = bay_config[r][c][0]
                         if ul_id != 0:
                             gy = start_row + r
                             gx = start_col + c
                             ul_positions[(gx, gy)] = [ul_id]
                         else:
                             # Initialize empty slots if needed to avoid key errors?
                             # No, defaultdict pattern or logic checks are safer.
                             pass

    # Collect decisions
    decisions = []
    if 'results' in data and 'decisions' in data['results']:
        for veh_id, veh_decs in data['results']['decisions'].items():
            if isinstance(veh_decs, list):
                for d in veh_decs:
                    decisions.append((d['time_step'], veh_id, d))
            elif isinstance(veh_decs, dict):
                 for t_str, d in veh_decs.items():
                     try:
                        t = int(t_str)
                        decisions.append((t, veh_id, d))
                     except ValueError:
                        pass # Skip non-integer keys if any
    
    decisions.sort(key=lambda x: x[0])
    
    current_time = 0
    
    # Iterate through events
    for t, vehicle, decision_obj in decisions:
        dt = t - current_time
        if dt > 0:
            for (gx, gy), uls in ul_positions.items():
                if uls and len(uls) > 0 and (uls[0] is not None and uls[0] != 0): # Check if occupied
                    # Ensure indices are valid
                    if 0 <= gy < warehouse_shape[0] and 0 <= gx < warehouse_shape[1]:
                        heatmap[gy, gx] += dt

        current_time = t
        
        # Apply Decision
        d_type = decision_obj['decision'][0]
        # move string example: "[1, 1] -> [2, 1]"
        match = re.search(r"\[(\d+), (\d+)\].*\[(\d+), (\d+)\]", decision_obj.get("move", ""))
        
        x1, y1 = None, None
        x2, y2 = None, None
        
        if match:
            i, j_slot, k, l_slot = map(int, match.groups())
            # Convert decision coords
            # Note: ap_id_to_coords returns (x, y)
            p1 = ap_id_to_coords(i, j_slot, data)
            p2 = ap_id_to_coords(k, l_slot, data)
            
            x1, y1 = p1 if p1 else (None, None)
            x2, y2 = p2 if p2 else (None, None)
            
            if d_type == 'x': # Transfer
                # Move from p1 to p2
                if x1 is not None and (x1, y1) in ul_positions and ul_positions[(x1, y1)]:
                    ul = ul_positions[(x1, y1)].pop(0)
                    if x2 is not None:
                         ul_positions.setdefault((x2, y2), []).append(ul)
            elif d_type == 'y': # Retrieval
                # Remove from p1
                if x1 is not None and (x1, y1) in ul_positions and ul_positions[(x1, y1)]:
                    ul_positions[(x1, y1)].pop(0)
            elif d_type == 'z': # Storage
                 # storage happens at p2
                 match_ul = re.search(r"n(\d+)", decision_obj["decision"])
                 if match_ul:
                     ul_id = int(match_ul.group(1))
                     if x2 is not None:
                         ul_positions.setdefault((x2, y2), []).append(ul_id)


def main():
    args = parse_args()
    base_dir = f'experiments/resultsBRR/{args.instance_type}'
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Find groups
    groups = {}
    print(f"Scanning {base_dir}...")
    for root, dirs, files in os.walk(base_dir):
        path_parts = root.split(os.sep)
        ul_part = next((p for p in path_parts if p.startswith('unit_loads_')), None)
        if ul_part:
            for file in files:
                if file.endswith('.json') and 'heuristic' in file:
                    groups.setdefault(ul_part, []).append(os.path.join(root, file))

    if not groups:
        print("No unit_loads_* folders containing heuristic JSONs found.")
        return

    output_dir = os.path.join("heatmaps", args.instance_type, "slots")
    os.makedirs(output_dir, exist_ok=True)
    
    for ul_config, files in groups.items():
        print(f"Processing {ul_config} ({len(files)} files)...")
        
        # Initialize heatmap with shape from first file
        # We assume all files in a group have same layout.
        with open(files[0], 'r') as f:
            first_data = json.load(f)
        
        layout_path = first_data.get('layout_file')
        
        try:
            warehouse = np.loadtxt(layout_path, delimiter=',', dtype=int)
        except Exception as e:
            print(f"Could not load layout {layout_path}: {e}")
            continue
            
        heatmap = np.zeros(warehouse.shape)
        
        for file_path in files:
            process_file(file_path, heatmap, warehouse.shape)
            
        if len(files) > 0:
            heatmap /= len(files)

        # Plot
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        im = plt.imshow(heatmap, cmap='plasma', interpolation='nearest')
        cbar = plt.colorbar(im)
        cbar.set_label('Average Occupied Timesteps')
        
        # Mark obstacles
        # Warehouse: 0 = Obstacle.
        # We can overlay a mask for obstacles.
        obstacles_mask = np.zeros_like(warehouse, dtype=float)
        obstacles_mask[:] = np.nan
        obstacles_mask[warehouse == 0] = 1 # Set obstacles to some value
        
        # Use a separate colormap for obstacles (e.g. gray)
        # plt.imshow(obstacles_mask, cmap='gray_r', interpolation='nearest', alpha=0.5, vmin=0, vmax=1)
        # Better: use a ListedColormap for just gray
        from matplotlib.colors import ListedColormap
        plt.imshow(obstacles_mask, cmap=ListedColormap(['gray']), interpolation='nearest', alpha=1.0)

        # Mark aisles (-5)
        aisles_mask = np.zeros_like(warehouse, dtype=float)
        aisles_mask[:] = np.nan
        aisles_mask[warehouse == -5] = 1
        plt.imshow(aisles_mask, cmap=ListedColormap(['lightgrey']), interpolation='nearest', alpha=1.0)

        # Mark Sink (2) and Source (3)
        source_sink_mask = np.zeros_like(warehouse, dtype=float)
        source_sink_mask[:] = np.nan
        source_sink_mask[(warehouse == 2) | (warehouse == 3)] = 1
        plt.imshow(source_sink_mask, cmap=ListedColormap(['white']), interpolation='nearest', alpha=1.0)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{ul_config}_heatmap.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
