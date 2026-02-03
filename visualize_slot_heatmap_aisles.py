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

def get_ap_coords(ap_id, data):
    # For travel, we want the Access Point location (entry to aisle)
    for ap in data['access_points']:
        if ap['ap_id'] == ap_id:
            return ap['global_x'], ap['global_y']
    return None

def bfs_path(warehouse, start, end):
    # start, end are (x, y)
    if start == end:
        return [start]
    
    rows, cols = warehouse.shape
    q = [(start, [start])]
    visited = set()
    visited.add(start)
    
    while q:
        (cx, cy), path = q.pop(0)
        
        if (cx, cy) == end:
            return path
            
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= ny < rows and 0 <= nx < cols:
                cell_val = warehouse[ny, nx]
                # Passable: Aisle (-5), Sink (2), Source (3), or explicitly Start/End nodes
                is_passable = (cell_val == -5 or cell_val == 2 or cell_val == 3 or (nx, ny) == end or (nx, ny) == start)
                
                if (nx, ny) not in visited and is_passable:
                    visited.add((nx, ny))
                    q.append(((nx, ny), path + [(nx, ny)]))
                    
    return []

def process_file(file_path, heatmap, warehouse):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    bays = data.get("initial_state", {})
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
                    if bay_config[r][c] and len(bay_config[r][c]) > 0:
                         ul_id = bay_config[r][c][0]
                         if ul_id != 0:
                             gy = start_row + r
                             gx = start_col + c
                             ul_positions[(gx, gy)] = [ul_id]

    # Initialize AMRs at Sink
    rows, cols = warehouse.shape
    sink_pos = None
    # Find Sink (2)
    for r in range(rows):
        for c in range(cols):
            if warehouse[r, c] == 2:
                sink_pos = (c, r) # x, y
                break
        if sink_pos: break
    
    if not sink_pos:
        sink_pos = (0, 0) # Fallback

    amr_positions = {} # veh_id -> (x, y)
    amr_last_time = {} # veh_id -> time
    
    # Collect decisions
    decisions = []
    if 'results' in data and 'decisions' in data['results']:
        for veh_id, veh_decs in data['results']['decisions'].items():
            amr_positions[veh_id] = sink_pos
            amr_last_time[veh_id] = 0
            
            if isinstance(veh_decs, list):
                for d in veh_decs:
                    decisions.append((d['time_step'], veh_id, d))
            elif isinstance(veh_decs, dict):
                 for t_str, d in veh_decs.items():
                     try:
                        t = int(t_str)
                        decisions.append((t, veh_id, d))
                     except ValueError:
                        pass
    
    decisions.sort(key=lambda x: x[0])
    
    current_time = 0
    
    for t, vehicle, decision_obj in decisions:
        dt = t - current_time
        
        # 1. Update Slots Heatmap (Occupancy) - SKIPPED for aisles view
        # if dt > 0:
        #     for (gx, gy), uls in ul_positions.items():
        #         if uls and len(uls) > 0 and (uls[0] is not None and uls[0] != 0):
        #             if 0 <= gy < rows and 0 <= gx < cols:
        #                 heatmap[gy, gx] += dt

        current_time = t
        
        # 2. Update Travel Heatmap (AMR Movement)
        match = re.search(r"\[(\d+), (\d+)\].*\[(\d+), (\d+)\]", decision_obj.get("move", ""))
        
        if match:
            i, j_slot, k, l_slot = map(int, match.groups())
            
            ap1_coords = get_ap_coords(i, data)
            ap2_coords = get_ap_coords(k, data)
            
            if ap1_coords and ap2_coords:
                # Calculate Path
                path = bfs_path(warehouse, ap1_coords, ap2_coords)
                
                travel_time = decision_obj.get('travel_time', 0)
                if travel_time > 0 and path:
                    # Distribute travel time
                    val_per_cell = travel_time / len(path)
                    for (cx, cy) in path:
                         if 0 <= cy < rows and 0 <= cx < cols:
                             heatmap[cy, cx] += val_per_cell
                             
                # Add Wait/Idle time at start node?
                total_elapsed = t - amr_last_time.get(vehicle, 0)
                wait_time = total_elapsed - travel_time
                if wait_time > 0:
                    wx, wy = ap1_coords  # Waiting at start node
                    if 0 <= wy < rows and 0 <= wx < cols:
                        heatmap[wy, wx] += wait_time
                
                # Update AMR position to end of move
                amr_positions[vehicle] = ap2_coords
                amr_last_time[vehicle] = t

        # 3. Update UL Positions
        d_type = decision_obj['decision'][0]
        if match:
             i, j_slot, k, l_slot = map(int, match.groups())
             p1 = ap_id_to_coords(i, j_slot, data)
             p2 = ap_id_to_coords(k, l_slot, data)
             
             x1, y1 = p1 if p1 else (None, None)
             x2, y2 = p2 if p2 else (None, None)
             
             if d_type == 'x': # Transfer
                 if x1 is not None and (x1, y1) in ul_positions and ul_positions[(x1, y1)]:
                     ul = ul_positions[(x1, y1)].pop(0)
                     if x2 is not None:
                          ul_positions.setdefault((x2, y2), []).append(ul)
             elif d_type == 'y': # Retrieval
                 if x1 is not None and (x1, y1) in ul_positions and ul_positions[(x1, y1)]:
                     ul_positions[(x1, y1)].pop(0)
             elif d_type == 'z': # Storage
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
        return

    output_dir = os.path.join("heatmaps", args.instance_type, "aisles")
    os.makedirs(output_dir, exist_ok=True)
    
    for ul_config, files in groups.items():
        print(f"Processing {ul_config} ({len(files)} files)...")
        
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
            process_file(file_path, heatmap, warehouse)

        if len(files) > 0:
            heatmap /= len(files)

        # Mask Sink (2) and Source (3) areas to avoid biasing the scale
        # Find sink/source indices
        for target_val in [2, 3]:
            indices = np.argwhere(warehouse == target_val)
            for r, c in indices:
                # Mask the cell itself
                heatmap[r, c] = 0
                # Mask neighbors (up, down, left, right) which are likely the entry aisles
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < warehouse.shape[0] and 0 <= nc < warehouse.shape[1]:
                        heatmap[nr, nc] = 0

        # Plotting
        plt.figure(figsize=(10, 8))
        
        # 1. Base Layer
        base_img = np.zeros((warehouse.shape[0], warehouse.shape[1], 4))
        base_img[:, :] = [1, 1, 1, 1] # White
        base_img[warehouse == 0] = [0.5, 0.5, 0.5, 1.0] # Gray
        base_img[warehouse == -5] = [0.85, 0.85, 0.85, 1.0] # LightGrey
        base_img[warehouse == 1] = [0, 0, 0, 1.0] # Black (Slots)
        
        plt.imshow(base_img)
        
        # 2. Heatmap Overlay
        heatmap_masked = np.ma.masked_where(heatmap <= 0, heatmap)
        
        plt.imshow(heatmap_masked, cmap='plasma', interpolation='nearest', alpha=0.7)
        
        cbar = plt.colorbar()
        cbar.set_label('Average Occupancy (Slots + Travel)')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{ul_config}_heatmap_aisles.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
