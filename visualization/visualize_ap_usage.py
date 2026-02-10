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
    parser = argparse.ArgumentParser(description="Generate frequency heatmaps for Access Point usage.")
    parser.add_argument('--instance-type', type=str, help='Instance type (e.g. manual, manual2)')
    parser.add_argument('--file', type=str, help='Path to a single result JSON file')
    return parser.parse_args()

def get_ap_coords(ap_id, data):
    # For travel, we want the Access Point location (entry to aisle)
    for ap in data['access_points']:
        if ap['ap_id'] == ap_id:
            return ap['global_x'], ap['global_y']
    return None

def process_file(file_path, heatmap, warehouse):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

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
                        pass
    
    decisions.sort(key=lambda x: x[0])
    
    for t, vehicle, decision_obj in decisions:
        # Update AP Usage based on moves
        match = re.search(r"\[(\d+), (\d+)\].*\[(\d+), (\d+)\]", decision_obj.get("move", ""))
        
        if match:
            i, j_slot, k, l_slot = map(int, match.groups())
            
            ap1_coords = get_ap_coords(i, data)
            ap2_coords = get_ap_coords(k, data)
            
            rows, cols = warehouse.shape

            # Increment count for Start AP
            if ap1_coords:
                 gx, gy = ap1_coords
                 if 0 <= gy < rows and 0 <= gx < cols:
                     heatmap[gy, gx] += 1
            
            # Increment count for End AP
            if ap2_coords:
                 gx, gy = ap2_coords
                 if 0 <= gy < rows and 0 <= gx < cols:
                     heatmap[gy, gx] += 1

def main():
    args = parse_args()
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
            
        # Infer instance_type and ul_config from path if possible
        path_parts = args.file.split(os.sep)
        instance_type = "single_files"
        if "resultsBRR" in path_parts:
            idx = path_parts.index("resultsBRR")
            if idx + 1 < len(path_parts):
                instance_type = path_parts[idx+1]
        
        ul_config = next((p for p in path_parts if p.startswith('unit_loads_')), "unknown_config")
        base_name = os.path.basename(args.file).replace(".json", "")
        
        groups = {f"{ul_config}_{base_name}": [args.file]}
        output_dir = os.path.join("heatmaps", instance_type, "access_points")
    else:
        if not args.instance_type:
            print("Error: Either --instance-type or --file must be provided.")
            return

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

        output_dir = os.path.join("heatmaps", args.instance_type, "access_points")

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
            heatmap = heatmap.astype(float) / len(files)

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
        
        # 2. Heatmap Overlay (Circles)
        y_coords, x_coords = np.nonzero(heatmap)
        usage_values = heatmap[y_coords, x_coords]
        
        if len(usage_values) > 0:
            sc = plt.scatter(x_coords, y_coords, c=usage_values, cmap='plasma', 
                            s=300, edgecolors='black', linewidths=0.5, alpha=0.9, zorder=10)
            cbar = plt.colorbar(sc)
            label = 'Average Access Point Usage Count' if not args.file and len(files) > 1 else 'Access Point Usage Count'
            cbar.set_label(label)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{ul_config}_heatmap_ap.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == '__main__':
    main()
