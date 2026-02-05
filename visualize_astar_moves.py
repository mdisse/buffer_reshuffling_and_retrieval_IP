import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
import argparse
import math
import imageio.v2 as imageio
import copy
import matplotlib.patheffects as pe
from matplotlib import cm

# This script visualizes A* search moves step-by-step, showing the warehouse state after each move.
# It uses the `buffer_state` from the `solution_states` in the A* results for accurate visualization.

MAX_URGENCY_WINDOW_DEFAULT = 30 # Default window for urgency coloring

parser = argparse.ArgumentParser(description="Visualize A* moves step by step from buffer_state")
parser.add_argument('--file', type=str, default=None, help='JSON file containing A* results')
parser.add_argument('--no-source', dest='source', action='store_false', help='Hide source locations')
parser.set_defaults(source=True)
parser.add_argument('--color', dest='color_vls', action='store_true', help='Color virtual lanes by direction')
parser.add_argument('--no-color', dest='color_vls', action='store_false', help='Disable virtual lane coloring')
parser.set_defaults(color_vls=True)
parser.add_argument('--debug', dest='debug', action='store_true', help='Show debug information')
parser.set_defaults(debug=False)
parser.add_argument('--max_urgency_window', type=int, help='Maximum window (in time steps) for unit load retrieval urgency coloring.')
args = parser.parse_args()
source = args.source
color_vls = args.color_vls
debug = args.debug
file = args.file

def get_ul_color(ul_id, current_move_step, ul_retrieval_times_map, max_urgency_window):
    """
    Calculates the color for a unit load based on its retrieval urgency.
    
    Color scheme:
    - Due/overdue: Dark red (high urgency)
    - Within urgency window: A red -> orange -> yellow gradient
    - Not urgent yet: Light grey (low urgency)
    - Empty/null: Light grey
    """
    if ul_id is None or ul_id == 0:
        return (0.85, 0.85, 0.85) # Light grey for empty or UL ID 0

    retrieval_end = ul_retrieval_times_map.get(ul_id)

    if retrieval_end is None:
        # Default color for a UL not in the initial list (e.g., carried by an AMR)
        return (0.6, 0.6, 0.6) # Medium grey

    # Use the move step as a proxy for time to calculate urgency
    time_difference = retrieval_end - current_move_step

    if time_difference <= 0: # Due or overdue
        return (0.8, 0.0, 0.0)
    elif time_difference > max_urgency_window: # Not urgent yet
        return (0.75, 0.75, 0.75)
    else: # Within the urgency window
        if max_urgency_window == 0:
            normalized_urgency = 0
        else:
            normalized_urgency = time_difference / max_urgency_window
        
        # Use a colormap for a smooth gradient from red (urgent) to yellow (less urgent)
        cmap = plt.get_cmap('hot_r')
        color = cmap(0.3 + (1 - normalized_urgency) * 0.3)
        return (color[0], color[1], color[2])


def ap_id_to_coords(ap_id, slot, data):
    """Converts an access point ID and slot to its (x, y) coordinates.
    
    Tier numbering: Tier 1 = BACK/DEEPEST, Tier N = FRONT/CLOSEST
    For each direction, we calculate position based on total lane length.
    """
    vl_length = next((lane['n_slots'] for lane in data['virtual_lanes'] if lane['ap_id'] == ap_id), None)
    if vl_length is None:
        return None
        
    for ap in data['access_points']:
        if ap['ap_id'] == ap_id:
            if ap['direction'] == 'north':
                # North: lane extends upward (increasing y), AP at bottom
                # Tier 1 (back) is farthest up, Tier N (front) is closest to AP
                return ap['global_x'], ap['global_y'] + vl_length - slot + 1
            elif ap['direction'] == 'south':
                # South: lane extends downward (decreasing y), AP at top
                # Tier 1 (back) is farthest down, Tier N (front) is closest to AP
                return ap['global_x'], ap['global_y'] - (vl_length - slot + 1)
            elif ap['direction'] == 'east':
                # East: lane extends leftward (decreasing x), AP at right
                # Tier 1 (back) is farthest left, Tier N (front) is closest to AP
                return ap['global_x'] - (vl_length - slot + 1), ap['global_y']
            elif ap['direction'] == 'west':
                # West: lane extends rightward (increasing x), AP at left
                # Tier 1 (back) is farthest right, Tier N (front) is closest to AP
                return ap['global_x'] + vl_length - slot + 1, ap['global_y']
    return None

def parse_position(position_str):
    """Parses a position string like '[11, 1]' into (ap_id, slot)."""
    match = re.match(r'\[(\d+),\s*(\d+)\]', position_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def process_file(file):
    with open(file, 'r') as f:
        data = json.load(f)

    if 'astar_result' not in data.get('results', {}):
        print(f"No A* results found in {file}. Skipping A* visualization.")
        return

    astar_result = data['results']['astar_result']
    solution_states = astar_result.get('solution_states', [])
    move_sequence = astar_result.get('move_sequence', [])

    if not solution_states:
        print(f"No A* solution states found in {file}. Skipping A* visualization.")
        return

    ul_retrieval_times_map = {ul['id']: ul['retrieval_end'] 
                              for ul in data.get('unit_loads', []) 
                              if isinstance(ul, dict) and 'id' in ul and 'retrieval_end' in ul}
    
    max_urgency_window = args.max_urgency_window if args.max_urgency_window is not None else MAX_URGENCY_WINDOW_DEFAULT

    layout_file = data["layout_file"]
    warehouse = np.loadtxt(layout_file, delimiter=',', dtype=int)
    colors = {
        -5: 'lightgrey', # Aisles
        1: 'white',      # Slots
        2: 'green',      # Sink
        3: 'blue',       # Source
        "ap": "red",     # Access Points
    }
    legend = {
        -5: 'Aisles',
        1: 'Slots',
        2: 'Sink',
        3: 'Source',
        "ap": "Access Points",
    }
    vl_colors = {
        "north": "#5B9BD5",
        "east": "#ED7D31",
        "south": "#FFC000",
        "west": "#70AD47"
    }

    if not source:
        del colors[3]

    bays = data["initial_state"]

    def plot_warehouse_astar(move_step, ul_positions, move_info=None, initial_state=False):
        plt.figure(figsize=(12, 10))
        
        virtual_lanes = {}
        if color_vls:
            for bay in data["bay_info"].values():
                coordinates = []
                for col in range(bay["width"]):
                    for row in range(bay["length"]):
                        coordinates.append((bay["x"] + col, bay["y"] + row))
                for coordinate in coordinates:
                    x, y = coordinate[0], coordinate[1]
                    for ap in data["access_points"]:
                        ap_x, ap_y = ap["global_x"], ap["global_y"]
                        if ap["direction"] == "north": ap_y += 1
                        elif ap["direction"] == "south": ap_y -= 1
                        elif ap["direction"] == "east": ap_x -= 1
                        elif ap["direction"] == "west": ap_x += 1
                        
                        if ap_x == x and ap_y == y:
                            virtual_lanes[(x, y)] = {"ap_id": ap["ap_id"], "access_direction": ap["direction"]}
            
            additional_lanes = {}
            # Use list() to avoid runtime error for changing dict size during iteration
            for vl_coord, vl_info in list(virtual_lanes.items()):
                ap_id = vl_info["ap_id"]
                for lane in data["virtual_lanes"]:
                    if lane["ap_id"] == ap_id:
                        for slot in range(1, lane["n_slots"]):
                            x, y = vl_coord
                            if vl_info["access_direction"] == "north": y += slot
                            elif vl_info["access_direction"] == "south": y -= slot
                            elif vl_info["access_direction"] == "east": x -= slot
                            elif vl_info["access_direction"] == "west": x += slot
                            additional_lanes[(x, y)] = vl_info
            virtual_lanes.update(additional_lanes)

        def get_color(row, col):
            if color_vls and warehouse[row, col] == 1 and not initial_state:
                vl_info = virtual_lanes.get((col, row))
                return vl_colors.get(vl_info["access_direction"], 'white') if vl_info else 'white'
            else:
                try:
                    return colors[warehouse[row, col]]
                except KeyError:
                    if warehouse[row, col] == 3:
                         raise KeyError("This instance file contains a source, please use this script without the --no-source flag")
                    return 'gray'

        for row in range(warehouse.shape[0]):
            for col in range(warehouse.shape[1]):
                if debug:
                    plt.text(col + 0.5, row + 0.5, f"{(row, col)}", color='black', ha='center', va='center')
                cell_color = get_color(row, col)
                plt.gca().add_patch(plt.Rectangle((col, row), 1, 1, color=cell_color))

        linewidth = 0.05
        for bay_name, config in data["bay_info"].items():
            x = config["x"]
            y = config["y"]
            w = config["width"]
            l = config["length"]
            plt.gca().add_patch(plt.Rectangle((x, y), w, l, fill=False, edgecolor='black', linewidth=linewidth))

        access_points_to_plot = data['access_points']
        used_aps = {lane['ap_id'] for lane in data['virtual_lanes']}
        access_points_to_plot = [ap for ap in access_points_to_plot if ap['ap_id'] in used_aps]
        
        if not source:
            try:
                source_aps = set(list(data['source_info'].values())[0]['access_point_ids'])
                access_points_to_plot = [ap for ap in access_points_to_plot if ap["ap_id"] not in source_aps]
            except (KeyError, IndexError):
                pass

        for ap in access_points_to_plot:
            x_coord, y_coord = ap["global_x"], ap["global_y"]
            if ap["direction"] == "north": y_coord += 1; x_coord += 0.5
            elif ap["direction"] == "south": x_coord += 0.5
            elif ap["direction"] == "east": y_coord += 0.5
            elif ap["direction"] == "west": y_coord += 0.5; x_coord += 1
            
            if not initial_state:
                ap_marker = plt.Circle((x_coord, y_coord), radius=0.1, color='red')
                plt.gca().add_patch(ap_marker)
                plt.text(x_coord, y_coord, ap["ap_id"], color='white', ha='center', va='center', fontsize=6)

        for (x, y), ul_infos in ul_positions.items():
            if ul_infos:
                ul_info = ul_infos[0]
                ul_id = ul_info['id']
                priority = ul_info.get('priority')
                
                if ul_id is not None and ul_id != 0:
                    ul_color = get_ul_color(ul_id, move_step, ul_retrieval_times_map, max_urgency_window)
                    plt.gca().add_patch(plt.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, color=ul_color))
                    
                    text_label = f"{ul_id:02d}"
                    if priority:
                        text_label += f"\n{priority}"

                    plt.text(x + 0.5, y + 0.5, text_label, color='white', ha='center', va='center',
                             fontsize=12, path_effects=[pe.withStroke(linewidth=1, foreground='black')])

        plt.xticks(np.arange(0, warehouse.shape[1] + 1, 1), labels=[])
        plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.yticks(np.arange(0, warehouse.shape[0] + 1, 1), labels=[])
        plt.gca().invert_yaxis()
        plt.grid(True, color='black', linewidth=.5)
        
        temp_legend = legend.copy()
        if initial_state:
            del temp_legend["ap"]
        plt.legend([plt.Rectangle((0,0),1,1, color=color) for color in colors.values()], temp_legend.values(), loc='center left')
        plt.axis('scaled')
        plt.xlim(0, warehouse.shape[1])
        plt.ylim(warehouse.shape[0], 0)

        if not initial_state:
            plt.text(warehouse.shape[1] - 0.5, warehouse.shape[0] + 0.5, f"A* Move: {move_step:03d}",
                     fontsize=12, ha='right', va='bottom', color='black',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                     path_effects=[pe.withStroke(linewidth=1, foreground='white')])
        
        if move_info:
            info_text = f"Type: {move_info['type']}, UL: {move_info['unit_load_id']}, From: {move_info['from_position']} → To: {move_info['to_position']}"
            plt.text(0.5, warehouse.shape[0] + 0.5, info_text,
                     fontsize=10, ha='left', va='bottom', color='black',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                     path_effects=[pe.withStroke(linewidth=1, foreground='white')])

        filename = "initial_state.png" if initial_state else f"move_{move_step:03d}.png"
        # Removing bbox_inches='tight' to ensure consistent image dimensions for GIF creation
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    base_name = os.path.basename(file).split('.')[0]
    output_dir = os.path.join(os.path.dirname(file), f"{base_name}_astar")
    
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"🗑️  Cleaned up old A* visualization directory: {os.path.basename(output_dir)}")
    
    os.makedirs(output_dir, exist_ok=True)

    # --- MAIN VISUALIZATION LOOP ---
    for i, state in enumerate(solution_states):
        buffer_state = state.get('buffer_state', [])
        ul_positions = {}
        for ap_info in buffer_state:
            ap_id = ap_info['ap_id']
            stacks = ap_info.get('stacks', [])
            n_slots = len(stacks)
            for slot_index, ul_data in enumerate(stacks):
                ul_id = None
                priority = None
                if isinstance(ul_data, list):
                    ul_id = ul_data[0]
                    priority = ul_data[1]
                elif isinstance(ul_data, int) and ul_data != 0:
                    ul_id = ul_data
                
                if ul_id is not None:
                    # Convert stack index to tier: tier 1 is at highest index (back/deepest)
                    # tier = n_slots - slot_index
                    slot_tier = n_slots - slot_index
                    coords = ap_id_to_coords(ap_id, slot_tier, data)
                    if coords:
                        if coords not in ul_positions:
                            ul_positions[coords] = []
                        ul_positions[coords].append({'id': ul_id, 'priority': priority})
        
        move_step = i
        if i == 0:
            plot_warehouse_astar(0, copy.deepcopy(ul_positions), initial_state=True)
            move_info = {'type': 'Initial', 'unit_load_id': 'N/A', 'from_position': 'N/A', 'to_position': 'N/A'}
            plot_warehouse_astar(0, copy.deepcopy(ul_positions), move_info=move_info)
        else:
            if i - 1 < len(move_sequence):
                move_info = move_sequence[i - 1]
                move_step = move_info['step']
                plot_warehouse_astar(move_step, copy.deepcopy(ul_positions), move_info=move_info)
            else:
                 print(f"Warning: Mismatch between solution_states ({len(solution_states)}) and move_sequence ({len(move_sequence)}).")
                 plot_warehouse_astar(move_step, copy.deepcopy(ul_positions))

    move_files = sorted(
        [f for f in os.listdir(output_dir) if re.match(r'move_\d+\.png', f)],
        key=lambda x: int(re.search(r'move_(\d+)', x).group(1))
    )
    
    if move_files:
        imageio.mimsave(os.path.join(output_dir, "astar_moves.gif"), 
                        [imageio.imread(os.path.join(output_dir, img)) for img in move_files], fps=1)
        print(f"✅ Saved A* visualization in {output_dir}")
        print(f"   Generated initial state + {len(solution_states) - 1} A* move visualizations")
    else:
        print(f"⚠️ No A* move images found in {output_dir} to create a GIF.")

if __name__ == "__main__":
    if file is None:
        print("Please provide a JSON file with the --file argument.")
        exit(1)
    
    if not os.path.exists(file):
        print(f"Error: File '{file}' does not exist.")
        exit(1)
        
    process_file(file)