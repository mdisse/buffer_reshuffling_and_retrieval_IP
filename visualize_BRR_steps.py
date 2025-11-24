import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import re
import argparse
import math
import imageio.v2 as imageio # For creating GIFs. Ensure this package is installed in your environment.
import copy # For deep copying data structures
import matplotlib.patheffects as pe # For text outlines
from matplotlib import cm # Ensure cm is imported

# matplotlib.use('TkAgg') # Make sure this is commented out if using Agg backend

MAX_URGENCY_WINDOW_DEFAULT = 30 # Default window for urgency coloring

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--no-source', dest='source', action='store_false')
parser.set_defaults(source=True)
parser.add_argument('--color', dest='color_vls', action='store_true')
parser.set_defaults(color_vls=False)
parser.add_argument('--all', dest='all', action='store_true')
parser.set_defaults(all=False)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.set_defaults(debug=False)
parser.add_argument('--subplot-timesteps', type=str, default="", help="Comma-separated list of timesteps to include in a combined subplot figure.") # Added argument
parser.add_argument('--max_urgency_window', type=int, help='Maximum window (in time steps) for unit load retrieval urgency coloring. ULs with retrieval_end within this window from current_time_step will be colored with red-orange-yellow gradient based on urgency. ULs beyond this window will be light grey. ULs due or past due will be dark red.')
parser.add_argument('--overwrite', dest='overwrite', action='store_true', help='Overwrite existing visualization directory if it exists.')
parser.set_defaults(overwrite=False)
args = parser.parse_args()
source = args.source
color_vls = args.color_vls
debug = args.debug
all_experiments = args.all
file = args.file
subplot_timesteps_arg = args.subplot_timesteps
overwrite = args.overwrite

def get_ul_color(ul_id, current_time_step, ul_retrieval_times_map, max_urgency_window):
    """Calculates the color for a unit load based on its retrieval urgency.
    
    Color scheme:
    - Due/overdue: Dark red (high urgency)
    - Within urgency window: Red → Orange → Yellow (gradient based on urgency)
    - Not urgent yet: Light grey (low urgency)
    - Empty/null: Light grey
    """
    if ul_id is None or ul_id == 0:
        return (0.85, 0.85, 0.85) # Light grey for empty or UL ID 0

    retrieval_end = ul_retrieval_times_map.get(ul_id)

    if retrieval_end is None:
        # Default color if UL ID not in map (e.g. an AMR is carrying a new UL not in initial list)
        return (0.6, 0.6, 0.6) # Medium grey

    time_difference = retrieval_end - current_time_step

    if time_difference <= 0: # Due or overdue
        return (0.8, 0.0, 0.0) # Dark red - highest urgency
    elif time_difference > max_urgency_window: # Not urgent yet
        return (0.75, 0.75, 0.75) # Light grey - lowest urgency
    else: # Within the urgency window (0 < time_difference <= max_urgency_window)
        # Create gradient from red (urgent) to yellow (less urgent)
        # Ensure max_urgency_window is not zero to avoid division by zero
        if max_urgency_window == 0:
            normalized_urgency = 0 # Treat as most urgent if window is zero
        else:
            normalized_urgency = time_difference / max_urgency_window # Ranges from close to 0 to 1
        
        # Color gradient: Red (0,0) → Orange (0.5) → Yellow (1.0)
        if normalized_urgency <= 0.5:
            # Red to Orange: increase green component
            red = 1.0
            green = normalized_urgency * 2.0 # 0 to 1
            blue = 0.0
        else:
            # Orange to Yellow: maintain red=1, increase green to 1
            red = 1.0
            green = 1.0
            blue = 0.0
        
        return (red, green, blue)

def process_file(file):
    # Load JSON data
    with open(file, 'r') as f:
        data = json.load(f)

    # Create a map of UL ID to retrieval_end time
    ul_retrieval_times_map = {ul['id']: ul['retrieval_end'] 
                              for ul in data.get('unit_loads', []) 
                              if isinstance(ul, dict) and 'id' in ul and 'retrieval_end' in ul}
    
    # Get max_urgency_window from args or use default
    max_urgency_window = args.max_urgency_window if args.max_urgency_window is not None else MAX_URGENCY_WINDOW_DEFAULT

    # Get layout file path from JSON
    layout_file = data["layout_file"]
    warehouse = np.loadtxt(layout_file, delimiter=',', dtype=int)
    colors = {
        -5: 'lightgrey', # Aisles
        1: 'white', # Slots
        2: 'green', # Sink
        3: 'blue', # Source
        "ap": "red", # Access Points
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
    init_dict = {}
    # Populate init_dict once, mapping (global_y, global_x) to ul_id
    for bay_key, bay_config in bays.items():
        size = int(bay_key[:1])
        pattern = r"row (\d+), column (\d+)"
        match = re.search(pattern, bay_key)
        if match:
            bay_layout_row = int(match.group(1)) # y-coordinate in layout
            bay_layout_col = int(match.group(2)) # x-coordinate in layout
            for i_local_row in range(size):
                for j_local_col in range(size):
                    global_y = bay_layout_row + i_local_row
                    global_x = bay_layout_col + j_local_col
                    # bay_config is [row][col][ul_id_in_list]
                    ul_id_val = bay_config[i_local_row][j_local_col][0]
                    init_dict[(global_y, global_x)] = ul_id_val

    def ap_id_to_coords(ap_id, slot):
        """Converts an access point ID and tier/slot to its (x, y) coordinates.
        
        Tier numbering: Tier 1 = BACK/DEEPEST, Tier N = FRONT/CLOSEST
        The 'slot' parameter represents the tier number from decision strings like [11, 1].
        For each direction, we calculate position based on total lane length.
        """
        vl_length = next((lane['n_slots'] for lane in data['virtual_lanes'] if lane['ap_id'] == ap_id), None)
        for ap in data['access_points']:
            if ap['ap_id'] == ap_id:
                if ap['direction'] == 'north':
                    # North: lane extends upward (increasing y), AP at bottom
                    # Tier 1 (back) is farthest up, Tier N (front) is closest to AP
                    return ap['global_x'], ap['global_y'] + 1 + vl_length - slot
                elif ap['direction'] == 'south':
                    # South: lane extends downward (decreasing y), AP at top
                    # Tier 1 (back) is farthest down, Tier N (front) is closest to AP
                    return ap['global_x'], ap['global_y'] - 1 - vl_length + slot
                elif ap['direction'] == 'east':
                    # East: lane extends leftward (decreasing x), AP at right
                    # Tier 1 (back) is farthest left, Tier N (front) is closest to AP
                    return ap['global_x'] - 1 - vl_length + slot, ap['global_y']
                elif ap['direction'] == 'west':
                    # West: lane extends rightward (increasing x), AP at left
                    # Tier 1 (back) is farthest right, Tier N (front) is closest to AP
                    return ap['global_x'] + 1 + vl_length - slot, ap['global_y']
        return None

    def plot_warehouse(time_step, amr_positions, ul_positions, ul_retrieval_times_map_func_arg, max_urgency_window_func_arg, decision_texts=None, initial_state=False):
        plt.figure()

        virtual_lanes = {} # Initialize virtual_lanes here
        if color_vls: # Populate if color_vls is true
            for bay in data["bay_info"].values():
                coordinates = []
                for col in range(bay["length"]):
                    for row in range(bay["width"]):
                        coordinates.append((bay["x"] + col, bay["y"] + row))
            for coordinate in coordinates:
                x = coordinate[0]
                y = coordinate[1]
                for ap in data["access_points"]:
                    if ap["direction"] == "north":
                        ap_y = ap["global_y"] + 1
                        ap_x = ap["global_x"]
                    elif ap["direction"] == "south":
                        ap_y = ap["global_y"] - 1
                        ap_x = ap["global_x"]
                    elif ap["direction"] == "east":
                        ap_y = ap["global_y"]
                        ap_x = ap["global_x"] - 1
                    elif ap["direction"] == "west":
                        ap_y = ap["global_y"]
                        ap_x = ap["global_x"] + 1
                    if ap_x == x and ap_y == y:
                        virtual_lanes[(x, y)] = {}
                        virtual_lanes[(x, y)]["ap_id"] = ap["ap_id"]
                        virtual_lanes[(x, y)]["access_direction"] = ap["direction"]
            additional_lanes = {}
            for vl in list(virtual_lanes.keys()): # Use list() to avoid runtime error for changing dict size
                ap_id = virtual_lanes[vl]["ap_id"]
                for lane in data["virtual_lanes"]:
                    if lane["ap_id"] == ap_id:
                        for slot in range(lane["n_slots"]-1):
                            x = vl[0]
                            y = vl[1]
                            if virtual_lanes[vl]["access_direction"] == "north":
                                y += (1 + slot)
                            elif virtual_lanes[vl]["access_direction"] == "south":
                                y -= (1 + slot)
                            elif virtual_lanes[vl]["access_direction"] == "east":
                                x -= (1 + slot)
                            elif virtual_lanes[vl]["access_direction"] == "west":
                                x += (1 + slot)
                            additional_lanes[(x, y)] = virtual_lanes[vl]
            virtual_lanes.update(additional_lanes)

        def get_color(row, col):
            # For initial_state, do not color virtual lanes, use default slot color
            if color_vls and warehouse[row, col] == 1 and not initial_state:
                # Check if the coordinate exists in virtual_lanes before accessing it
                vl_info = virtual_lanes.get((col, row))
                if vl_info:
                    return vl_colors.get(vl_info["access_direction"], 'white')
                else:
                    return 'white' # Default color if coordinate not in virtual_lanes
            else:
                try:
                    return colors[warehouse[row, col]]
                except KeyError:
                    # Check if the key exists before raising the error
                    if warehouse[row, col] == 3: # Assuming 3 is the source
                         raise KeyError(f"This instance file contains a source, please use this script without the --no-source flag")
                    else:
                         return 'gray' # Or some other default color for unexpected values

        # Color cells
        for row in range(warehouse.shape[0]):
            for col in range(warehouse.shape[1]):
                # For debugging, print the slot coordinates
                if debug:
                    plt.text(col + 0.5, row + 0.5, f"{(row, col)}", color='black', ha='center', va='center')
                cell_color = get_color(row, col)
                plt.gca().add_patch(plt.Rectangle((col, row), 1, 1, color=cell_color))
                # Removed initial UL drawing logic from here


        # Bay borders
        linewidth = 0.025
        for bay, config in bays.items():
            size = int(bay[:1])
            pattern = r"row (\d+), column (\d+)"
            match = re.search(pattern, bay)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))

                # Draw top and bottom borders
                plt.gca().add_patch(plt.Rectangle((col, row), size, linewidth, color='black'))  # Top
                plt.gca().add_patch(plt.Rectangle((col, row + size - linewidth), size, linewidth, color='black'))  # Bottom
                # Draw left and right borders
                plt.gca().add_patch(plt.Rectangle((col, row), linewidth, size, color='black'))  # Left
                plt.gca().add_patch(plt.Rectangle((col + size - linewidth, row), linewidth, size, color='black'))  # Right

        # Plot access points
        access_points = data['access_points']
        aps = [access_point['ap_id'] for access_point in access_points]
        used_aps = []
        for lane in data['virtual_lanes']:
            used_aps.append(lane['ap_id'])
        hide_aps = [ap for ap in aps if ap not in used_aps]

        if not source:
            try:
                source_aps = list(data['source_info'].values())[0]['access_point_ids']
            except:
                source_aps = []

        # Remove the access points that are not used and written in hide_aps
        access_points = [access_point for access_point in access_points if access_point["ap_id"] not in hide_aps]

        for access_point in access_points:
            if not source and access_point["ap_id"] in source_aps:
                continue
            x_coord = access_point["global_x"]
            y_coord = access_point["global_y"]
            direction = access_point["direction"]
            ap_id = access_point["ap_id"]
            if direction == "north":
                y_coord += 1
                x_coord += 0.5
            elif direction == "south":
                x_coord += 0.5
            elif direction == "east":
                y_coord += 0.5
            elif direction == "west":
                y_coord += 0.5
                x_coord += 1
            if not initial_state:
                # Plot the AP (using a red circle here)
                ap_marker = plt.Circle((x_coord, y_coord), radius=0.1, color='red')
                plt.gca().add_patch(ap_marker)
                # Add label to the circle
                plt.text(x_coord, y_coord, ap_id, color='white', ha='center', va='center', fontsize=6)

        # Plot unit loads
        for (x, y), ul_ids in ul_positions.items():
            if ul_ids:  # Check if the list is not empty
                if ul_ids[0] is not None and ul_ids[0] != 0: # Ensure there's a UL and it's not '0'
                    ul_id = ul_ids[0]  # Visualize the top unit load
                    
                    current_ul_color = get_ul_color(ul_id, time_step, ul_retrieval_times_map_func_arg, max_urgency_window_func_arg)
                    
                    plt.gca().add_patch(plt.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, color=current_ul_color))
                    plt.text(x + 0.5, y + 0.5, f"{ul_id:02d}", color='white', ha='center', va='center',
                             path_effects=[pe.withStroke(linewidth=1, foreground='black')])

        # Plot AMRs
        if not initial_state:
            for amr_id, (x, y) in amr_positions.items():
                amr_marker = plt.Circle((x + 0.5, y + 0.5), radius=0.3, facecolor='orange', edgecolor='black', linewidth=1.5, alpha=1.0)  # Adjust color/alpha as needed
                plt.gca().add_patch(amr_marker)
                plt.text(x + 0.5, y + 1.0, amr_id, color='black', ha='center', va='center',
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

        plt.xticks(np.arange(0, warehouse.shape[1] + 1, 1), rotation='vertical', ha='center', labels=[])  # Gridlines on the x-axis
        plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
        plt.yticks(np.arange(0, warehouse.shape[0] + 1, 1), labels=[])  # Gridlines on the y-axis
        plt.gca().invert_yaxis()
        plt.grid(True, color='black', linewidth=.5)
        temp_legend = legend.copy()
        if initial_state:
            del temp_legend["ap"]
        plt.legend([plt.Rectangle((0,0),1,1, color=color) for color in colors.values()], temp_legend.values(), loc='center left')
        # legend_ul = plt.legend(handles, ul_labels, loc='lower left')
        # plt.gca().add_artist(legend_ul)
        plt.axis('scaled')
        plt.xlim(0, warehouse.shape[1])
        plt.ylim(warehouse.shape[0], 0)

        # Add timestep counter with 3 digits
        if not initial_state:
            plt.text(warehouse.shape[1] - 0.5, warehouse.shape[0] + 0.5, f"Timestep: {time_step:03d}",
                    fontsize=12, ha='right', va='bottom', color='black',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

        # Add decision text to the bottom left corner
        if decision_texts:
            for i, decision_text in enumerate(decision_texts):
                plt.text(0.5, warehouse.shape[0] + 0.5 + i * 0.5, f"{decision_text}",
                        fontsize=10, ha='left', va='bottom', color='black',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                        path_effects=[pe.withStroke(linewidth=1, foreground='white')])

        # Save the figure in the subdirectory
        if initial_state:
            plt.savefig(os.path.join(output_dir, f"initial_state.png"), dpi=300)
        else:
            plt.savefig(os.path.join(output_dir, f"{time_step}.png"), dpi=300)
        plt.close()  # Close the figure to free up memory

    # Initialize AMR and unit load positions
    # Initialize AMR positions at the sink
    sink_position = None
    for row in range(warehouse.shape[0]):
        for col in range(warehouse.shape[1]):
            if warehouse[row, col] == 2:  # Sink is represented by 2
                sink_position = (col, row)
                break
        if sink_position:
            break

    amr_positions = {f"v{i+1}": sink_position for i in range(data['fleet_size'])}
    
    # Create subdirectory, removing old one if it exists
    output_dir = os.path.join(os.path.dirname(file), os.path.basename(file).split('.')[0])
    
    # Check if visualization already exists
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        if not overwrite:
            print(f"⏭️  Skipping existing visualization: {os.path.basename(output_dir)}")
            return
        else:
            # Clean up old visualization directory only if overwrite is True
            import shutil
            shutil.rmtree(output_dir)
            print(f"🗑️  Cleaned up old visualization directory: {os.path.basename(output_dir)}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Plot initial state (initial_state.png)
    initial_ul_positions = {}
    for (global_y, global_x), ul_id_val in init_dict.items(): # Changed ul_id to ul_id_val to avoid conflict
        if ul_id_val != 0:
            initial_ul_positions[(global_x, global_y)] = [ul_id_val]
        else:
            initial_ul_positions[(global_x, global_y)] = [] # Keep empty lists for all slots
    plot_warehouse(0, {}, initial_ul_positions, ul_retrieval_times_map, max_urgency_window, initial_state=True)

    # Populate ul_positions for 0.png and subsequent simulation steps
    ul_positions = {}
    for (global_y, global_x), ul_id in init_dict.items():
        if ul_id != 0:
            ul_positions[(global_x, global_y)] = [ul_id]
        else:
            ul_positions[(global_x, global_y)] = []

    # Plot timestep 0 (0.png) (with AMRs and AP IDs, and correct ULs)
    plot_warehouse(0, amr_positions.copy(), ul_positions.copy(), ul_retrieval_times_map, max_urgency_window)
    
    target_subplot_timesteps = []
    if subplot_timesteps_arg:
        try:
            target_subplot_timesteps = [int(ts.strip()) for ts in subplot_timesteps_arg.split(',')]
        except ValueError:
            print("Warning: Invalid format for --subplot-timesteps. Please use comma-separated integers.")
            # Keep target_subplot_timesteps empty if parsing fails, so no subplots are generated.
            target_subplot_timesteps = []

    subplot_frames_data = [] # Initialize list to store data for subplots

    # If t=0 is a target for subplots, collect its data
    if 0 in target_subplot_timesteps:
        subplot_frames_data.append({
            "time_step": 0,
            "amr_positions": amr_positions.copy(),
            "ul_positions": copy.deepcopy(ul_positions), # Changed to deepcopy
            "decision_texts": [] # No decisions at t=0 for the plot itself
        })

    # Plot subsequent time steps
    # Concatenate decisions of all vehicles
    all_decisions = []
    for vehicle in data['results']['decisions'].keys():
        for t, decision in data['results']['decisions'][vehicle].items():
            all_decisions.append((int(t), vehicle, decision))

    # Sort decisions by time steps
    all_decisions.sort(key=lambda x: x[0])


    # Loop over sorted decisions
    current_time_step = -1
    current_decisions = []

    for t, vehicle, decision in all_decisions:
        if t != current_time_step:
            if current_time_step != -1:
                # ALWAYS plot individual warehouse state
                plot_warehouse(current_time_step, amr_positions.copy(), ul_positions.copy(), ul_retrieval_times_map, max_urgency_window, decision_texts=current_decisions.copy())
                
                # If current_time_step is a target for subplots, collect its data
                if target_subplot_timesteps and current_time_step in target_subplot_timesteps:
                    # Avoid duplicates if already added (e.g. if t=0 was processed and is also in decisions)
                    if not any(frame["time_step"] == current_time_step for frame in subplot_frames_data):
                        subplot_frames_data.append({
                            "time_step": current_time_step,
                            "amr_positions": amr_positions.copy(),
                            "ul_positions": copy.deepcopy(ul_positions), # Changed to deepcopy
                            "decision_texts": current_decisions.copy()
                        })
            current_time_step = t
            current_decisions = []

        current_decisions.append(decision["decision"])

        pattern = r"\[(\d+), (\d+)\].*\[(\d+), (\d+)\]"
        match = re.search(pattern, decision["move"])
        decision_type = decision["decision"][0]
        if match:
            i = int(match.group(1))  # Origin AP
            j = int(match.group(2))  # Slot
            k = int(match.group(3))  # Destination AP
            l = int(match.group(4))  # Slot
            x1, y1 = ap_id_to_coords(i, j)
            x2, y2 = ap_id_to_coords(k, l)

            if decision_type == "e":  # Empty repositioning
                amr_positions[vehicle] = (x2, y2)
            elif decision_type == "x":  # Load transfer
                pattern_ul = r"n(\d+)"
                match_ul = re.search(pattern_ul, decision["decision"])
                if match_ul:
                    ul_id = int(match_ul.group(1))
                    ul_positions[(x2, y2)].append(ul_positions[(x1, y1)].pop(0))  # Move the unit load to the end
                    amr_positions[vehicle] = (x2, y2)
            elif decision_type == "y":  # Retrieval
                if (x1, y1) in ul_positions and ul_positions[(x1, y1)]:  # Check if the list is not empty
                    ul_positions[(x1, y1)].pop(0)  # Remove the top unit load
                amr_positions[vehicle] = (x2, y2)
            elif decision_type == "z":  # Storage
                pattern_ul = r"n(\d+)"
                match_ul = re.search(pattern_ul, decision["decision"])
                if match_ul:
                    ul_id = int(match_ul.group(1))
                    ul_positions[(x2, y2)].append(ul_id)  # Add new unit load to the end
                    amr_positions[vehicle] = (x2, y2)

    # Capture the last time step
    if current_time_step != -1:
        # ALWAYS plot individual warehouse state for the last step
        plot_warehouse(current_time_step, amr_positions.copy(), ul_positions.copy(), ul_retrieval_times_map, max_urgency_window, decision_texts=current_decisions.copy())

        # If the last time_step is a target for subplots, collect its data
        if target_subplot_timesteps and current_time_step in target_subplot_timesteps:
            if not any(frame["time_step"] == current_time_step for frame in subplot_frames_data):
                subplot_frames_data.append({
                    "time_step": current_time_step,
                    "amr_positions": amr_positions.copy(),
                    "ul_positions": copy.deepcopy(ul_positions), # Changed to deepcopy
                    "decision_texts": current_decisions.copy()
                })

    if target_subplot_timesteps and subplot_frames_data:
        # Sort frames by timestep just in case they were added out of order (e.g. t=0)
        subplot_frames_data.sort(key=lambda x: x["time_step"])
        
        if subplot_frames_data:
            num_plots = len(subplot_frames_data)
            if num_plots == 0:
                return # Or handle as an error/warning

            # Determine the grid layout for subplots
            # Original taller layout (commented out for clarity):
            rows = math.ceil(math.sqrt(num_plots)) # Aim for a layout that's taller or square
            cols = math.ceil(num_plots / rows)
            
            # Adjust figsize based on the new layout
            fig_width_per_subplot = 6  # Inches per subplot
            fig_height_per_subplot = 5 # Inches per subplot
            
            fig_width = cols * fig_width_per_subplot
            fig_height = rows * fig_height_per_subplot
            
            # Ensure a minimum overall figure dimension for aesthetics, especially for few plots
            min_overall_width = 6 # Minimum width for the entire figure in inches
            min_overall_height = 5 # Minimum height for the entire figure in inches
            fig_width = max(fig_width, min_overall_width)
            fig_height = max(fig_height, min_overall_height)

            fig, axs = plt.subplots(rows, cols, figsize=(fig_width, fig_height)) # Create subplots with new dimensions
            axs_flat = axs.flatten() if num_plots > 1 else [axs] # Handle single plot case

            for i, frame_data in enumerate(subplot_frames_data):
                ax = axs_flat[i]
                # Add a lettered tag as the title for each subplot
                ax.set_title(f"({chr(ord('a') + i)})", loc='left', fontsize='medium')

                # Define flags for initial state and legend display for the current subplot
                is_true_initial_for_subplot = (frame_data["time_step"] == 0 and 
                                               not frame_data["decision_texts"] and 
                                               not frame_data["amr_positions"])
                show_legend_for_this_subplot = (frame_data["time_step"] == 0)

                # Call the new function to plot on the specific subplot axis
                plot_warehouse_to_ax(ax, frame_data["time_step"], frame_data["amr_positions"], 
                                     frame_data["ul_positions"], data, warehouse, colors, vl_colors, 
                                     color_vls, source, debug, legend, 
                                     ul_retrieval_times_map, max_urgency_window, # Pass the map and window
                                     decision_texts=frame_data["decision_texts"], 
                                     initial_state=is_true_initial_for_subplot, 
                                     show_legend_in_subplot=show_legend_for_this_subplot)
            
            # Hide any unused subplots
            for j in range(num_plots, rows * cols):
                fig.delaxes(axs_flat[j])

            # Adjust spacing between subplots
            fig.subplots_adjust(hspace=0.6, wspace=0.35) # Increased hspace for more vertical room

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap and make space for suptitle
            # Create a title for the entire figure
            title_timesteps = sorted(list(s["time_step"] for s in subplot_frames_data))
            fig.suptitle(f"Warehouse States at Timesteps: {', '.join(map(str, title_timesteps))}", fontsize=16)
            fig.suptitle("")
            
            subplot_filename = os.path.join(output_dir, f"subplots_{'_'.join(map(str, title_timesteps))}.png")
            plt.savefig(subplot_filename, dpi=300)
            plt.close(fig) # Close the subplot figure
            print(f"Saved subplot figure to {subplot_filename}")

    # Create a video from the images
    images = []
    for image_filename in os.listdir(output_dir):
        if image_filename.endswith(".png") and \
           image_filename != "initial_state.png" and \
           not image_filename.startswith("subplots_"): # Exclude subplot images
            images.append(image_filename)
    
    if images: # Only proceed if there are images to make a GIF from
        images.sort(key=lambda x: int(x.split('.')[0]))
        imageio.mimsave(os.path.join(output_dir, "simulation.gif"), [imageio.imread(os.path.join(output_dir, img)) for img in images], fps=1)
        print(f"Saved visualization in {output_dir}")
    elif not target_subplot_timesteps: # If no subplots were made and no other images, print a different message or handle as needed
        print(f"No individual step images found in {output_dir} to create a GIF.")
    # If only subplots were created, the GIF part is skipped, which is fine.

def plot_warehouse_to_ax(ax, time_step, amr_positions, ul_positions,
                         data_func_arg, warehouse, colors, vl_colors, color_vls, source, debug, legend,
                         ul_retrieval_times_map_func_arg, max_urgency_window_func_arg,
                         decision_texts=None, initial_state=False, show_legend_in_subplot=True):
    """Plots the warehouse state onto a given Matplotlib Axes object."""
    bays = data_func_arg["initial_state"]

    virtual_lanes = {}
    if color_vls:
        for bay in data_func_arg["bay_info"].values():
            coordinates = []
            for col in range(bay["length"]):
                for row_val in range(bay["width"]):
                    coordinates.append((bay["x"] + col, bay["y"] + row_val))
            for coordinate in coordinates:
                x = coordinate[0]
                y = coordinate[1]
                for ap in data_func_arg["access_points"]:
                    if ap["direction"] == "north":
                        ap_y = ap["global_y"] + 1
                        ap_x = ap["global_x"]
                    elif ap["direction"] == "south":
                        ap_y = ap["global_y"] - 1
                        ap_x = ap["global_x"]
                    elif ap["direction"] == "east":
                        ap_y = ap["global_y"]
                        ap_x = ap["global_x"] - 1
                    elif ap["direction"] == "west":
                        ap_y = ap["global_y"]
                        ap_x = ap["global_x"] + 1
                    if ap_x == x and ap_y == y:
                        virtual_lanes[(x, y)] = {}
                        virtual_lanes[(x, y)]["ap_id"] = ap["ap_id"]
                        virtual_lanes[(x, y)]["access_direction"] = ap["direction"]
        additional_lanes = {}
        for vl in list(virtual_lanes.keys()):
            ap_id = virtual_lanes[vl]["ap_id"]
            for lane in data_func_arg["virtual_lanes"]:
                if lane["ap_id"] == ap_id:
                    for slot in range(lane["n_slots"]-1):
                        x = vl[0]
                        y = vl[1]
                        if virtual_lanes[vl]["access_direction"] == "north":
                            y += (1 + slot)
                        elif virtual_lanes[vl]["access_direction"] == "south":
                            y -= (1 + slot)
                        elif virtual_lanes[vl]["access_direction"] == "east":
                            x -= (1 + slot)
                        elif virtual_lanes[vl]["access_direction"] == "west":
                            x += (1 + slot)
                        additional_lanes[(x, y)] = virtual_lanes[vl]
        virtual_lanes.update(additional_lanes)

    def get_color_ax(row_val, col_val):
        if color_vls and warehouse[row_val, col_val] == 1 and not initial_state:
            vl_info = virtual_lanes.get((col_val, row_val))
            if vl_info:
                return vl_colors.get(vl_info["access_direction"], 'white')
            else:
                return 'white'
        else:
            try:
                return colors[warehouse[row_val, col_val]]
            except KeyError:
                if warehouse[row_val, col_val] == 3:
                     raise KeyError(f"This instance file contains a source, please use this script without the --no-source flag")
                else:
                     return 'gray'

    # Color cells
    for row_idx in range(warehouse.shape[0]):
        for col_idx in range(warehouse.shape[1]):
            if debug:
                ax.text(col_idx + 0.5, row_idx + 0.5, f"{(row_idx, col_idx)}", color='black', ha='center', va='center', fontsize=5)
            cell_color = get_color_ax(row_idx, col_idx)
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, color=cell_color))

    # Bay borders
    linewidth = 0.025
    for bay, config in bays.items():
        size = int(bay[:1])
        pattern = r"row (\d+), column (\d+)"
        match = re.search(pattern, bay)
        if match:
            row_val = int(match.group(1))
            col_val = int(match.group(2))
            ax.add_patch(plt.Rectangle((col_val, row_val), size, linewidth, color='black'))
            ax.add_patch(plt.Rectangle((col_val, row_val + size - linewidth), size, linewidth, color='black'))
            ax.add_patch(plt.Rectangle((col_val, row_val), linewidth, size, color='black'))
            ax.add_patch(plt.Rectangle((col_val + size - linewidth, row_val), linewidth, size, color='black'))

    # Plot access points
    access_points = data_func_arg['access_points']
    aps = [access_point['ap_id'] for access_point in access_points]
    used_aps = [lane['ap_id'] for lane in data_func_arg['virtual_lanes']]
    hide_aps = [ap for ap in aps if ap not in used_aps]

    source_aps = []
    if not source:
        try:
            source_aps = list(data_func_arg['source_info'].values())[0]['access_point_ids']
        except (KeyError, IndexError):
            source_aps = []

    access_points_to_plot = [ap for ap in access_points if ap["ap_id"] not in hide_aps]

    for access_point in access_points_to_plot:
        if not source and access_point["ap_id"] in source_aps:
            continue
        x_coord = access_point["global_x"]
        y_coord = access_point["global_y"]
        direction = access_point["direction"]
        ap_id = access_point["ap_id"]
        if direction == "north":
            y_coord += 1
            x_coord += 0.5
        elif direction == "south":
            x_coord += 0.5
        elif direction == "east":
            y_coord += 0.5
        elif direction == "west":
            y_coord += 0.5
            x_coord += 1
        if not initial_state:
            ap_marker = plt.Circle((x_coord, y_coord), radius=0.1, color='red')
            ax.add_patch(ap_marker)
            ax.text(x_coord, y_coord, ap_id, color='white', ha='center', va='center', fontsize=8) 

    # Plot unit loads
    for (x, y), ul_ids in ul_positions.items():
        if ul_ids and ul_ids[0] is not None and ul_ids[0] != 0:
            ul_id = ul_ids[0]
            current_ul_color = get_ul_color(ul_id, time_step, ul_retrieval_times_map_func_arg, max_urgency_window_func_arg)
            ax.add_patch(plt.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, color=current_ul_color))
            ax.text(x + 0.5, y + 0.5, f"{ul_id:02d}", color='white', ha='center', va='center',
                    fontsize=10, path_effects=[pe.withStroke(linewidth=1, foreground='black')]) 

    # Plot AMRs
    if not initial_state:
        for amr_id, (amr_x, amr_y) in amr_positions.items():
            amr_marker = plt.Circle((amr_x + 0.5, amr_y + 0.5), radius=0.3, facecolor='orange', edgecolor='black', linewidth=1.5, alpha=1.0)
            ax.add_patch(amr_marker)
            ax.text(amr_x + 0.5, amr_y + 1.0, amr_id, color='black', ha='center', va='center', fontsize=10, 
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    ax.set_xticks(np.arange(0, warehouse.shape[1] + 1, 1))
    ax.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False, labelcolor='none')
    ax.set_yticks(np.arange(0, warehouse.shape[0] + 1, 1))
    ax.tick_params(axis='y', labelcolor='none')

    ax.invert_yaxis()
    ax.grid(True, color='black', linewidth=.5)

    if show_legend_in_subplot:
        temp_legend = legend.copy()
        if initial_state:
            if "ap" in temp_legend: del temp_legend["ap"]
        legend_handles = [plt.Rectangle((0,0),1,1, color=colors[key]) for key in temp_legend if key in colors]
        legend_labels = [temp_legend[key] for key in temp_legend if key in colors]
        if legend_handles:
             ax.legend(legend_handles, legend_labels, loc='center left', fontsize='small')

    ax.axis('scaled')
    ax.set_xlim(0, warehouse.shape[1])
    ax.set_ylim(warehouse.shape[0], 0)

    # Add timestep counter
    if not initial_state:
        ax.text(warehouse.shape[1] - 0.5, warehouse.shape[0] + 0.5, f"T: {time_step:03d}",
                fontsize=12, ha='right', va='bottom', color='black', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                path_effects=[pe.withStroke(linewidth=1, foreground='white')])

    # Add decision text
    if decision_texts:
        for i, decision_text in enumerate(decision_texts):
            ax.text(0.5, warehouse.shape[0] + 0.5 + i * 0.4, f"{decision_text}",
                    fontsize=9, ha='left', va='bottom', color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')])

if all_experiments:
    number = 0
    processed_files = set()
    results_file_path = os.path.join(os.path.dirname(__file__), 'experiments/hashesBRR/results.txt')
    with open(results_file_path, 'r') as results_file:
        for line in results_file:
            try:
                file_path = line.strip()
                process_file(file_path)
                if line.strip() not in processed_files:
                    processed_files.add(line.strip())
                    number += 1
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    print(f"Processed {number} files")
else:
    process_file(file)