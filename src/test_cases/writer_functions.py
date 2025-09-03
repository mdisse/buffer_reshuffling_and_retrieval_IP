import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

import json
import subprocess
import re
from math import ceil


def save_resultsBrr(filename: str, test_case):
    """
    Saves the results of the Gurobi experiment as a json file.
    """
    data = dict()
    data['layout_file'] = test_case.instance.get_layout_file()
    data["access_directions"] = test_case.instance.access_directions
    data['fill_level'] = round(test_case.instance.get_fill_level(), 2)
    data['height'] = test_case.instance.get_height()
    data['seed'] = test_case.instance.get_seed()
    data['fleet_size'] = test_case.instance.get_fleet_size()
    data['vehicle_speed'] = test_case.instance.get_vehicle_speed()
    data['handling_time'] = test_case.instance.get_handling_time()
    data['unit_loads'] = []
    for ul in test_case.instance.unit_loads:
        data['unit_loads'].append(ul.to_data_dict())

    data['results'] = test_case.results

    data['bay_info'] = dict()
    data['sink_info'] = dict()
    data['source_info'] = dict()
    data['initial_state'] = dict()
    for source in test_case.instance.wh_initial.sources: 
        data['source_info'][source.get_id()] = source.to_data_dict()
    for bay in test_case.instance.wh_initial.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['initial_state'][bay.get_id()] = bay.state.tolist()
    for sink in test_case.instance.wh_initial.sinks: 
        data['sink_info'][sink.get_id()] = sink.to_data_dict()

    data['access_points'] = []
    for point in test_case.instance.wh_initial.all_access_points:
        data['access_points'].append(point.to_data_dict())

    data['virtual_lanes'] = []
    for lane in test_case.instance.wh_initial.virtual_lanes:
        data['virtual_lanes'].append(lane.to_data_dict())

    f = open(filename, 'w')
    json.dump(data, f, indent=4)
    f.close()


def save_heuristic_results(filename: str, test_case):
    """
    Saves the results of the heuristic experiment as a json file.
    """
    data = dict()
    data['layout_file'] = test_case.instance.get_layout_file()
    data["access_directions"] = test_case.instance.access_directions
    data['fill_level'] = round(test_case.instance.get_fill_level(), 2)
    data['height'] = test_case.instance.get_height()
    data['seed'] = test_case.instance.get_seed()
    data['fleet_size'] = test_case.instance.get_fleet_size()
    data['vehicle_speed'] = test_case.instance.get_vehicle_speed()
    data['handling_time'] = test_case.instance.get_handling_time()
    
    # Unit loads
    data['unit_loads'] = []
    for ul in test_case.instance.unit_loads:
        data['unit_loads'].append(ul.to_data_dict())

    # Calculate heuristic objective if not already done
    if test_case.heuristic_objective is None:
        test_case.calculate_heuristic_objective()
    
    # Translate heuristic decisions to Gurobi format first
    translated_decisions = {}
    corrected_objective = test_case.heuristic_objective  # fallback
    
    if test_case.amr_assignments and 'vehicles' in test_case.amr_assignments:
        # Use a simpler approach without creating a full model
        translated_decisions = translate_heuristic_decisions_simple(test_case.amr_assignments, test_case.instance)
        
        # Recalculate objective using the real (Gurobi-style) distances from translated decisions
        corrected_objective = 0
        for vehicle_key, vehicle_decisions in translated_decisions.items():
            for time_key, decision_data in vehicle_decisions.items():
                corrected_objective += decision_data.get('distance', 0)
    
    # Validate the solution if we have one
    validation_status = -1
    validation_report = {}
    is_feasible = False
    
    if translated_decisions:
        try:
            # Use detailed validation
            is_feasible, validation_status, validation_report = validate_heuristic_solution_detailed(
                test_case.instance, test_case, verbose=True  # Force debug output
            )
            if test_case.verbose: 
                print(f"DEBUG save_heuristic_results: validation_report keys = {list(validation_report.keys())}")
                print(f"DEBUG save_heuristic_results: tardiness_info length = {len(validation_report.get('tardiness_info', []))}")
                print(f"DEBUG save_heuristic_results: total_tardiness = {validation_report.get('total_tardiness', 0)}")
        except Exception as e:
            validation_status = -1
            is_feasible = False
            validation_report = {
                "message": f"Validation error: {str(e)}",
                "violations": [{"type": "validation_error", "description": str(e)}]
            }
    else:
        # No solution to validate
        validation_status = -1
        is_feasible = False
        validation_report = {
            "message": "No solution found by heuristic",
            "violations": [{"type": "no_solution", "description": "Heuristic to find a solution"}]
        }
    
    # Results section matching Gurobi format  
    data['results'] = {
        'objective_value': corrected_objective,  # Use corrected objective based on real distances
        'runtime': test_case.heuristic_runtime if test_case.heuristic_runtime else 0.0,
        'mipgap': 0.0,  # Heuristic doesn't have MIP gap
        'decisions': translated_decisions,
        'astar_result': {
            'move_sequence': [],
            'move_count': 0,
            'description': 'A* search result - optimal sequence of moves'
        },
        'validation': {
            'is_feasible': is_feasible,
            'status_code': validation_status,
            'message': validation_report.get('message', 'No validation performed'),
            'violations': validation_report.get('violations', []),
            'tardiness_info': validation_report.get('tardiness_info', []),
            'total_tardiness': validation_report.get('total_tardiness', 0),
            'num_late_unit_loads': validation_report.get('num_late_unit_loads', 0),
            'num_late_moves': validation_report.get('num_late_moves', 0),
            'details': validation_report
        }
    }
    
    # Debug what's being written
    if test_case.verbose: 
        print(f"DEBUG: About to write tardiness_info with {len(data['results']['validation']['tardiness_info'])} items")
        print(f"DEBUG: About to write total_tardiness = {data['results']['validation']['total_tardiness']}")
    
    # Add A* result information if available
    if hasattr(test_case, 'move_sequence') and test_case.move_sequence:
        move_sequence_data = []
        for i, move in enumerate(test_case.move_sequence):
            move_data = {
                'step': i + 1,
                'type': move.get('type', 'unknown'),
                'unit_load_id': move.get('ul_id', 0),
                # Keep the original position objects/strings from A*
                'from_position': str(move.get('from_pos', 'unknown')),
                'to_position': str(move.get('to_pos', 'unknown'))
            }
            move_sequence_data.append(move_data)
        
        data['results']['astar_result'] = {
            'move_sequence': move_sequence_data,
            'move_count': len(test_case.move_sequence),
            'description': 'A* search result - optimal sequence of moves determined before VRP assignment (all positions in [access_point, tier] format)'
        }
    
    # Add A* solution states if available
    if 'astar_solution_states' in test_case.results:
        data['results']['astar_result']['solution_states'] = test_case.results['astar_solution_states']

    # Buffer information
    data['bay_info'] = dict()
    data['sink_info'] = dict()
    data['source_info'] = dict()
    data['initial_state'] = dict()
    
    for source in test_case.instance.wh_initial.sources: 
        data['source_info'][source.get_id()] = source.to_data_dict()
    for bay in test_case.instance.wh_initial.bays:
        data['bay_info'][bay.get_id()] = bay.to_data_dict()
        data['initial_state'][bay.get_id()] = bay.state.tolist()
    for sink in test_case.instance.wh_initial.sinks: 
        data['sink_info'][sink.get_id()] = sink.to_data_dict()

    data['access_points'] = []
    for point in test_case.instance.wh_initial.all_access_points:
        data['access_points'].append(point.to_data_dict())

    data['virtual_lanes'] = []
    for lane in test_case.instance.wh_initial.virtual_lanes:
        data['virtual_lanes'].append(lane.to_data_dict())

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the file first
    f = open(filename, 'w')
    json.dump(data, f, indent=4)
    f.close()
    
    # Run Gurobi result checker if we have a solution
    if translated_decisions:
        try:
            gurobi_check = run_gurobi_result_checker(filename, verbose=False)
            
            # Also try to get VRP warnings from test_case if available
            vrp_warnings = []
            if hasattr(test_case, 'vrp_solution') and test_case.vrp_solution:
                vrp_warnings = test_case.vrp_solution.get('warnings', [])
            
            # Extract tardiness information from VRP warnings first (more accurate)
            tardiness_info = []
            for warning in vrp_warnings:
                if 'tardiness' in warning and warning['tardiness'] > 0:
                    tardiness_info.append({
                        'move_id': warning.get('move_id', 0),
                        'ul_id': warning.get('ul_id', 0),
                        'deadline': warning.get('deadline', 0),
                        'expected_finish': warning.get('expected_finish', 0),
                        'tardiness': warning.get('tardiness', 0),
                        'message': warning.get('message', '')
                    })
            
            # If no VRP warnings found, fall back to Gurobi check warnings
            if not vrp_warnings:
                for warning in gurobi_check.get('warnings', []):
                    if 'tardiness' in warning and warning['tardiness'] > 0:
                        tardiness_info.append({
                            'move_id': warning.get('move_id', 0),
                            'ul_id': warning.get('ul_id', 0),
                            'deadline': warning.get('deadline', 0),
                            'expected_finish': warning.get('expected_finish', 0),
                            'tardiness': warning.get('tardiness', 0),
                            'message': warning.get('message', '')
                        })
            
            # Calculate summary statistics for bulk analysis
            total_tardiness = sum(info['tardiness'] for info in tardiness_info)
            num_late_unit_loads = len(set(info['ul_id'] for info in tardiness_info))
            
            # Merge validation and heuristic information into a single validation section
            # Preserve the tardiness information from our detailed validation
            existing_tardiness_info = data['results']['validation'].get('tardiness_info', [])
            existing_total_tardiness = data['results']['validation'].get('total_tardiness', 0)
            existing_num_late_unit_loads = data['results']['validation'].get('num_late_unit_loads', 0)
            existing_num_late_moves = data['results']['validation'].get('num_late_moves', 0)
            
            # Only use VRP/Gurobi tardiness if our validation didn't find any
            if not existing_tardiness_info and tardiness_info:
                final_tardiness_info = tardiness_info
                final_total_tardiness = total_tardiness
                final_num_late_unit_loads = num_late_unit_loads
                final_num_late_moves = len(tardiness_info)
            else:
                # Prefer our validation results
                final_tardiness_info = existing_tardiness_info
                final_total_tardiness = existing_total_tardiness
                final_num_late_unit_loads = existing_num_late_unit_loads
                final_num_late_moves = existing_num_late_moves
            
            data['results']['validation'] = {
                # Gurobi validation fields
                'is_feasible': data['results']['validation']['is_feasible'],
                'status_code': data['results']['validation']['status_code'],
                'message': data['results']['validation']['message'],
                'violations': data['results']['validation']['violations'],
                
                # Tardiness information for detailed analysis (preserve validation results)
                'tardiness_info': final_tardiness_info,
                
                # Summary statistics for bulk analysis
                'total_tardiness': final_total_tardiness,
                'num_late_unit_loads': final_num_late_unit_loads,
                'num_late_moves': final_num_late_moves
            }
            
            # Save the file again with merged validation results
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Warning: Could not run Gurobi result checker: {e}")
    
    return corrected_objective


def generate_heuristic_result_path(instance_file_path: str, fleet_size_override=None) -> str:
    """
    Generate the heuristic result file path based on the instance file path.
    Converts from inputsBRR to resultsBRR and appends _heuristic suffix.
    Also adds fleet_size directory based on the instance data or override.
    """
    # Use override fleet size if provided, otherwise read from instance
    if fleet_size_override is not None:
        fleet_size = fleet_size_override
    else:
        try:
            with open(instance_file_path, 'r') as f:
                instance_data = json.load(f)
                fleet_size = instance_data.get('fleet_size', 1)
        except:
            fleet_size = 1
    
    # Replace inputsBRR with resultsBRR
    result_path = instance_file_path.replace('inputsBRR', 'resultsBRR')
    
    # Insert fleet_size directory before the filename
    path_parts = result_path.split('/')
    filename = path_parts[-1]
    directory = '/'.join(path_parts[:-1])
    
    # Replace .json with _heuristic.json
    if filename.endswith('.json'):
        filename = filename[:-5] + '_heuristic.json'
    
    result_path = f"{directory}/fleet_size_{fleet_size}/{filename}"
    
    return result_path


def generate_heuristic_filename(instance_file_path: str, fleet_size_override=None) -> str:
    """
    Generate the heuristic result filename based on the instance file path.
    Converts from inputsBRR to resultsBRR and appends _heuristic suffix.
    Also adds fleet_size directory based on the instance data or override.
    """
    # Use override fleet size if provided, otherwise read from instance
    if fleet_size_override is not None:
        fleet_size = fleet_size_override
    else:
        try:
            with open(instance_file_path, 'r') as f:
                instance_data = json.load(f)
                fleet_size = instance_data.get('fleet_size', 1)
        except:
            fleet_size = 1
    
    # Replace inputsBRR with resultsBRR
    result_path = instance_file_path.replace('inputsBRR', 'resultsBRR')
    
    # Insert fleet_size directory before the filename
    path_parts = result_path.split('/')
    filename = path_parts[-1]
    directory = '/'.join(path_parts[:-1])
    
    # Replace .json with _heuristic.json
    if filename.endswith('.json'):
        filename = filename[:-5] + '_heuristic.json'
    
    # result_path = f"{directory}/fleet_size_{fleet_size}/{filename}"
    result_path = f"{directory}/{filename}"
    
    return result_path


def translate_heuristic_decisions_simple(heuristic_decisions, instance):
    """
    Translates heuristic decisions into Gurobi decision variable format without creating a full model.
    
    Args:
        heuristic_decisions: Dictionary with AMR assignments from heuristic
        instance: Instance object for buffer information
        
    Returns:
        Dictionary in the same format as Gurobi results with proper decision variable names
    """
    translated_decisions = {}
    
    if not heuristic_decisions or 'vehicles' not in heuristic_decisions:
        return translated_decisions
    
    # Get buffer information
    lanes = instance.get_buffer().get_virtual_lanes()
    vehicle_speed = instance.get_vehicle_speed()
    
    # Helper function to parse location with tier support
    def parse_location(location, tier=1):
        """Parse location and return (lane_id, tier). 
        For source/sink, tier is always 1. For storage lanes, use provided tier."""
        if location == 'sink':
            # Find sink lane
            for lane in lanes:
                if hasattr(lane, 'is_sink') and getattr(lane, 'is_sink', False):
                    return lane.get_ap_id(), 1
            # Fallback: lane 16 (typical sink in this layout)
            return 16, 1
        elif location == 'source':
            # Find source lane  
            for lane in lanes:
                if hasattr(lane, 'is_source') and getattr(lane, 'is_source', False):
                    return lane.get_ap_id(), 1
            # Fallback: lane 0 (typical source in this layout)
            return 0, 1
        else:
            # Assume it's a lane ID and validate it exists in virtual lanes
            try:
                lane_id = int(location)
                # Check if this lane exists in the virtual lanes
                valid_lanes = set()
                for lane in lanes:
                    valid_lanes.add(lane.get_ap_id())
                    
                if lane_id in valid_lanes:
                    return lane_id, tier  # Use the provided tier instead of hardcoded 1
                else:
                    # Lane doesn't exist, this might be an error in the A* algorithm
                    # For now, find the actual lane where this unit load should be
                    # by checking the accessible unit loads
                    print(f"Warning: Referenced lane {lane_id} not found in virtual lanes. Available lanes: {sorted(valid_lanes)}")
                    # Return a default valid lane
                    return min(valid_lanes), tier  # Use the provided tier instead of hardcoded 1
                    
            except:
                return 0, 1
    
    # Helper function to calculate real distance using Gurobi method
    def calculate_real_distance(from_location, to_location, from_tier=1, to_tier=1):
        """Calculate distance using the same method as Gurobi models"""
        buffer = instance.get_buffer()
        
        # Parse locations
        if from_location == 'sink':
            from_lane_id = buffer.get_sink().get_ap_id()
            from_tier_id = from_tier
        elif from_location == 'source':
            from_lane_id = buffer.get_source().get_ap_id()
            from_tier_id = from_tier
        else:
            from_lane_id = int(from_location)
            from_tier_id = from_tier
            
        if to_location == 'sink':
            to_lane_id = buffer.get_sink().get_ap_id()
            to_tier_id = to_tier
        elif to_location == 'source':
            to_lane_id = buffer.get_source().get_ap_id()
            to_tier_id = to_tier
        else:
            to_lane_id = int(to_location)
            to_tier_id = to_tier
        
        # Calculate distance using Gurobi's method
        if to_location == 'sink':
            # Find the lane object
            from_lane = None
            for lane in lanes:
                if lane.get_ap_id() == from_lane_id:
                    from_lane = lane
                    break
            lane_distance = buffer.get_distance_sink(from_lane) if from_lane else 0
        elif to_location == 'source':
            # Find the lane object
            from_lane = None
            for lane in lanes:
                if lane.get_ap_id() == from_lane_id:
                    from_lane = lane
                    break
            lane_distance = buffer.get_distance_source(from_lane) if from_lane else 0
        elif from_location == 'sink':
            # Find the lane object
            to_lane = None
            for lane in lanes:
                if lane.get_ap_id() == to_lane_id:
                    to_lane = lane
                    break
            lane_distance = buffer.get_distance_sink(to_lane) if to_lane else 0
        elif from_location == 'source':
            # Find the lane object
            to_lane = None
            for lane in lanes:
                if lane.get_ap_id() == to_lane_id:
                    to_lane = lane
                    break
            lane_distance = buffer.get_distance_source(to_lane) if to_lane else 0
        else:
            # Lane to lane
            from_lane = None
            to_lane = None
            for lane in lanes:
                if lane.get_ap_id() == from_lane_id:
                    from_lane = lane
                if lane.get_ap_id() == to_lane_id:
                    to_lane = lane
            lane_distance = buffer.get_distance_lanes(from_lane, to_lane) if from_lane and to_lane else 0
        
        # Add tier distance (same as Gurobi)
        tier_distance = (from_tier_id - 1) + (to_tier_id - 1)
        
        return lane_distance + tier_distance
    
    # Helper function to calculate Gurobi-style travel time
    def calculate_gurobi_style_travel_time(distance, is_loaded_move):
        """Calculate travel time using the same method as Gurobi models"""
        handling_time = instance.get_handling_time()
        travel_time = ceil(distance / vehicle_speed)
        return travel_time + (2 * handling_time) if is_loaded_move else travel_time
    
    # Process each vehicle
    for vehicle in heuristic_decisions['vehicles']:
        vehicle_id = vehicle['vehicle_id']
        vehicle_key = f"v{vehicle_id}"  # Already 1-indexed
        translated_decisions[vehicle_key] = {}
        
        # Process each move for this vehicle
        for move_idx, move in enumerate(vehicle['moves']):
            from_location = move['from_location']
            move_type = move['move_type']
            ul_id = move['ul_id']
            from_tier = move.get('from_tier', 1)
            to_tier = move.get('to_tier', 1)
            to_location = move['to_location']

            # Find source AP ID to correct tier for moves to source
            source_ap_id = None
            for lane in lanes:
                if hasattr(lane, 'is_source') and getattr(lane, 'is_source', False):
                    source_ap_id = lane.get_ap_id()
                    break
            
            # If the destination is the source, force the tier to be 1
            if str(to_location) == str(source_ap_id):
                to_tier = 1

            # Use VRP's calculated timing if available, otherwise calculate our own
            # Ensure first move starts at timestep 1, not 0
            if 'start_time' in move:
                current_time = move['start_time']
                if move_idx == 0 and current_time == 0:
                    current_time = 1  # Force first move to start at t=1
            else:
                # Fallback to old calculation if VRP timing not available
                current_time = 1 if move_idx == 0 else current_time
            
            # Calculate real distance using Gurobi method instead of heuristic estimate
            real_distance = calculate_real_distance(from_location, to_location, from_tier, to_tier)
            service_time = move['service_time']
            empty_travel_distance = move.get('empty_travel_distance', 0)
            
            # Handle empty travel moves
            if move_type == 'empty':
                # This is an explicit empty travel move
                from_lane_id, from_tier_id = parse_location(from_location, from_tier)
                to_lane_id, to_tier_id = parse_location(to_location, to_tier)
                
                empty_decision = f"e_i{from_lane_id}_j{from_tier}_k{to_lane_id}_l{to_tier}_t{current_time}_v{vehicle_id}"
                
                translated_decisions[vehicle_key][str(current_time)] = {
                    'decision': empty_decision,
                    'move': f"[{from_lane_id}, {from_tier}] ⇄ [{to_lane_id}, {to_tier}]",
                    'distance': real_distance,
                    'travel_time': calculate_gurobi_style_travel_time(real_distance, False)  # Empty move
                }
                current_time += calculate_gurobi_style_travel_time(real_distance, False)
                current_location = to_location
                continue
            
            # Handle legacy empty travel if needed (for backward compatibility)
            if empty_travel_distance > 0:
                # Add empty travel move (e variable)
                from_lane_id, from_tier_id = parse_location(current_location, 1)  # Current location tier is always 1 for legacy moves
                to_lane_id, to_tier_id = parse_location(from_location, from_tier)
                
                # Calculate real distance for empty travel
                real_empty_distance = calculate_real_distance(current_location, from_location, 1, from_tier)
                
                empty_decision = f"e_i{from_lane_id}_j{from_tier_id}_k{to_lane_id}_l{to_tier}_t{current_time}_v{vehicle_id}"
                
                translated_decisions[vehicle_key][str(current_time)] = {
                    'decision': empty_decision,
                    'move': f"[{from_lane_id}, {from_tier_id}] ⇄ [{to_lane_id}, {to_tier}]",
                    'distance': real_empty_distance,
                    'travel_time': calculate_gurobi_style_travel_time(real_empty_distance, False)  # Empty move
                }
                current_time += calculate_gurobi_style_travel_time(real_empty_distance, False)
            
            # Create the main move decision
            if move_type == 'retrieve':
                from_lane_id, _ = parse_location(from_location, from_tier)
                # Find the actual sink AP ID
                sink_ap_id = 16  # Default fallback
                for lane in lanes:
                    if hasattr(lane, 'is_sink') and getattr(lane, 'is_sink', False):
                        sink_ap_id = lane.get_ap_id()
                        break
                decision = f"y_i{from_lane_id}_j{from_tier}_n{ul_id}_t{current_time}_v{vehicle_id}"
                move_symbol = "→"
                move_text = f"[{from_lane_id}, {from_tier}] {move_symbol} [{sink_ap_id}, 1]"
            elif move_type == 'store':
                to_lane_id, _ = parse_location(to_location, to_tier)
                # Find the actual source AP ID
                source_ap_id = 0  # Default fallback
                for lane in lanes:
                    if hasattr(lane, 'is_source') and getattr(lane, 'is_source', False):
                        source_ap_id = lane.get_ap_id()
                        break
                decision = f"z_i{to_lane_id}_j{to_tier}_n{ul_id}_t{current_time}_v{vehicle_id}"
                move_symbol = "→"
                move_text = f"[{source_ap_id}, 1] {move_symbol} [{to_lane_id}, {to_tier}]"
            elif move_type == 'reshuffle':
                from_lane_id, _ = parse_location(from_location, from_tier)
                to_lane_id, _ = parse_location(to_location, to_tier)
                decision = f"x_i{from_lane_id}_j{from_tier}_k{to_lane_id}_l{to_tier}_n{ul_id}_t{current_time}_v{vehicle_id}"
                move_symbol = "⇄"
                move_text = f"[{from_lane_id}, {from_tier}] {move_symbol} [{to_lane_id}, {to_tier}]"
            elif move_type == 'direct_retrieve':
                # Direct retrieve from source to sink - this is a load move from source to sink
                # Find source and sink access point IDs
                source_ap_id = None
                sink_ap_id = None
                for lane in lanes:
                    if hasattr(lane, 'is_source') and getattr(lane, 'is_source', False):
                        source_ap_id = lane.get_ap_id()
                    elif hasattr(lane, 'is_sink') and getattr(lane, 'is_sink', False):
                        sink_ap_id = lane.get_ap_id()
                
                # Direct retrieve is essentially a load move from source to sink
                # Use a retrieval-style decision variable format but from source to sink
                decision = f"y_i{source_ap_id}_j1_n{ul_id}_t{current_time}_v{vehicle_id}"
                move_symbol = "→"
                move_text = f"[{source_ap_id}, 1] {move_symbol} [{sink_ap_id}, 1]"
            else:
                # Unknown move type, use generic format
                from_lane_id, from_tier_id = parse_location(from_location, from_tier)
                to_lane_id, to_tier_id = parse_location(to_location, to_tier)
                decision = f"unknown_{move_type}_t{current_time}_v{vehicle_id}"
                move_symbol = "→"
                move_text = f"[{from_lane_id}, {from_tier}] {move_symbol} [{to_lane_id}, {to_tier}]"
            
            # Add the move to translated decisions
            translated_decisions[vehicle_key][str(current_time)] = {
                'decision': decision,
                'move': move_text,
                'distance': real_distance,
                'travel_time': calculate_gurobi_style_travel_time(real_distance, True)  # Loaded move
            }
            
            # Update current location and time
            # Use VRP's end_time if available, otherwise calculate
            if 'end_time' in move:
                current_time = move['end_time']
            else:
                current_time += calculate_gurobi_style_travel_time(real_distance, True)
    
    return translated_decisions


def validate_heuristic_solution_detailed(instance, test_case, verbose=False):
    """
    Validate a heuristic solution with detailed constraint violation reporting.
    
    Returns:
        tuple: (is_valid, status_code, detailed_report)
    """
    try:
        # Only validate if we have a solution
        if not test_case.amr_assignments or 'vehicles' not in test_case.amr_assignments:
            return False, -1, {"message": "No solution to validate", "violations": []}
        
        # Import here to avoid circular imports
        from src.test_cases.test_case_brr import TestCaseBrr
        
        # Extract the decisions from the test case
        translated_decisions = translate_heuristic_decisions_simple(test_case.amr_assignments, test_case.instance)
        
        if not translated_decisions:
            return False, -1, {"message": "No decisions to validate", "violations": []}
        
        # Extract decision strings and add idle moves for validation
        all_decision_values = []
        
        # Get sink AP ID for vehicle starting positions
        sink_ap_id = instance.get_buffer().get_sink().get_ap_id()

        for vehicle_id_str, vehicle_decisions in translated_decisions.items():
            vehicle_id = int(vehicle_id_str[1:])
            
            # Sort moves by start time
            sorted_moves = sorted(vehicle_decisions.items(), key=lambda item: int(item[0]))
            
            # Initial vehicle state
            current_time = 1
            current_lane = sink_ap_id
            current_tier = 1

            for start_time_str, move_details in sorted_moves:
                start_time = int(start_time_str)
                
                # Add idle moves for the gap before this move
                # for t in range(current_time, start_time):
                    # idle_decision = f"e_i{current_lane}_j{current_tier}_k{current_lane}_l{current_tier}_t{t}_v{vehicle_id}"
                    # all_decision_values.append(idle_decision)
                
                # Add the actual move
                all_decision_values.append(move_details['decision'])
                
                # Update vehicle state for the next iteration
                travel_time = move_details.get('travel_time', 1)
                current_time = start_time + travel_time
                
                # Determine the end location of the move
                move_text = move_details.get('move', '')
                match = re.search(r'\[(\d+), (\d+)\]$', move_text.strip())
                if match:
                    current_lane = int(match.group(1))
                    current_tier = int(match.group(2))

        if verbose:
            print(f"  Found {len(all_decision_values)} decisions to validate (including idle moves)")
            if all_decision_values:
                print(f"  Sample decisions: {all_decision_values[:5]}")
        
        if not all_decision_values:
            return False, -1, {"message": "No decision values found", "violations": []}
        
        # Create solution dictionary (same format as result checker)
        solution = {}
        for decision in all_decision_values:
            solution[decision] = 1
        
        # Create detailed report
        detailed_report = {
            "message": "",
            "decisions_count": len(all_decision_values),
            "decisions": all_decision_values[:5] if verbose else [],  # Show first 5 decisions if verbose
            "vehicles": len(translated_decisions),
            "violations": []
        }
        
        # Run constraint validation with verbose output to capture violations
        try:
            validation_test_case = TestCaseBrr(instance=instance, variant="dynamic_multiple", solution=solution, verbose=verbose, mode="check")
            
            # Capture the validation status
            status = validation_test_case.check_solution()
            
            # For heuristic solutions, if status is 3 but no violations are found,
            # treat it as feasible since our heuristic respects the important constraints
            # Note: We'll do the full analysis below, so just capture the status here
        except Exception as e:
            # If we can't create the validation test case, report this as a validation error
            detailed_report["message"] = f"Validation initialization failed: {str(e)}"
            detailed_report["violations"] = [{"type": "validation_error", "description": str(e)}]
            return False, -1, detailed_report
        
        # Always analyze for tardiness information, regardless of feasibility status
        violations, tardiness_info = analyze_constraint_violations(instance, test_case, all_decision_values, verbose)
        
        if verbose:
            print(f"  Analysis found {len(violations)} violations and {len(tardiness_info)} tardiness items")
            if tardiness_info:
                print(f"  Tardiness details: {tardiness_info[:3]}...")  # Show first 3 items
        
        # Trust Gurobi validation completely - do not override status 3
        # If Gurobi says infeasible (status 3), the solution is invalid
        
        # Add tardiness information to all reports
        total_tardiness = sum(info['tardiness'] for info in tardiness_info)
        num_late_unit_loads = len(set(info['ul_id'] for info in tardiness_info))
        detailed_report["tardiness_info"] = tardiness_info
        detailed_report["total_tardiness"] = total_tardiness
        detailed_report["num_late_unit_loads"] = num_late_unit_loads
        detailed_report["num_late_moves"] = len(tardiness_info)
        
        if status == 2:  # Optimal/feasible
            detailed_report["message"] = "Solution is feasible and optimal"
            # Even feasible solutions might have tardiness in some formulations
            detailed_report["violations"] = violations  # Include any soft violations
            return True, status, detailed_report
        elif status == 13:  # Suboptimal but feasible
            detailed_report["message"] = "Solution is feasible but suboptimal"
            detailed_report["violations"] = violations
            return True, status, detailed_report
        elif status == 9:  # Time limit reached but feasible solution found
            detailed_report["message"] = "Solution is feasible (time limit reached)"
            detailed_report["violations"] = violations
            return True, status, detailed_report
        elif status == 10:  # Solution limit reached
            detailed_report["message"] = "Solution is feasible (solution limit reached)"
            detailed_report["violations"] = violations
            return True, status, detailed_report
        elif status == 3:  # Infeasible
            detailed_report["message"] = "Solution is infeasible (Gurobi validation failed)"
            detailed_report["violations"] = violations
            if verbose:
                print(f"  Gurobi validation: INFEASIBLE - solution violates mathematical programming constraints")
            return False, status, detailed_report
        else:
            detailed_report["message"] = f"Validation failed with status {status}"
            return False, status, detailed_report
            
    except Exception as e:
        detailed_report = {
            "message": f"Validation error: {str(e)}",
            "violations": [{"type": "validation_error", "description": str(e)}]
        }
        return False, -1, detailed_report

def analyze_constraint_violations(instance, test_case, decisions, verbose=False):
    """
    Analyze potential constraint violations in the heuristic solution.
    
    Returns:
        List of violation descriptions
    """
    violations = []
    
    try:
        # Extract timeline information from decisions
        moves_by_vehicle = {}
        time_conflicts = []
        
        for decision in decisions:
            # Parse decision strings to extract information
            if decision.startswith('e_'):  # Empty travel
                # e_i{from_lane}_j{from_tier}_k{to_lane}_l{to_tier}_t{time}_v{vehicle}
                parts = decision.split('_')
                if len(parts) >= 8:
                    try:
                        from_lane = int(parts[1][1:])  # Remove 'i' prefix
                        to_lane = int(parts[3][1:])    # Remove 'k' prefix
                        time = int(parts[5][1:])       # Remove 't' prefix
                        vehicle = int(parts[6][1:])    # Remove 'v' prefix
                        
                        if vehicle not in moves_by_vehicle:
                            moves_by_vehicle[vehicle] = []
                        moves_by_vehicle[vehicle].append({
                            'type': 'empty',
                            'time': time,
                            'from': from_lane,
                            'to': to_lane,
                            'decision': decision
                        })
                    except (ValueError, IndexError):
                        violations.append({
                            "type": "decision_parsing_error",
                            "description": f"Could not parse empty travel decision: {decision}"
                        })
                        
            elif decision.startswith('z_'):  # Store operation
                # z_i{lane}_j{tier}_n{unit_load}_t{time}_v{vehicle}
                parts = decision.split('_')
                if len(parts) >= 6:
                    try:
                        lane = int(parts[1][1:])      # Remove 'i' prefix
                        tier = int(parts[2][1:])      # Remove 'j' prefix
                        unit_load = int(parts[3][1:]) # Remove 'n' prefix
                        time = int(parts[4][1:])      # Remove 't' prefix
                        vehicle = int(parts[5][1:])   # Remove 'v' prefix
                        
                        if vehicle not in moves_by_vehicle:
                            moves_by_vehicle[vehicle] = []
                        moves_by_vehicle[vehicle].append({
                            'type': 'store',
                            'time': time,
                            'lane': lane,
                            'tier': tier,
                            'unit_load': unit_load,
                            'decision': decision
                        })
                    except (ValueError, IndexError):
                        violations.append({
                            "type": "decision_parsing_error",
                            "description": f"Could not parse store decision: {decision}"
                        })
                        
            elif decision.startswith('y_'):  # Retrieve operation
                # y_i{lane}_j{tier}_n{unit_load}_t{time}_v{vehicle}
                parts = decision.split('_')
                if len(parts) >= 6:
                    try:
                        lane = int(parts[1][1:])      # Remove 'i' prefix
                        tier = int(parts[2][1:])      # Remove 'j' prefix
                        unit_load = int(parts[3][1:]) # Remove 'n' prefix
                        time = int(parts[4][1:])      # Remove 't' prefix
                        vehicle = int(parts[5][1:])   # Remove 'v' prefix
                        
                        # Check if this is a direct retrieval from source
                        move_type = 'retrieve'
                        for vl in instance.get_buffer().get_virtual_lanes():
                            if vl.get_ap_id() == lane and hasattr(vl, 'is_source') and getattr(vl, 'is_source', False):
                                move_type = 'direct_retrieve'
                                break
                        
                        if vehicle not in moves_by_vehicle:
                            moves_by_vehicle[vehicle] = []
                        moves_by_vehicle[vehicle].append({
                            'type': move_type,
                            'time': time,
                            'lane': lane,
                            'tier': tier,
                            'unit_load': unit_load,
                            'decision': decision
                        })
                    except (ValueError, IndexError):
                        violations.append({
                            "type": "decision_parsing_error",
                            "description": f"Could not parse retrieve decision: {decision}"
                        })
                        
            elif decision.startswith('x_'):  # Reshuffle operation
                # x_i{from_lane}_j{from_tier}_k{to_lane}_l{to_tier}_n{unit_load}_t{time}_v{vehicle}
                parts = decision.split('_')
                if len(parts) >= 8:
                    try:
                        from_lane = int(parts[1][1:])  # Remove 'i' prefix
                        from_tier = int(parts[2][1:])  # Remove 'j' prefix
                        to_lane = int(parts[3][1:])    # Remove 'k' prefix
                        to_tier = int(parts[4][1:])    # Remove 'l' prefix
                        unit_load = int(parts[5][1:])  # Remove 'n' prefix
                        time = int(parts[6][1:])       # Remove 't' prefix
                        vehicle = int(parts[7][1:])    # Remove 'v' prefix
                        
                        if vehicle not in moves_by_vehicle:
                            moves_by_vehicle[vehicle] = []
                        moves_by_vehicle[vehicle].append({
                            'type': 'reshuffle',
                            'time': time,
                            'from_lane': from_lane,
                            'from_tier': from_tier,
                            'to_lane': to_lane,
                            'to_tier': to_tier,
                            'unit_load': unit_load,
                            'decision': decision
                        })
                    except (ValueError, IndexError):
                        violations.append({
                            "type": "decision_parsing_error",
                            "description": f"Could not parse reshuffle decision: {decision}"
                        })
        
        # Check for time window violations and calculate tardiness
        unit_loads = instance.get_unit_loads()
        ul_time_windows = {}
        for ul in unit_loads:
            ul_time_windows[ul.get_id()] = {
                'arrival_start': ul.get_arrival_start(),
                'arrival_end': ul.get_arrival_end(),
                'retrieval_start': ul.get_retrieval_start(),
                'retrieval_end': ul.get_retrieval_end()
            }
        
        # Track tardiness information
        tardiness_info = []
        
        # Check each vehicle's schedule
        for vehicle_id, moves in moves_by_vehicle.items():
            # Sort moves by time
            moves.sort(key=lambda x: x['time'])
            
            # Check for time window violations and calculate tardiness
            for move in moves:
                if move['type'] in ['store', 'retrieve', 'direct_retrieve'] and 'unit_load' in move:
                    ul_id = move['unit_load']
                    if ul_id in ul_time_windows:
                        tw = ul_time_windows[ul_id]
                        
                        if move['type'] == 'store':
                            # Store operations should be within arrival window
                            if move['time'] < tw['arrival_start'] or move['time'] > tw['arrival_end']:
                                tardiness = 0
                                if move['time'] > tw['arrival_end']:
                                    tardiness = move['time'] - tw['arrival_end']
                                
                                violations.append({
                                    "type": "arrival_time_window_violation",
                                    "description": f"Unit load {ul_id} stored at time {move['time']}, but arrival window is [{tw['arrival_start']}, {tw['arrival_end']}]",
                                    "unit_load": ul_id,
                                    "time": move['time'],
                                    "window": [tw['arrival_start'], tw['arrival_end']]
                                })
                                
                                if tardiness > 0:
                                    tardiness_info.append({
                                        'move_id': f"store_{ul_id}",
                                        'ul_id': ul_id,
                                        'deadline': tw['arrival_end'],
                                        'expected_finish': move['time'],
                                        'tardiness': tardiness,
                                        'message': f"Store operation for UL {ul_id} late by {tardiness} time units"
                                    })
                                    if verbose:
                                        print(f"    Added tardiness item for store UL {ul_id}: tardiness={tardiness}")
                        
                        elif move['type'] in ['retrieve', 'direct_retrieve']:
                            # For retrieval, the time window applies to the arrival at the sink.
                            # We must calculate the travel time from the buffer to the sink.
                            buffer = instance.get_buffer()
                            from_lane_id = move['lane']
                            from_tier = move['tier']
                            
                            # Find the lane object to calculate distance
                            from_lane_obj = next((lane for lane in buffer.get_virtual_lanes() if lane.get_ap_id() == from_lane_id), None)
                            
                            if from_lane_obj:
                                # Calculate Gurobi-style travel time
                                lane_distance = buffer.get_distance_sink(from_lane_obj)
                                tier_distance = from_tier - 1
                                total_distance = lane_distance + tier_distance
                                travel_time = ceil(total_distance / instance.get_vehicle_speed()) + (2 * instance.get_handling_time())
                            else:
                                # Fallback if lane not found (should not happen)
                                travel_time = 0

                            arrival_at_sink_time = move['time'] + travel_time
                            
                            if arrival_at_sink_time < tw['retrieval_start'] or arrival_at_sink_time > tw['retrieval_end']:
                                tardiness = 0
                                if arrival_at_sink_time > tw['retrieval_end']:
                                    tardiness = arrival_at_sink_time - tw['retrieval_end']
                                
                                move_name = "directly retrieved" if move['type'] == 'direct_retrieve' else "retrieved"
                                violations.append({
                                    "type": "retrieval_time_window_violation",
                                    "description": f"Unit load {ul_id} {move_name} arrives at sink at time {arrival_at_sink_time}, but retrieval window is [{tw['retrieval_start']}, {tw['retrieval_end']}]",
                                    "unit_load": ul_id,
                                    "time": arrival_at_sink_time,
                                    "window": [tw['retrieval_start'], tw['retrieval_end']]
                                })
                                
                                if tardiness > 0:
                                    move_name = "Direct retrieve" if move['type'] == 'direct_retrieve' else "Retrieve"
                                    tardiness_info.append({
                                        'move_id': f"{move['type']}_{ul_id}",
                                        'ul_id': ul_id,
                                        'deadline': tw['retrieval_end'],
                                        'expected_finish': arrival_at_sink_time,
                                        'tardiness': tardiness,
                                        'message': f"{move_name} operation for UL {ul_id} late by {tardiness} time units"
                                    })
                                    if verbose:
                                        print(f"    Added tardiness item for {move['type']} UL {ul_id}: tardiness={tardiness}")
                            
                            # Special check for direct retrievals: source blocking violation
                            if move['type'] == 'direct_retrieve':
                                # For direct retrievals, the unit load sits at source from arrival_end until retrieval
                                # This blocks the source and should be flagged if the gap is too long
                                source_blocking_start = tw['arrival_end']
                                source_blocking_end = move['time']
                                blocking_duration = source_blocking_end - source_blocking_start
                                
                                if blocking_duration > 0:
                                    violations.append({
                                        "type": "source_blocking_violation",
                                        "description": f"Unit load {ul_id} blocks source from time {source_blocking_start} to {source_blocking_end} (duration: {blocking_duration})",
                                        "unit_load": ul_id,
                                        "blocking_start": source_blocking_start,
                                        "blocking_end": source_blocking_end,
                                        "blocking_duration": blocking_duration
                                    })
                                    
                                    # Add to tardiness info as well since this affects system efficiency
                                    tardiness_info.append({
                                        'move_id': f"source_blocking_{ul_id}",
                                        'ul_id': ul_id,
                                        'deadline': tw['arrival_end'],  # Should be moved out of source by arrival_end
                                        'expected_finish': move['time'],
                                        'tardiness': blocking_duration,
                                        'message': f"Unit load {ul_id} blocks source for {blocking_duration} time units"
                                    })
                                    if verbose:
                                        print(f"    Source blocking violation: UL {ul_id} blocks source for {blocking_duration} time units")
            
            # Check for vehicle conflicts (vehicle can't be in two places at once)
            for i in range(len(moves) - 1):
                current_move = moves[i]
                next_move = moves[i + 1]
                
                # Check if there's enough time between moves
                if next_move['time'] <= current_move['time']:
                    violations.append({
                        "type": "vehicle_time_conflict",
                        "description": f"Vehicle {vehicle_id} has overlapping moves at times {current_move['time']} and {next_move['time']}",
                        "vehicle": vehicle_id,
                        "move1": current_move['decision'],
                        "move2": next_move['decision']
                    })
        
        # Check for precedence violations (unit load must be stored before retrieved)
        unit_load_operations = {}
        for vehicle_id, moves in moves_by_vehicle.items():
            for move in moves:
                if move['type'] in ['store', 'retrieve', 'direct_retrieve'] and 'unit_load' in move:
                    ul_id = move['unit_load']
                    if ul_id not in unit_load_operations:
                        unit_load_operations[ul_id] = {'store': None, 'retrieve': [], 'direct_retrieve': []}
                    
                    if move['type'] == 'store':
                        if unit_load_operations[ul_id]['store'] is not None:
                            violations.append({
                                "type": "multiple_store_violation",
                                "description": f"Unit load {ul_id} is stored multiple times",
                                "unit_load": ul_id
                            })
                        unit_load_operations[ul_id]['store'] = move
                    elif move['type'] == 'retrieve':
                        unit_load_operations[ul_id]['retrieve'].append(move)
                    elif move['type'] == 'direct_retrieve':
                        unit_load_operations[ul_id]['direct_retrieve'].append(move)
        
        # Check precedence: store must happen before retrieve (but not for direct retrieve)
        for ul_id, ops in unit_load_operations.items():
            if ops['store'] and ops['retrieve']:
                store_time = ops['store']['time']
                for retrieve_move in ops['retrieve']:
                    if retrieve_move['time'] <= store_time:
                        violations.append({
                            "type": "precedence_violation", 
                            "description": f"Unit load {ul_id} retrieved at time {retrieve_move['time']} before being stored at time {store_time}",
                            "unit_load": ul_id,
                            "store_time": store_time,
                            "retrieve_time": retrieve_move['time']
                        })
            elif ops['retrieve'] and not ops['store'] and not ops['direct_retrieve']:
                # Only flag missing storage if there's no direct retrieval
                # Check if this unit load is already stored (arrival_start is None or 0)
                ul_is_prestored = False
                
                # Check if unit load has arrival_start = 0 or None (meaning already stored)
                for ul in unit_loads:
                    if ul.get_id() == ul_id:
                        arrival_start = ul.get_arrival_start() if hasattr(ul, 'get_arrival_start') else getattr(ul, 'arrival_start', None)
                        if arrival_start is None or arrival_start <= 0:
                            ul_is_prestored = True
                            break
                
                if not ul_is_prestored:
                    violations.append({
                        "type": "missing_store_violation",
                        "description": f"Unit load {ul_id} is retrieved but never stored",
                        "unit_load": ul_id
                    })
            # Note: Direct retrievals don't need storage operations, so no violation check needed
        
        if verbose and violations:
            print(f"\n🔍 Found {len(violations)} potential constraint violations:")
            for i, violation in enumerate(violations[:5], 1):  # Show first 5
                print(f"  {i}. {violation['type']}: {violation['description']}")
            if len(violations) > 5:
                print(f"  ... and {len(violations) - 5} more violations")
    
    except Exception as e:
        violations.append({
            "type": "analysis_error",
            "description": f"Error analyzing violations: {str(e)}"
        })
        tardiness_info = []
    
    return violations, tardiness_info

def run_gurobi_result_checker(heuristic_result_path, verbose=False):
    """
    Run the Gurobi result checker on the heuristic solution and capture output.
    
    Args:
        heuristic_result_path: Path to the heuristic result JSON file
        verbose: Whether to show verbose output
        
    Returns:
        Dictionary containing feasibility status and any warnings/violations
    """
    try:
        # Get the path to the result checker script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(script_dir, '../..')
        result_checker_path = os.path.join(root_dir, 'result_checker_BRR.py')
        
        # Run the result checker
        cmd = [sys.executable, result_checker_path, '--solution_path', heuristic_result_path]
        
        if verbose:
            print(f"Running Gurobi result checker: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=root_dir)
        
        gurobi_check = {
            'feasible': None,
            'status': None,
            'warnings': [],
            'violations': [],
            'raw_output': result.stdout.strip() if result.stdout else '',
            'error_output': result.stderr.strip() if result.stderr else ''
        }
        
        # Parse the output for feasibility status
        output_lines = result.stdout.strip().split('\n') if result.stdout else []
        
        for line in output_lines:
            # Look for status codes (common Gurobi status codes)
            if line.strip().isdigit():
                status_code = int(line.strip())
                gurobi_check['status'] = status_code
                
                # Interpret status codes
                if status_code == 2:  # Optimal
                    gurobi_check['feasible'] = True
                elif status_code == 13:  # Suboptimal but feasible
                    gurobi_check['feasible'] = True
                elif status_code == 3:  # Infeasible
                    gurobi_check['feasible'] = False
                elif status_code == 9:  # Time limit reached
                    gurobi_check['feasible'] = True
                elif status_code == 10:  # Solution limit reached
                    gurobi_check['feasible'] = True
                else:
                    gurobi_check['feasible'] = False
            
            # Look for warning messages about timing violations
            if "Warning:" in line:
                warning_match = re.search(r'Warning: Move (\d+) \(UL (\d+)\) cannot finish by deadline (\d+)\. Expected finish: (\d+), Vehicle available: (\d+)', line)
                if warning_match:
                    move_id, ul_id, deadline, expected_finish, vehicle_available = warning_match.groups()
                    tardiness = int(expected_finish) - int(deadline)
                    gurobi_check['warnings'].append({
                        'move_id': int(move_id),
                        'ul_id': int(ul_id),
                        'deadline': int(deadline),
                        'expected_finish': int(expected_finish),
                        'vehicle_available': int(vehicle_available),
                        'tardiness': tardiness,
                        'message': line.strip()
                    })
                else:
                    # Generic warning
                    gurobi_check['warnings'].append({
                        'message': line.strip()
                    })
        
        # If no explicit feasibility status found, try to infer from return code
        if gurobi_check['feasible'] is None:
            if result.returncode == 0:
                gurobi_check['feasible'] = True
            else:
                gurobi_check['feasible'] = False
        
        return gurobi_check
        
    except Exception as e:
        if verbose:
            print(f"Error running Gurobi result checker: {e}")
        return {
            'feasible': None,
            'status': None,
            'warnings': [],
            'violations': [],
            'error': str(e),
            'raw_output': '',
            'error_output': ''
        }