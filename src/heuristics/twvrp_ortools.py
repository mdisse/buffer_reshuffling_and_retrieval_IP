import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
from math import ceil
import re

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from src.bay import tier


@dataclass
class VRPMove:
    move_id: int
    from_location: str
    to_location: str
    ul_id: int
    move_type: str
    travel_distance: float
    service_time: int = 1
    earliest_start: int = 0
    latest_finish: int = 9999
    from_tier: int = 1
    to_tier: int = 1


class TWVRPORToolsSolver:
    def __init__(self, buffer, num_vehicles: int = 1, vehicle_capacity: int = 1, instance=None, verbose: bool = False):
        """
        Initialize the TWVRPORToolsSolver with buffer, vehicle count, capacity, and instance.
        buffer: Buffer object with layout and distance info.
        num_vehicles: Number of vehicles (AMRs) available.
        vehicle_capacity: Capacity per vehicle (default 1).
        instance: Problem instance (for unit load info).
        verbose: If True, print debug info.
        """
        self.verbose = verbose
        self.buffer = buffer
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.instance = instance
        self.source_lane_id = str(self.buffer.get_source().get_ap_id()) if self.buffer.get_source() else None
        self.sink_lane_id = str(self.buffer.get_sink().get_ap_id()) if self.buffer.get_sink() else None

    def solve_twvrp(self, moves: List[Dict], time_limit: Optional[int] = None) -> Dict:
        """
        Solve the time window VRP for the given move sequence using OR-Tools.
        moves: List of move dicts (from A* or heuristic).
        time_limit: Max time in seconds for solver.
        Returns a dict with vehicle assignments, total distance, total time, and status.
        """
        if not moves:
            return {
                'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'no_moves'
            }
        
        optimized_moves = self._optimize_direct_retrievals(moves)
        vrp_moves = self._convert_moves_to_vrp_jobs(optimized_moves)
        return self._solve_ortools(vrp_moves, time_limit)

    def _optimize_direct_retrievals(self, moves: List[Dict]) -> List[Dict]:
        """
        Post-processes the A* move list to find a 'store' move and a subsequent
        'retrieve' move for the same unit load that can be combined into a single
        direct retrieval, provided no intermediate moves conflict with the storage location.
        """
        if not self.instance or len(moves) < 2:
            return moves

        optimized_moves = []
        processed_indices = set()
        
        for i in range(len(moves)):
            if i in processed_indices:
                continue

            move1 = moves[i]
            
            # We are looking for a 'store' move that hasn't been processed yet.
            if move1.get('type') != 'store':
                optimized_moves.append(move1)
                continue

            ul_id_to_match = move1.get('ul_id')
            storage_location = self._parse_location_str(move1.get('to_pos'))
            
            found_match = False
            # Look ahead for a matching 'retrieve' move.
            for j in range(i + 1, len(moves)):
                if j in processed_indices: continue

                # Check for a conflict with the storage location in intermediate moves.
                # A conflict occurs if an intermediate move uses the temporary storage lane.
                intermediate_indices = range(i + 1, j)
                conflict = False
                for k in intermediate_indices:
                    inter_move = moves[k]
                    inter_from = self._parse_location_str(inter_move.get('from_pos'))
                    inter_to = self._parse_location_str(inter_move.get('to_pos'))
                    if storage_location in [inter_from, inter_to]:
                        conflict = True
                        break
                if conflict:
                    # Conflict found. The UL at the storage location was disturbed.
                    # Stop searching for a match for move1.
                    break

                move2 = moves[j]
                # Check if we found the matching retrieve move.
                if (move2.get('type') == 'retrieve' and move2.get('ul_id') == ul_id_to_match):
                    # Potential match found. Check time window feasibility.
                    ul = next((u for u in self.instance.get_unit_loads() if u.get_id() == ul_id_to_match), None)
                    if not ul: break

                    direct_travel_time = self._calculate_gurobi_style_travel_time(
                        'source', 'sink', 1, 1, handling_time_enabled=True
                    )
                    earliest_dep = max(ul.get_arrival_start() or 0, (ul.get_retrieval_start() or 0) - direct_travel_time)
                    latest_dep = min(ul.get_arrival_end() or float('inf'), (ul.get_retrieval_end() or float('inf')) - direct_travel_time)

                    if earliest_dep <= latest_dep:
                        # Feasible! Perform the optimization.
                        if self.verbose:
                            print(f"  OPTIMIZATION: Combining store (idx {i}) and retrieve (idx {j}) for UL {ul_id_to_match} into direct retrieval.")
                        
                        # Add the intermediate moves that occurred before the original store move
                        for k in range(len(optimized_moves)):
                            if moves[k]['type'] != 'store' and moves[k]['type'] != 'retrieve':
                                processed_indices.add(k)

                        # Then, add the new combined direct_retrieve move.
                        new_move = {
                            'ul_id': ul_id_to_match,
                            'from_pos': 'source',
                            'to_pos': 'sink',
                            'type': 'direct_retrieve',
                        }
                        # Add the intermediate moves first
                        for k in intermediate_indices:
                            if k not in processed_indices:
                                optimized_moves.append(moves[k])
                                processed_indices.add(k)
                        
                        optimized_moves.append(new_move)
                        
                        # Mark ONLY the original store and retrieve moves as processed.
                        processed_indices.add(i)
                        processed_indices.add(j)
                        
                        found_match = True
                        break # Stop searching for other matches for move1.
            
            # If after checking all subsequent moves, no match was found, add the original store move.
            if not found_match:
                optimized_moves.append(move1)

        return optimized_moves

    def _create_data_model(self, vrp_moves: List[VRPMove]) -> Dict:
        """
        Build the data model for OR-Tools: time matrix, time windows, depot, moves, etc.
        vrp_moves: List of VRPMove objects.
        Returns a dict for OR-Tools routing model.
        """
        locations = ['depot'] + [i for i in range(len(vrp_moves))]
        n_locations = len(locations)
        
        time_matrix = []
        for i in range(n_locations):
            time_row = []
            for j in range(n_locations):
                if i == j:
                    time_row.append(0)
                    continue
                
                # From depot to a move
                if i == 0:
                    move = vrp_moves[j-1]
                    # Travel from depot (sink) to start of move.
                    # This is just the empty travel time.
                    travel_time = self._calculate_gurobi_style_travel_time(
                        'sink', move.from_location, 1, move.from_tier, handling_time_enabled=False
                    )
                    time_row.append(travel_time)
                # From a move to depot
                elif j == 0:
                    from_move = vrp_moves[i-1]
                    # Travel from end of move back to depot (sink)
                    travel_time = self._calculate_gurobi_style_travel_time(
                        from_move.to_location, 'sink', from_move.to_tier, 1, handling_time_enabled=False
                    )
                    time_row.append(travel_time)
                # From one move to another
                else:
                    from_move = vrp_moves[i-1]
                    to_move = vrp_moves[j-1]
                    
                    # Time from end of from_move to start of to_move (empty travel)
                    travel_time = self._calculate_gurobi_style_travel_time(
                        from_move.to_location, to_move.from_location, from_move.to_tier, to_move.from_tier, handling_time_enabled=False
                    )
                    time_row.append(travel_time)
            time_matrix.append(time_row)
        
        latest_move_finish = max(move.latest_finish for move in vrp_moves) if vrp_moves else 1
        time_windows = [(0, latest_move_finish + 5000)]
        
        for move in vrp_moves:
            time_windows.append((move.earliest_start, move.latest_finish))
        
        data = {
            'time_matrix': time_matrix,
            'time_windows': time_windows,
            'num_vehicles': self.num_vehicles,
            'depot': 0,
            'moves': vrp_moves
        }
        
        return data

    def _convert_solution(self, data: Dict, manager, routing, solution, vrp_moves: List[VRPMove]) -> Dict:
        """
        Convert the OR-Tools solution to a dictionary with vehicle routes, moves, and stats.
        data: Data model dict.
        manager, routing: OR-Tools objects.
        solution: OR-Tools solution object.
        vrp_moves: List of VRPMove objects.
        Returns a dict with vehicle assignments, total distance, total time, and status.
        """
        if not solution:
            return {
                'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'no_solution'
            }
        
        time_dimension = routing.GetDimensionOrDie('Time')
        vehicles = []
        total_tardiness = 0
        
        for vehicle_id in range(data['num_vehicles']):
            if not routing.IsVehicleUsed(solution, vehicle_id):
                continue
                
            vehicle = {
                'vehicle_id': vehicle_id + 1,
                'moves': [],
                'total_distance': 0,
                'total_time': 0
            }
            
            index = routing.Start(vehicle_id)
            
            # Initial state of the vehicle at the depot (sink)
            current_location = 'sink'
            current_tier = 1
            # The time the vehicle is ready to start its tour
            last_event_end_time = solution.Min(time_dimension.CumulVar(index))

            while not routing.IsEnd(index):
                next_index = solution.Value(routing.NextVar(index))
                
                if routing.IsEnd(next_index):
                    break

                node_index = manager.IndexToNode(next_index)
                move = vrp_moves[node_index - 1] # -1 because depot is 0
                
                # The time the vehicle arrives at the start of the service move
                arrival_at_service_loc = solution.Min(time_dimension.CumulVar(next_index))

                # Calculate tardiness for this move and add to total
                tardiness = max(0, arrival_at_service_loc - move.latest_finish)
                total_tardiness += tardiness

                # Add the empty travel move
                empty_dist = self._calculate_distance(current_location, move.from_location, current_tier, move.from_tier)
                if empty_dist > 0:
                    vehicle['moves'].append({
                        'ul_id': 0,
                        'from_location': current_location,
                        'to_location': move.from_location,
                        'move_type': 'empty',
                        'travel_distance': empty_dist,
                        'service_time': 0,
                        'start_time': last_event_end_time,
                        'end_time': arrival_at_service_loc,
                        'from_tier': current_tier,
                        'to_tier': move.from_tier,
                    })
                    vehicle['total_distance'] += empty_dist

                # Now, add the actual service move
                # The service itself starts when the empty travel ends.
                service_start_time = arrival_at_service_loc
                service_duration = self._calculate_gurobi_style_travel_time(
                    move.from_location, move.to_location, move.from_tier, move.to_tier, handling_time_enabled=True
                )
                service_end_time = service_start_time + service_duration

                vehicle['moves'].append({
                    'ul_id': move.ul_id,
                    'from_location': move.from_location,
                    'to_location': move.to_location,
                    'move_type': move.move_type,
                    'travel_distance': move.travel_distance,
                    'service_time': service_duration,
                    'start_time': service_start_time,
                    'end_time': service_end_time,
                    'from_tier': move.from_tier,
                    'to_tier': move.to_tier,
                })
                vehicle['total_distance'] += move.travel_distance
                
                # Update state for the next iteration
                current_location = move.to_location
                current_tier = move.to_tier
                last_event_end_time = service_end_time
                index = next_index

            # After the loop, handle the final empty move back to the sink
            final_arrival_time = solution.Min(time_dimension.CumulVar(routing.End(vehicle_id)))
            empty_dist_to_sink = self._calculate_distance(current_location, 'sink', current_tier, 1)
            if empty_dist_to_sink > 0:
                vehicle['moves'].append({
                    'ul_id': 0,
                    'from_location': current_location,
                    'to_location': 'sink',
                    'move_type': 'empty',
                    'travel_distance': empty_dist_to_sink,
                    'service_time': 0,
                    'start_time': last_event_end_time,
                    'end_time': final_arrival_time,
                    'from_tier': current_tier,
                    'to_tier': 1,
                })
                vehicle['total_distance'] += empty_dist_to_sink
            
            vehicle['total_time'] = final_arrival_time
            vehicles.append(vehicle)
        
        total_distance = sum(v['total_distance'] for v in vehicles)
        
        makespan = 0
        if solution:
            time_dimension = routing.GetDimensionOrDie('Time')
            
            # Calculate makespan (max end time of all vehicles)
            for vehicle_id in range(data['num_vehicles']):
                if routing.IsVehicleUsed(solution, vehicle_id):
                    makespan = max(makespan, solution.Min(time_dimension.CumulVar(routing.End(vehicle_id))))

            # The separate tardiness calculation loop is no longer needed.

        return {
            'vehicles': vehicles,
            'total_distance': total_distance,
            'total_time': makespan,
            'total_tardiness': total_tardiness,
            'solver_status': 'optimal'
        }

    def _parse_location_str(self, loc_str: any) -> str:
        """
        Parse a location string (e.g., '[11, 1]', 'sink', 'source') to extract the AP id or keyword.
        Returns 'source', 'sink', or the AP id as a string.
        """
        s = str(loc_str)
        if s.lower() in ['source', 'sink']:
            return s.lower()
        
        ap_id = re.search(r'\d+', s)
        return ap_id.group(0) if ap_id else s

    def _parse_location_and_tier_str(self, loc_str: any) -> tuple[str, int]:
        """
        Parse a location string (e.g., '[11, 1]', 'sink', 'source') to extract the AP id and tier.
        Returns a tuple of (ap_id_string, tier_int).
        """
        s = str(loc_str)
        if s.lower() in ['source', 'sink']:
            return s.lower(), 1
        
        # Find all numbers in the string
        numbers = re.findall(r'\d+', s)
        if len(numbers) >= 2:
            return numbers[0], int(numbers[1])
        elif len(numbers) == 1:
            return numbers[0], 1 # Default to tier 1 if not specified
        
        return s, 1 # Fallback

    def _calculate_distance(self, from_loc: str, to_loc: str, from_tier: int = 1, to_tier: int = 1) -> float:
        """
        Wrapper to calculate distance using the instance's method.
        It converts string locations and tier numbers to the objects required by the instance.
        """
        def get_lane_or_str(loc: any):
            """Helper to get lane object from string or return if already object/special string."""
            if isinstance(loc, str):
                if loc.lower() in ['source', 'sink']:
                    return loc.lower()
                try:
                    ap_id = int(self._parse_location_str(loc))
                    for lane in self.buffer.get_virtual_lanes():
                        if lane.get_ap_id() == ap_id:
                            return lane
                    raise ValueError(f"Lane with AP ID {ap_id} not found from location '{loc}'")
                except (ValueError, TypeError):
                    raise ValueError(f"Could not parse lane from location: {loc}")
            return loc # Assume it's already a lane object

        def get_tier_or_int(lane_obj: any, tier_id: int):
            """Helper to get tier object or return the integer ID."""
            if isinstance(lane_obj, str) or tier_id == 1:
                return tier_id
            for tier in lane_obj.get_tiers():
                if tier.get_id() == tier_id:
                    return tier
            return tier_id # Fallback

        lane1_obj = get_lane_or_str(from_loc)
        lane2_obj = get_lane_or_str(to_loc)

        tier1_obj = get_tier_or_int(lane1_obj, from_tier)
        tier2_obj = get_tier_or_int(lane2_obj, to_tier)

        # Handle the special case where both are source/sink and identical
        if lane1_obj == lane2_obj and isinstance(lane1_obj, str):
            return 0.0

        return self.instance.calculate_distance(lane1_obj, tier1_obj, lane2_obj, tier2_obj)
        
    def _calculate_gurobi_style_travel_time(self, from_loc: str, to_loc: str, from_tier: int, to_tier: int, handling_time_enabled: bool) -> int:
        """
        Calculate travel time between two locations using EXACTLY the same logic as the IP model.
        Returns the travel time as an integer.
        """
        # Use the main distance calculation method
        distance = self._calculate_distance(from_loc, to_loc, from_tier, to_tier)
        travel_time = distance / self.instance.vehicle_speed
        
        if handling_time_enabled: 
            return max(1, ceil(travel_time) + 2*self.instance.get_handling_time())
        return max(1, ceil(travel_time))  # return at least 1 to avoid multiple repositionings at a time step
    
    def _calculate_distance_ip_style(self, lane1, tier1, lane2, tier2):
        """
        DEPRECATED: This method is now replaced by the main _calculate_distance wrapper.
        Kept for compatibility in case other parts of the code call it directly.
        It now forwards the call to the new central method.
        """
        warnings.warn("_calculate_distance_ip_style is deprecated. Use _calculate_distance instead.", DeprecationWarning)
        
        # The inputs lane1/lane2 can be objects or strings. The new _calculate_distance handles this.
        from_loc = lane1.get_ap_id() if hasattr(lane1, 'get_ap_id') else lane1
        to_loc = lane2.get_ap_id() if hasattr(lane2, 'get_ap_id') else lane2
        
        from_tier = tier1.get_id() if hasattr(tier1, 'get_id') else tier1
        to_tier = tier2.get_id() if hasattr(tier2, 'get_id') else tier2

        return self._calculate_distance(str(from_loc), str(to_loc), from_tier, to_tier)

    def _convert_moves_to_vrp_jobs(self, moves: List[Dict]) -> List[VRPMove]:
        """
        Convert a list of move dicts to a list of VRPMove dataclass objects, extracting time windows from unit loads.
        Returns a list of VRPMove objects.
        """
        vrp_moves = []
        for i, move in enumerate(moves):
            ul_id = move.get('ul_id', 0)
            move_type = move.get('type', 'unknown')
            
            from_pos, from_tier_parsed = self._parse_location_and_tier_str(move.get('from_pos', 'source'))
            to_pos, to_tier_parsed = self._parse_location_and_tier_str(move.get('to_pos', 'sink'))

            # Use tier information from move data if available, otherwise fall back to parsed tier
            from_tier = move.get('from_tier', from_tier_parsed)
            to_tier = move.get('to_tier', to_tier_parsed)

            earliest_start, latest_start = 0, 99999
            
            ul = next((u for u in self.instance.get_unit_loads() if u.get_id() == ul_id), None)
            if ul:
                # The service_time is the duration of the loaded move itself.
                service_time = self._calculate_gurobi_style_travel_time(
                    from_pos, to_pos, from_tier, to_tier, handling_time_enabled=True
                )

                if move_type == 'store':
                    # REQUIREMENT: The storage move must START within the arrival window.
                    # The VRP solver constrains the start time, so we use the window directly.
                    earliest_start = ul.get_arrival_start() or 0
                    latest_start = ul.get_arrival_end() or 99999
                
                elif move_type == 'retrieve':
                    # REQUIREMENT: The move must FINISH within the retrieval window [retrieval_start, retrieval_end].
                    # This implies:
                    #   start_time + service_time >= retrieval_start  =>  start_time >= retrieval_start - service_time
                    #   start_time + service_time <= retrieval_end    =>  start_time <= retrieval_end - service_time
                    
                    earliest_finish_time = ul.get_retrieval_start() or 0
                    earliest_start = earliest_finish_time - service_time

                    latest_finish_time = ul.get_retrieval_end() or 99999
                    latest_start = latest_finish_time - service_time
                
                elif move_type == 'direct_retrieve':
                    # REQUIREMENT: A direct move must satisfy BOTH arrival and retrieval windows.
                    # The move must START within the arrival window [arrival_start, arrival_end].
                    # The move must FINISH within the retrieval window [retrieval_start, retrieval_end].
                    
                    # Calculate the valid start window based on retrieval constraints
                    retrieval_earliest_start = (ul.get_retrieval_start() or 0) - service_time
                    retrieval_latest_start = (ul.get_retrieval_end() or 99999) - service_time
                    
                    # The final start window is the intersection of the arrival window and the calculated retrieval start window.
                    earliest_start = max(ul.get_arrival_start() or 0, retrieval_earliest_start)
                    latest_start = min(ul.get_arrival_end() or 99999, retrieval_latest_start)

            # Ensure the calculated latest_start is not before the earliest_start.
            # This prevents invalid time windows like [16, 8].
            if latest_start < earliest_start:
                if self.verbose:
                    print(f"  WARNING: Infeasible time window for move {i} (UL {ul_id}, {move_type}). "
                          f"Calculated start window: [{earliest_start}, {latest_start}]. Adjusting latest_start.")
                latest_start = earliest_start

            vrp_moves.append(VRPMove(
                move_id=i, from_location=from_pos, to_location=to_pos, ul_id=ul_id,
                move_type=move_type, travel_distance=self._calculate_distance(from_pos, to_pos, from_tier, to_tier),
                earliest_start=max(0, earliest_start), latest_finish=max(0, latest_start),
                from_tier=from_tier, to_tier=to_tier
            ))
        
        # --- NEW: Pre-process and propagate time constraints ---
        # This gives the solver a more realistic starting point for earliest_start times.
        if self.verbose:
            print("  Propagating time constraints through move sequence...")
        
        # Build precedence graph based on the same logic used in the solver
        precedence_rules = self._get_precedence_rules(vrp_moves)
        
        # Iteratively update earliest start times based on precedence
        for _ in range(len(vrp_moves)): # Iterate to ensure propagation through long chains
            updated = False
            for earlier_move_id, later_move_id in precedence_rules:
                earlier_move = vrp_moves[earlier_move_id]
                later_move = vrp_moves[later_move_id]
                
                service_time = self._calculate_gurobi_style_travel_time(
                    earlier_move.from_location, earlier_move.to_location, 
                    earlier_move.from_tier, earlier_move.to_tier, handling_time_enabled=True
                )
                
                new_earliest_start = earlier_move.earliest_start + service_time
                
                if new_earliest_start > later_move.earliest_start:
                    vrp_moves[later_move_id].earliest_start = new_earliest_start
                    # If the new earliest start makes the window invalid, adjust the latest finish
                    if vrp_moves[later_move_id].earliest_start > vrp_moves[later_move_id].latest_finish:
                        vrp_moves[later_move_id].latest_finish = vrp_moves[later_move_id].earliest_start
                    updated = True
            if not updated:
                break # No more updates, propagation is complete

        return vrp_moves

    def _solve_ortools(self, vrp_moves: List[VRPMove], time_limit: Optional[int] = None) -> Dict:
        """
        Run the OR-Tools solver on the VRP moves and return the solution as a dictionary.
        Handles time windows, precedence constraints, and search parameters.
        Returns a dict with vehicle assignments, total distance, total time, and status.
        """
        if self.verbose:
            print("\n--- VRP Solver Input ---")
            print(f"  Moves to solve: {len(vrp_moves)}")
            print(f"  Number of vehicles: {self.num_vehicles}")
            print(f"  Time limit: {time_limit}s")
            for i, move in enumerate(vrp_moves[:15]): 
                print(f"  Move {i}: UL {move.ul_id} ({move.move_type}) from {move.from_location} to {move.to_location}, "
                      f"TW: [{move.earliest_start}, {move.latest_finish}]")
            if len(vrp_moves) > 15:
                print("  ...")
            print("------------------------\n")
            
        try:
            data = self._create_data_model(vrp_moves)
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
            routing = pywrapcp.RoutingModel(manager)
            
            # Create a service time callback for each move
            service_times = [0] # Depot has 0 service time
            for move in vrp_moves:
                service_time = self._calculate_gurobi_style_travel_time(
                    move.from_location, move.to_location, move.from_tier, move.to_tier, handling_time_enabled=True
                )
                service_times.append(service_time)

            # --- Time Dimension Callback (includes service time) ---
            # This callback defines how time accumulates along a route.
            # It's travel_time(A->B) + service_time(A).
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                travel_time = data['time_matrix'][from_node][to_node]
                service_time_at_from_node = service_times[from_node]
                return travel_time + service_time_at_from_node
            
            transit_callback_index = routing.RegisterTransitCallback(time_callback)
            
            # --- Objective Function Callback (cost to minimize) ---
            # The objective is to minimize empty travel time. Tardiness is handled by penalties.
            def travel_time_cost_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['time_matrix'][from_node][to_node]

            travel_time_cost_callback_index = routing.RegisterTransitCallback(travel_time_cost_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(travel_time_cost_callback_index)

            # --- Add Time Dimension ---
            time_dimension_name = 'Time'
            horizon = 100000 
            routing.AddDimension(
                transit_callback_index, # Use the callback that includes service time
                3000,  # Allow waiting time
                horizon, # Total time horizon
                False, # Don't force start at 0
                time_dimension_name
            )
            time_dimension = routing.GetDimensionOrDie(time_dimension_name)
            
            # --- Add Time Window Constraints ---
            # A large penalty for being late on any task. This encourages the solver
            # to meet deadlines but allows it to find a solution if constraints are too tight.
            late_task_penalty = 100000 

            for i, move in enumerate(vrp_moves):
                location_idx = i + 1 # +1 because depot is at index 0
                time_window = data['time_windows'][location_idx]
                index = manager.NodeToIndex(location_idx)
                
                # The earliest start time is always a hard constraint.
                time_dimension.CumulVar(index).SetMin(time_window[0])

                # For retrieval moves, the deadline is a HARD constraint to respect downstream processes.
                if move.move_type in ['retrieve', 'direct_retrieve']:
                    # SetMax enforces a hard upper bound on the start time.
                    # The latest_start (time_window[1]) is calculated as retrieval_end - service_time,
                    # so this ensures the move FINISHES by the retrieval deadline.
                    time_dimension.CumulVar(index).SetMax(time_window[1])
                    if self.verbose:
                        print(f"  Applying HARD deadline for move {move.move_id} ({move.move_type}) to start by {time_window[1]}")
                else:
                    # For other moves (store, reshuffle), use a SOFT deadline for flexibility.
                    time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], late_task_penalty)
            
            self._add_precedence_constraints(routing, manager, time_dimension, vrp_moves)
            
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            # Strategy to find the first solution. PATH_CHEAPEST_ARC is a good heuristic.
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            )
            
            # Instruct the solver to stop after finding the first solution.
            search_parameters.solution_limit = 1

            if self.verbose: 
                search_parameters.log_search = True
            
            safe_time_limit = int(time_limit) if time_limit is not None else 30
            search_parameters.time_limit.FromSeconds(safe_time_limit)
            
            solution = routing.SolveWithParameters(search_parameters)

            if self.verbose:
                print(routing.status())
            if solution:
                return self._convert_solution(data, manager, routing, solution, vrp_moves)
            else:
                if self.verbose:
                    print(f"OR-Tools failed to find solution. Solver status: {routing.status()}")
                    # You can add more detailed debug information here if needed
                    # For example, check if any nodes were unperformed
                return {'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'failed'}
            
        except Exception as e:
            if self.verbose:
                print(f"Error in OR-Tools solver: {e}")
            return {'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'error', 'error_message': str(e)}

    def _get_precedence_rules(self, vrp_moves: List[VRPMove]) -> List[Tuple[int, int]]:
        """Helper function to extract all precedence rules for pre-processing."""
        rules = []
        
        # Same UL precedence
        ul_moves = {}
        for move in vrp_moves:
            if move.ul_id not in ul_moves:
                ul_moves[move.ul_id] = []
            ul_moves[move.ul_id].append(move)

        for ul_id, moves in ul_moves.items():
            if len(moves) > 1:
                sorted_moves = sorted(moves, key=lambda m: m.move_id)
                for i in range(len(sorted_moves) - 1):
                    rules.append((sorted_moves[i].move_id, sorted_moves[i+1].move_id))

        # Same Lane (LIFO) precedence
        lane_moves = {}
        excluded_locations = {'source', 'sink', self.source_lane_id, self.sink_lane_id}
        excluded_locations.discard(None)

        for move in vrp_moves:
            if move.from_location not in excluded_locations:
                lane_id = move.from_location
                if lane_id not in lane_moves: lane_moves[lane_id] = []
                lane_moves[lane_id].append(move)

            if move.to_location not in excluded_locations and move.to_location != move.from_location:
                lane_id = move.to_location
                if lane_id not in lane_moves: lane_moves[lane_id] = []
                lane_moves[lane_id].append(move)

        for lane_id, moves in lane_moves.items():
            if len(moves) > 1:
                sorted_moves = sorted(moves, key=lambda m: m.move_id)
                for i in range(len(sorted_moves) - 1):
                    rules.append((sorted_moves[i].move_id, sorted_moves[i+1].move_id))
        
        return list(set(rules)) # Return unique rules

    def _add_precedence_constraints(self, routing, manager, time_dimension, vrp_moves):
        """
        Add precedence constraints to the OR-Tools routing model to enforce the A* move sequence
        for moves involving the same unit load.
        """
        if self.verbose:
            print("  Enforcing VRP precedence for moves involving the same unit load.")

        # Create a dictionary to hold service times for each move
        service_times = {}
        for move in vrp_moves:
            service_time = self._calculate_gurobi_style_travel_time(
                move.from_location, move.to_location, move.from_tier, move.to_tier, handling_time_enabled=True
            )
            service_times[move.move_id] = service_time

        # Precedence constraints for same unit load
        ul_moves = {}
        for i, move in enumerate(vrp_moves):
            if move.ul_id not in ul_moves:
                ul_moves[move.ul_id] = []
            ul_moves[move.ul_id].append(move)

        for ul_id, moves in ul_moves.items():
            if len(moves) > 1:
                # Sort moves by their original sequence order (move_id)
                sorted_moves = sorted(moves, key=lambda m: m.move_id)
                for i in range(len(sorted_moves) - 1):
                    earlier_move = sorted_moves[i]
                    later_move = sorted_moves[i+1]

                    earlier_node_idx = manager.NodeToIndex(earlier_move.move_id + 1)
                    later_node_idx = manager.NodeToIndex(later_move.move_id + 1)
                    
                    # REMOVED: The constraint forcing the same vehicle was over-constraining the problem.
                    # It is valid for one vehicle to store a UL and another to retrieve it later.
                    # routing.solver().Add(
                    #     routing.VehicleVar(earlier_node_idx) == routing.VehicleVar(later_node_idx)
                    # )

                    # This enforces that the start of the later move must be after the earlier move is finished.
                    # This is a temporal constraint, independent of which vehicle performs the move.
                    # EndTime(earlier) = StartTime(earlier) + ServiceTime(earlier)
                    earlier_service_time = service_times[earlier_move.move_id]
                    routing.solver().Add(
                        time_dimension.CumulVar(later_node_idx) >=
                        time_dimension.CumulVar(earlier_node_idx) + earlier_service_time
                    )
                    
                    if self.verbose:
                        print(f"    VRP PRECEDENCE (UL {ul_id}): Move {earlier_move.move_id} must precede Move {later_move.move_id}")

        # Precedence constraints for moves at the same lane (access point), respecting LIFO
        if self.verbose:
            print("  Enforcing VRP precedence for moves at the same lane (LIFO).")
        
        lane_moves = {}
        # Define locations that should not have precedence constraints (source/sink are not buffer lanes)
        excluded_locations = {'source', 'sink', self.source_lane_id, self.sink_lane_id}
        excluded_locations.discard(None)

        for move in vrp_moves:
            # A move conflicts with a lane if it stores a UL into it or retrieves a UL from it.
            # Empty moves to/from a lane do not cause a LIFO conflict with other service moves.
            
            # Conflict at 'from_location' if it's a retrieve from a buffer lane
            if move.from_location not in excluded_locations:
                lane_id = move.from_location
                if lane_id not in lane_moves:
                    lane_moves[lane_id] = []
                lane_moves[lane_id].append(move)

            # Conflict at 'to_location' if it's a store to a buffer lane
            if move.to_location not in excluded_locations:
                lane_id = move.to_location
                # Avoid double-counting if from and to are the same (reshuffle in same lane)
                if lane_id != move.from_location:
                    if lane_id not in lane_moves:
                        lane_moves[lane_id] = []
                    lane_moves[lane_id].append(move)

        for lane_id, moves in lane_moves.items():
            if len(moves) > 1:
                # Sort moves by their original A* sequence order (move_id)
                sorted_moves = sorted(moves, key=lambda m: m.move_id)
                if self.verbose:
                    print(f"    VRP PRECEDENCE (Lane {lane_id}): Enforcing sequence for {len(sorted_moves)} moves.")

                for i in range(len(sorted_moves) - 1):
                    earlier_move = sorted_moves[i]
                    later_move = sorted_moves[i+1]

                    earlier_node_idx = manager.NodeToIndex(earlier_move.move_id + 1)
                    later_node_idx = manager.NodeToIndex(later_move.move_id + 1)

                    # For lane precedence, only wait for the handling time at the lane
                    # (not the full travel time to the next location)
                    handling_time_at_lane = self.instance.get_handling_time()
                    routing.solver().Add(
                        time_dimension.CumulVar(later_node_idx) >=
                        time_dimension.CumulVar(earlier_node_idx) + handling_time_at_lane
                    )
                    if self.verbose:
                        print(f"      - Move {earlier_move.move_id} handling must complete before Move {later_move.move_id} starts at lane {lane_id}")


def solve_twvrp_with_ortools(buffer, moves: List[Dict], num_vehicles: int = 1, instance=None, time_limit: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Module-level helper to solve the time window VRP using OR-Tools.
    buffer: Buffer object.
    moves: List of move dicts.
    num_vehicles: Number of vehicles (AMRs).
    instance: Problem instance.
    time_limit: Max time in seconds for solver.
    verbose: If True, print debug info.
    Returns a dict with vehicle assignments, total distance, total time, and status.
    """
    solver = TWVRPORToolsSolver(buffer, num_vehicles, vehicle_capacity=1, instance=instance, verbose=verbose)
    return solver.solve_twvrp(moves, time_limit)