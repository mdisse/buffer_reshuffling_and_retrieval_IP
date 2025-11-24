import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
from math import ceil
import re
from functools import lru_cache

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from src.bay import tier
from functools import lru_cache


class VRPSolutionCallback:
    """Callback to monitor and control the OR-Tools solving process."""
    
    def __init__(self, routing, manager, time_dimension, max_solutions=10, verbose=False):
        self.routing = routing
        self.manager = manager
        self.time_dimension = time_dimension
        self.max_solutions = max_solutions
        self.verbose = verbose
        self.solution_count = 0
        self.best_objective = float('inf')
        self.solutions = []
        
    def __call__(self):
        """Called by OR-Tools when a solution is found."""
        self.solution_count += 1
        current_objective = self.routing.CostVar().Max()
        
        if current_objective < self.best_objective:
            self.best_objective = current_objective
            if self.verbose:
                print(f"  Solution {self.solution_count}: New best objective = {current_objective}")
        
        # Store solution info
        self.solutions.append({
            'count': self.solution_count,
            'objective': current_objective
        })
        
        # Stop if we found enough good solutions
        if self.solution_count >= self.max_solutions:
            if self.verbose:
                print(f"  Reached max solutions ({self.max_solutions}), stopping search.")
            self.routing.solver().FinishCurrentSearch()


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
    def __init__(self, buffer, num_vehicles: int = 1, vehicle_capacity: int = 1, instance=None, verbose: bool = False,
                 storage_slack: int = 0, retrieval_slack: int = 0):
        """
        Initialize the TWVRPORToolsSolver with buffer, vehicle count, capacity, and instance.
        
        Args:
            buffer: Buffer/warehouse layout
            num_vehicles: Number of AMRs available  
            vehicle_capacity: Capacity of each AMR (typically 1)
            instance: Problem instance with unit loads and time windows
            verbose: Enable detailed logging
            storage_slack: Time units to relax storage deadlines (default 0 - HARD constraints)
            retrieval_slack: Time units to relax retrieval deadlines (default 0 - HARD constraints)
                            
        Design rationale:
            - Lane occupancy now correctly modeled with in-lane travel time
            - Precedence constraints account for actual lane clearing time
            - Hard time windows (slack=0) ensure zero tardiness
            - OR-Tools can insert idle/wait time between moves for feasibility
        """
        self.verbose = verbose
        self.buffer = buffer
        self.num_vehicles = num_vehicles
        # self.num_vehicles = 15
        self.vehicle_capacity = vehicle_capacity
        self.instance = instance
        self.storage_slack = storage_slack
        self.retrieval_slack = retrieval_slack
        self.source_lane_id = str(self.buffer.get_source().get_ap_id()) if self.buffer.get_source() else None
        self.sink_lane_id = str(self.buffer.get_sink().get_ap_id()) if self.buffer.get_sink() else None
        self.lane_map = {lane.get_ap_id(): lane for lane in self.buffer.get_virtual_lanes()}
        
        # Cache the number of slots for each lane (for lane occupancy calculations)
        self.lane_n_slots = {}
        for lane in self.buffer.get_virtual_lanes():
            n_slots = len(lane.get_tiers())
            # Store as string to match move locations
            self.lane_n_slots[str(lane.get_ap_id())] = n_slots
        
        # Initialize caches for location parsing and travel time calculations
        self._location_parse_cache = {}
        self._location_tier_parse_cache = {}
        self._travel_time_cache = {}

    def solve_twvrp(self, moves: List[Dict], time_limit: Optional[int] = None) -> Dict:
        """
        Solve the time window VRP for the given move sequence using OR-Tools.
        
        Args:
            moves: List of move dicts (from A* or heuristic).
            time_limit: Max time in seconds for solver.
            use_rolling_horizon: If True, use rolling horizon approach. If None, auto-decide based on problem size.
        
        Returns:
            Dict with vehicle assignments, total distance, total time, and status.
        """
        if not moves:
            return {
                'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'no_moves'
            }
        
        # Direct retrieval optimization is now done BEFORE A* search,
        # so we can directly convert moves to VRP jobs
        vrp_moves = self._convert_moves_to_vrp_jobs(moves)
        return self._solve_ortools(vrp_moves, time_limit)

    # NOTE: _optimize_direct_retrievals is no longer used here - optimization happens
    # before A* search in test_case_brr.py to avoid occupying buffer slots unnecessarily

    def _create_data_model(self, vrp_moves: List[VRPMove]) -> Dict:
        """
        Build the data model for OR-Tools: time matrix, time windows, depot, moves, etc.
        OPTIMIZED: Pre-calculate all distances and times once.
        """
        n_locations = len(vrp_moves) + 1  # +1 for depot
        time_matrix = [[0] * n_locations for _ in range(n_locations)]
        
        # Pre-compute depot transitions
        depot_to_moves = []
        moves_to_depot = []
        
        for move in vrp_moves:
            depot_to_moves.append(self._calculate_gurobi_style_travel_time(
                'sink', move.from_location, 1, move.from_tier, handling_time_enabled=False
            ))
            moves_to_depot.append(self._calculate_gurobi_style_travel_time(
                move.to_location, 'sink', move.to_tier, 1, handling_time_enabled=False
            ))
        
        # Calculate service times for each move
        service_times = [0]  # Depot has 0 service time
        for move in vrp_moves:
            service_time = self._calculate_gurobi_style_travel_time(
                move.from_location, move.to_location, move.from_tier, move.to_tier, handling_time_enabled=True
            )
            service_times.append(service_time)
        
        time_matrix = []
        for i in range(n_locations):
            time_row = []  # Initialize time_row for each row
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
        
        # Calculate realistic horizon based on actual time windows
        latest_move_finish = max(move.latest_finish for move in vrp_moves) if vrp_moves else 1
        total_service_time = sum(service_times)
        horizon = latest_move_finish + total_service_time + 1000  # Much tighter than 100000
        
        time_windows = [(0, horizon)]
        for move in vrp_moves:
            time_windows.append((move.earliest_start, move.latest_finish))
        
        data = {
            'time_matrix': time_matrix,
            'time_windows': time_windows,
            'num_vehicles': self.num_vehicles,
            'depot': 0,
            'moves': vrp_moves,
            'service_times': service_times,
            'horizon': horizon
        }
        
        return data

    def _convert_solution(self, data: Dict, manager, routing, solution, vrp_moves: List[VRPMove]) -> Dict:
        """
        Convert the OR-Tools solution to a dictionary with vehicle routes, moves, and stats.
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
            current_location = 'sink'
            current_tier = 1
            last_event_end_time = solution.Min(time_dimension.CumulVar(index))

            while not routing.IsEnd(index):
                next_index = solution.Value(routing.NextVar(index))
                
                if routing.IsEnd(next_index):
                    break

                node_index = manager.IndexToNode(next_index)
                move = vrp_moves[node_index - 1]
                
                arrival_at_service_loc = solution.Min(time_dimension.CumulVar(next_index))
                tardiness = max(0, arrival_at_service_loc - move.latest_finish)
                total_tardiness += tardiness
                
                # Log tardiness for debugging
                if self.verbose and tardiness > 0:
                    print(f"  ⚠️  TARDINESS detected: Move {move.move_id} (UL {move.ul_id}) arrived at {arrival_at_service_loc}, deadline was {move.latest_finish}, late by {tardiness}")

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
                    # Store original time window for validation
                    'earliest_start': move.earliest_start,
                    'latest_finish': move.latest_finish,
                })
                vehicle['total_distance'] += move.travel_distance
                
                current_location = move.to_location
                current_tier = move.to_tier
                last_event_end_time = service_end_time
                index = next_index

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

        # Check if solution is actually feasible (no time window violations)
        solver_status = 'optimal'
        violation_details = []
        
        # Detailed validation of each move's time windows
        for vehicle in vehicles:
            for move in vehicle['moves']:
                if move['ul_id'] == 0:  # Skip empty moves
                    continue
                
                start_time = move['start_time']
                earliest = move.get('earliest_start', 0)
                latest = move.get('latest_finish', 99999)
                
                if start_time < earliest:
                    violation = {
                        'vehicle': vehicle['vehicle_id'],
                        'ul_id': move['ul_id'],
                        'move_type': move['move_type'],
                        'violation_type': 'too_early',
                        'start_time': start_time,
                        'earliest_allowed': earliest,
                        'violation_amount': earliest - start_time
                    }
                    violation_details.append(violation)
                    if self.verbose:
                        print(f"  ⚠️  VIOLATION: UL {move['ul_id']} ({move['move_type']}) started at {start_time}, "
                              f"but earliest allowed is {earliest} (too early by {earliest - start_time})")
                
                if start_time > latest:
                    violation = {
                        'vehicle': vehicle['vehicle_id'],
                        'ul_id': move['ul_id'],
                        'move_type': move['move_type'],
                        'violation_type': 'too_late',
                        'start_time': start_time,
                        'latest_allowed': latest,
                        'violation_amount': start_time - latest
                    }
                    violation_details.append(violation)
                    if self.verbose:
                        print(f"  ⚠️  VIOLATION: UL {move['ul_id']} ({move['move_type']}) started at {start_time}, "
                              f"but latest allowed is {latest} (too late by {start_time - latest})")
        
        if total_tardiness > 0 or violation_details:
            if self.verbose:
                print(f"\n  ⚠️  WARNING: Solution has {len(violation_details)} time window violation(s)!")
                print(f"     Total tardiness: {total_tardiness} time units")
                print(f"     This solution is INFEASIBLE in practice.")
            solver_status = 'suboptimal_with_violations'
        
        return {
            'vehicles': vehicles,
            'total_distance': total_distance,
            'total_time': makespan,
            'total_tardiness': total_tardiness,
            'solver_status': solver_status,
            'time_window_violations': violation_details
        }

    def _parse_location_str(self, loc_str: any) -> str:
        """Parse a location string to extract the AP id or keyword."""
        s = str(loc_str)
        
        # Check cache first
        if s in self._location_parse_cache:
            return self._location_parse_cache[s]
        
        if s.lower() in ['source', 'sink']:
            result = s.lower()
        else:
            ap_id = re.search(r'\d+', s)
            result = ap_id.group(0) if ap_id else s
        
        self._location_parse_cache[s] = result
        return result

    def _parse_location_and_tier_str(self, loc_str: any) -> tuple[str, int]:
        """Parse a location string to extract the AP id and tier."""
        s = str(loc_str)
        
        # Check cache first
        if s in self._location_tier_parse_cache:
            return self._location_tier_parse_cache[s]
        
        if s.lower() in ['source', 'sink']:
            return s.lower(), 1
        
        # Find all numbers in the string
        numbers = re.findall(r'\d+', s)
        if len(numbers) >= 2:
            return numbers[0], int(numbers[1])
        elif len(numbers) == 1:
            return numbers[0], 1 # Default to tier 1 if not specified
        
        return s, 1 # Fallback

    @lru_cache(maxsize=None)
    def _get_in_lane_travel_time(self, loc: str, tier: int) -> int:
        """
        Calculates the one-way travel time from the lane's AP to the specified tier.
        """
        if loc in {self.source_lane_id, self.sink_lane_id, 'source', 'sink', None}:
            return 0
            
        n_slots = self.lane_n_slots.get(loc)
        if not n_slots:
            return 0 # Fallback

        # 'tier' is 1-based from deepest (tier 1) to shallowest (tier n_slots)
        # Travel distance (in units) from AP to tier = n_slots - tier
        in_lane_dist = n_slots - tier
        
        # Convert distance to time
        travel_time = ceil(max(0, in_lane_dist) / self.instance.vehicle_speed)
        return travel_time

    @lru_cache(maxsize=None)
    def _get_lane_occupancy_duration(self, loc: str, tier: int) -> int:
        """
        Calculates total time a lane is occupied for a task.
        (Travel In) + (Handling) + (Travel Out)
        """
        if loc in {self.source_lane_id, self.sink_lane_id, 'source', 'sink', None}:
            # Source/Sink are not exclusive lanes, just use handling time
            return self.instance.get_handling_time()
        
        # 1. Get one-way in-lane travel time
        one_way_travel_time = self._get_in_lane_travel_time(loc, tier)
        
        # 2. Get handling time
        handling_time = self.instance.get_handling_time()
        
        # 3. Total Occupancy = (Travel In) + (Handling) + (Travel Out)
        total_time = (2 * one_way_travel_time) + handling_time
        
        return max(1, total_time) # Ensure at least 1 time unit

    def _calculate_distance(self, from_loc: str, to_loc: str, from_tier: int = 1, to_tier: int = 1) -> float:
        """
        OPTIMIZED: Cached distance calculation wrapper.
        """
        def get_lane_or_str(loc: any):
            if isinstance(loc, str):
                if loc.lower() in ['source', 'sink']:
                    return loc.lower()
                try:
                    ap_id = int(self._parse_location_str(loc))
                    lane = self.lane_map.get(ap_id)
                    if lane is not None:
                        return lane
                except (ValueError, TypeError):
                    raise ValueError(f"Could not parse lane from location: {loc}")
            return loc

        def get_tier_or_int(lane_obj: any, tier_id: int):
            if isinstance(lane_obj, str) or tier_id == 1:
                return tier_id
            for tier in lane_obj.get_tiers():
                if tier.get_id() == tier_id:
                    return tier
            return tier_id

        lane1_obj = get_lane_or_str(from_loc)
        lane2_obj = get_lane_or_str(to_loc)
        tier1_obj = get_tier_or_int(lane1_obj, from_tier)
        tier2_obj = get_tier_or_int(lane2_obj, to_tier)

        if lane1_obj == lane2_obj and isinstance(lane1_obj, str):
            return 0.0

        return self.instance.calculate_distance(lane1_obj, tier1_obj, lane2_obj, tier2_obj)
        
    def _calculate_gurobi_style_travel_time(self, from_loc: str, to_loc: str, from_tier: int, to_tier: int, handling_time_enabled: bool) -> int:
        """
        OPTIMIZED: Cached travel time calculation.
        """
        cache_key = (from_loc, to_loc, from_tier, to_tier, handling_time_enabled)
        if cache_key in self._travel_time_cache:
            return self._travel_time_cache[cache_key]
        
        distance = self._calculate_distance(from_loc, to_loc, from_tier, to_tier)
        travel_time = distance / self.instance.vehicle_speed
        
        if handling_time_enabled: 
            result = max(1, ceil(travel_time) + 2*self.instance.get_handling_time())
        else:
            result = max(1, ceil(travel_time))
        
        self._travel_time_cache[cache_key] = result
        return result

    def _convert_moves_to_vrp_jobs(self, moves: List[Dict]) -> List[VRPMove]:
        """
        OPTIMIZED: Convert moves to VRP jobs with efficient time window propagation.
        """
        vrp_moves = []
        for i, move in enumerate(moves):
            ul_id = move.get('ul_id', 0)
            move_type = move.get('type', 'unknown')
            
            from_pos, from_tier_parsed = self._parse_location_and_tier_str(move.get('from_pos', 'source'))
            to_pos, to_tier_parsed = self._parse_location_and_tier_str(move.get('to_pos', 'sink'))
            from_tier = move.get('from_tier', from_tier_parsed)
            to_tier = move.get('to_tier', to_tier_parsed)

            earliest_start, latest_start = 0, 99999
            
            ul = next((u for u in self.instance.get_unit_loads() if u.get_id() == ul_id), None)
            if ul:
                service_time = self._calculate_gurobi_style_travel_time(
                    from_pos, to_pos, from_tier, to_tier, handling_time_enabled=True
                )

                if move_type == 'store':
                    # HARD constraint: must start storage after arrival_start
                    earliest_start = ul.get_arrival_start() or 0
                    # SOFT constraint with BOUNDED slack: prefer arrival_end, but allow small delay
                    # This prevents infeasibility when precedence forces slightly later storage
                    latest_start = (ul.get_arrival_end() or 99999) + self.storage_slack
                
                elif move_type == 'retrieve':
                    # Allow retrieval anytime (UL waiting in buffer is fine, early retrieval OK)
                    earliest_start = 0
                    # SOFT constraint with SMALL slack: retrieval deadline is important but allow minor delays
                    # Small slack (10 units) handles precedence conflicts without excessive tardiness
                    latest_finish_time = (ul.get_retrieval_end() or 99999) + self.retrieval_slack
                    latest_start = max(0, latest_finish_time - service_time)
                
                elif move_type == 'direct_retrieve':
                    # Direct retrieve: source -> sink
                    # HARD: must start after arrival_start
                    earliest_start = ul.get_arrival_start() or 0
                    # SOFT constraint with SMALL slack: retrieval deadline important but allow minor delays
                    # Small slack (10 units) handles timing conflicts
                    retrieval_latest_finish = (ul.get_retrieval_end() or 99999) + self.retrieval_slack
                    retrieval_latest_start = max(0, retrieval_latest_finish - service_time)
                    latest_start = retrieval_latest_start

            # Fix infeasible windows
            if latest_start < earliest_start:
                if self.verbose:
                    print(f"  WARNING: Infeasible time window for move {i} (UL {ul_id}, {move_type}). "
                          f"Window [{earliest_start}, {latest_start}] -> Adjusting to [{earliest_start}, {earliest_start}]")
                latest_start = earliest_start

            vrp_moves.append(VRPMove(
                move_id=i, from_location=from_pos, to_location=to_pos, ul_id=ul_id,
                move_type=move_type, travel_distance=self._calculate_distance(from_pos, to_pos, from_tier, to_tier),
                earliest_start=max(0, earliest_start), latest_finish=max(earliest_start, latest_start),
                from_tier=from_tier, to_tier=to_tier
            ))
        
        # --- Pre-process and propagate time constraints (with feasibility preservation) ---
        # This gives the solver a more realistic starting point for earliest_start times.
        # CRITICAL: We must NOT create infeasible windows by propagation!
        if self.verbose:
            print("  Propagating time constraints through move sequence...")
        
        precedence_rules = self._get_precedence_rules(vrp_moves)
        
        # Iteratively update earliest start times based on precedence
        # BUT: Do NOT tighten windows beyond their original constraints
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
                
                # Only update if it improves the bound WITHOUT violating the original time window
                if new_earliest_start > later_move.earliest_start:
                    # Check if this would make the window infeasible
                    # IMPORTANT: With slack on STORAGE, allow some violations - let OR-Tools try
                    # For RETRIEVALS, we use a tolerance check but keep hard deadlines
                    if new_earliest_start > later_move.latest_finish:
                        gap = new_earliest_start - later_move.latest_finish
                        
                        # Determine if this is a critical violation based on move type
                        # Storage: allow up to storage_slack gap
                        # Retrieval: allow minimal tolerance (let OR-Tools handle with waiting)
                        is_storage = 'store' in later_move.move_type
                        tolerance = self.storage_slack if is_storage else 5  # Small tolerance for retrievals
                        
                        if gap > tolerance or self.verbose:
                            print(f"  ⚠️  Time window conflict during propagation:")
                            print(f"      Move {later_move_id} (UL {later_move.ul_id}, {later_move.move_type})")
                            print(f"      Precedence requires start >= {new_earliest_start}")
                            print(f"      But time window allows start <= {later_move.latest_finish}")
                            print(f"      Gap: {gap} time units (tolerance: storage={self.storage_slack}, retrieval=5)")
                            if gap <= tolerance:
                                print(f"      Within tolerance - letting OR-Tools attempt solution...")
                        
                        # Update earliest_start and let OR-Tools try to solve
                        vrp_moves[later_move_id].earliest_start = new_earliest_start
                        updated = True
                    else:
                        vrp_moves[later_move_id].earliest_start = new_earliest_start
                        updated = True
            if not updated:
                break
        
        # Final diagnostic check
        if self.verbose:
            infeasible = [m for m in vrp_moves if m.earliest_start > m.latest_finish]
            if infeasible:
                print(f"\n  ⚠️  {len(infeasible)} moves still have infeasible windows after propagation!")
                for m in infeasible[:3]:
                    print(f"     Move {m.move_id} (UL {m.ul_id}): [{m.earliest_start}, {m.latest_finish}]")

        return vrp_moves

    def _solve_ortools(self, vrp_moves: List[VRPMove], time_limit: Optional[int] = None) -> Dict:
        """
        OPTIMIZED: Run OR-Tools solver with improved search parameters and infeasibility handling.
        """
        if self.verbose:
            print("\n--- VRP Solver Input ---")
            print(f"  Moves to solve: {len(vrp_moves)}")
            print(f"  Number of vehicles: {self.num_vehicles}")
            print(f"  Time limit: {time_limit}s")
            
            # Diagnostic: Check for infeasible time windows
            infeasible_count = 0
            for i, move in enumerate(vrp_moves[:15]): 
                is_infeasible = move.earliest_start > move.latest_finish
                marker = " ⚠️ INFEASIBLE" if is_infeasible else ""
                if is_infeasible:
                    infeasible_count += 1
                print(f"  Move {i}: UL {move.ul_id} ({move.move_type}) from {move.from_location} to {move.to_location}, "
                      f"TW: [{move.earliest_start}, {move.latest_finish}]{marker}")
            if len(vrp_moves) > 15:
                print(f"  ... ({len(vrp_moves) - 15} more moves)")
            
            if infeasible_count > 0:
                print(f"\n  ⚠️  WARNING: {infeasible_count} moves have infeasible time windows!")
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
            
            def travel_time_cost_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['time_matrix'][from_node][to_node]

            travel_time_cost_callback_index = routing.RegisterTransitCallback(travel_time_cost_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(travel_time_cost_callback_index)

            time_dimension_name = 'Time'
            routing.AddDimension(
                transit_callback_index,
                5000,  # Increased slack for more flexibility
                data['horizon'],
                False,
                time_dimension_name
            )
            time_dimension = routing.GetDimensionOrDie(time_dimension_name)
            
            # --- Add Time Window Constraints ---
            # IMPORTANT: All time windows must be HARD constraints to ensure feasibility.
            # The heuristic's purpose is to find feasible solutions quickly, not to find
            # better solutions by violating time windows. If no solution exists with hard
            # constraints, that's the correct answer - the problem is infeasible.
            
            for i, move in enumerate(vrp_moves):
                location_idx = i + 1 # +1 because depot is at index 0
                time_window = data['time_windows'][location_idx]
                index = manager.NodeToIndex(location_idx)
                
                # Both earliest start and latest start are HARD constraints for all moves
                time_dimension.CumulVar(index).SetMin(time_window[0])
                time_dimension.CumulVar(index).SetMax(time_window[1])
                
                if self.verbose:
                    print(f"  Applying HARD time window for move {move.move_id} ({move.move_type}): "
                          f"[{time_window[0]}, {time_window[1]}]")

            
            self._add_precedence_constraints(routing, manager, time_dimension, vrp_moves, service_times)
            
            # Register solution callback
            solution_callback = VRPSolutionCallback(routing, manager, time_dimension, max_solutions=5, verbose=self.verbose)
            routing.AddAtSolutionCallback(solution_callback)
            
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            
            # Prioritize finding feasible solutions with no tardiness
            if len(vrp_moves) < 20:
                # For small problems, use PATH_CHEAPEST_ARC which is good for time windows
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
                )
            else:
                # For larger problems, use PARALLEL_CHEAPEST_INSERTION which respects time windows
                search_parameters.first_solution_strategy = (
                    routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
                )
            
            # Use GUIDED_LOCAL_SEARCH for better optimization of time windows
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            
            # Increase effort to find better solutions
            search_parameters.lns_time_limit.FromMilliseconds(100)

            if self.verbose: 
                search_parameters.log_search = True
            
            # Give more time for complex problems to find tardiness-free solutions
            safe_time_limit = int(time_limit) if time_limit is not None else 60  # Increased from 30s
            search_parameters.time_limit.FromSeconds(safe_time_limit)
            
            # Allow more solutions to be explored for better quality
            # No solution limit - search fully within time budget for best tardiness-free solution
            
            solution = routing.SolveWithParameters(search_parameters)

            if self.verbose:
                status_names = {
                    0: "ROUTING_NOT_SOLVED",
                    1: "ROUTING_SUCCESS", 
                    2: "ROUTING_FAIL",
                    3: "ROUTING_FAIL_TIMEOUT",
                    4: "ROUTING_INVALID",
                }
                status_code = routing.status()
                status_name = status_names.get(status_code, f"UNKNOWN({status_code})")
                print(f"\nSolver status: {status_code} ({status_name})")
                
            if solution:
                return self._convert_solution(data, manager, routing, solution, vrp_moves)
            else:
                # FALLBACK: Try with all soft constraints
                if self.verbose:
                    print(f"\n⚠️  Initial solve failed. Attempting with ALL SOFT deadlines...")
                
                return self._solve_with_relaxed_constraints(vrp_moves, time_limit)
            
        except Exception as e:
            if self.verbose:
                print(f"Error in OR-Tools solver: {e}")
            return {'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'error', 'error_message': str(e)}

    def _solve_with_relaxed_constraints(self, vrp_moves: List[VRPMove], time_limit: Optional[int] = None) -> Dict:
        """
        Fallback solver with ALL soft time window constraints.
        Used when the main solver fails to find a feasible solution.
        """
        if self.verbose:
            print("\n--- Attempting RELAXED solve with all soft constraints ---\n")
        
        try:
            data = self._create_data_model(vrp_moves)
            manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']), data['num_vehicles'], data['depot'])
            routing = pywrapcp.RoutingModel(manager)
            
            service_times = data['service_times']

            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                travel_time = data['time_matrix'][from_node][to_node]
                service_time_at_from_node = service_times[from_node]
                return travel_time + service_time_at_from_node
            
            transit_callback_index = routing.RegisterTransitCallback(time_callback)
            
            def travel_time_cost_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return data['time_matrix'][from_node][to_node]

            travel_time_cost_callback_index = routing.RegisterTransitCallback(travel_time_cost_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(travel_time_cost_callback_index)

            time_dimension_name = 'Time'
            routing.AddDimension(
                transit_callback_index,
                10000,  # Very large slack
                data['horizon'] * 2,  # Double horizon for more flexibility
                False,
                time_dimension_name
            )
            time_dimension = routing.GetDimensionOrDie(time_dimension_name)
            
            # Calculate penalties based on problem scale to strongly discourage tardiness
            # Penalties should be much higher than the total travel time of all moves
            max_time = max([tw[1] for tw in data['time_windows'][1:]] + [data['horizon']])
            num_moves = len(vrp_moves)
            
            # Make penalties proportional to problem: 
            # - Early penalty: moderate (being early is less critical)
            # - Late penalty: VERY high (tardiness violates hard constraints)
            # Each unit of tardiness should cost more than the entire problem's travel time
            early_penalty = max_time * num_moves * 100  # 100x total time scale
            late_penalty = max_time * num_moves * 10000  # 10,000x total time scale (massive)
            
            if self.verbose:
                print(f"  Using soft time windows with SCALED penalties:")
                print(f"    Problem scale: {num_moves} moves, max_time: {max_time}")
                print(f"    Early penalty: {early_penalty} per time unit ({early_penalty / (max_time * num_moves):.0f}x problem scale)")
                print(f"    Late penalty: {late_penalty} per time unit ({late_penalty / (max_time * num_moves):.0f}x problem scale)")
                print(f"    This makes tardiness EXTREMELY expensive vs finding a feasible solution")

            for i, move in enumerate(vrp_moves):
                location_idx = i + 1
                time_window = data['time_windows'][location_idx]
                index = manager.NodeToIndex(location_idx)
                
                # ALL constraints are soft in relaxed mode
                time_dimension.SetCumulVarSoftLowerBound(index, time_window[0], early_penalty)
                time_dimension.SetCumulVarSoftUpperBound(index, time_window[1], late_penalty)
            
            # Still add precedence constraints - these are structural
            self._add_precedence_constraints(routing, manager, time_dimension, vrp_moves, data)
            
            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
            )

            if self.verbose: 
                search_parameters.log_search = True
            
            safe_time_limit = int(time_limit) if time_limit is not None else 20
            search_parameters.time_limit.FromSeconds(safe_time_limit)
            
            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                if self.verbose:
                    print("\n✓ Found solution with relaxed constraints\n")
                result = self._convert_solution(data, manager, routing, solution, vrp_moves)
                result['solver_status'] = 'relaxed_solution'
                
                # CRITICAL: If slack is 0 (hard constraints desired), reject solutions with any tardiness
                if self.storage_slack == 0 and self.retrieval_slack == 0:
                    # Check if result has tardiness
                    has_tardiness = False
                    for vehicle in result.get('vehicles', []):
                        for move in vehicle.get('moves', []):
                            if move.get('ul_id', 0) > 0:  # Not an empty move
                                # Check arrival time vs deadline
                                move_id = move.get('move_id', -1)
                                if move_id >= 0 and move_id < len(vrp_moves):
                                    vrp_move = vrp_moves[move_id]
                                    start_time = move.get('start_time', 0)
                                    if start_time > vrp_move.latest_finish:
                                        has_tardiness = True
                                        break
                        if has_tardiness:
                            break
                    
                    if has_tardiness:
                        if self.verbose:
                            print("  ❌ REJECTING relaxed solution: has tardiness but hard constraints (slack=0) required!")
                            print("     This indicates A* sequence is time-infeasible.")
                        return {'vehicles': [], 'total_distance': 0, 'total_time': 0, 
                                'solver_status': 'failed_hard_constraints', 
                                'error_message': 'No feasible solution with zero tardiness'}
                
                return result
            else:
                if self.verbose:
                    print("\n✗ Even relaxed solve failed\n")
                return {'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'failed'}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in relaxed solver: {e}")
            return {'vehicles': [], 'total_distance': 0, 'total_time': 0, 'solver_status': 'error', 'error_message': str(e)}

    def _get_precedence_rules(self, vrp_moves: List[VRPMove]) -> List[Tuple[int, int]]:
        """Helper function to extract all precedence rules for pre-processing."""
        rules = []
        
        # Same UL precedence - use defaultdict for efficiency
        from collections import defaultdict
        ul_moves = defaultdict(list)
        for move in vrp_moves:
            if move.ul_id not in ul_moves:
                ul_moves[move.ul_id] = []
            ul_moves[move.ul_id].append(move)

        for ul_id, moves in ul_moves.items():
            if len(moves) > 1:
                sorted_moves = sorted(moves, key=lambda m: m.move_id)
                for i in range(len(sorted_moves) - 1):
                    rules.append((sorted_moves[i].move_id, sorted_moves[i+1].move_id))

        # Same Lane (LIFO) precedence - exclude direct retrievals
        lane_moves = defaultdict(list)
        excluded_locations = {'source', 'sink', self.source_lane_id, self.sink_lane_id}
        excluded_locations.discard(None)

        for move in vrp_moves:
            # Skip direct retrievals - they don't use buffer lanes
            if move.move_type == 'direct_retrieve':
                continue
                
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
        
        return list(rules)

    def _add_precedence_constraints(self, routing, manager, time_dimension, vrp_moves, service_times):
        """
        OPTIMIZED: More efficient precedence constraint addition.
        """
        solver = routing.solver()

        if self.verbose:
            print("  Enforcing VRP precedence constraints.")

        # Create a dictionary to hold full service times for each move (travel + handling)
        service_times_dict = {}
        for i, move in enumerate(vrp_moves):
            # service_times is a list: [depot_time, move1_time, move2_time, ...]
            # So for move at index i in vrp_moves, the service time is at service_times[i+1]
            service_times_dict[move.move_id] = service_times[i + 1]
    
        ul_moves = {}
        for move in vrp_moves:
            if move.ul_id not in ul_moves:
                ul_moves[move.ul_id] = []
            ul_moves[move.ul_id].append(move)

        for ul_id, moves in ul_moves.items():
            if len(moves) > 1:
                sorted_moves = sorted(moves, key=lambda m: m.move_id)
                for i in range(len(sorted_moves) - 1):
                    earlier_move = sorted_moves[i]
                    later_move = sorted_moves[i+1]
                    earlier_node_idx = manager.NodeToIndex(earlier_move.move_id + 1)
                    later_node_idx = manager.NodeToIndex(later_move.move_id + 1)
                
                    # The full service time of the earlier move must complete before the next move for that same UL can start.
                    earlier_service_time = service_times_dict[earlier_move.move_id]
                
                    solver.Add(
                        time_dimension.CumulVar(later_node_idx) >=
                        time_dimension.CumulVar(earlier_node_idx) + earlier_service_time
                    )
                    if self.verbose:
                        print(f"    VRP PRECEDENCE (UL {ul_id}): Move {earlier_move.move_id} must precede Move {later_move.move_id}")

        # --- 2. LIFO and Lane Exclusivity Precedence ---
        if self.verbose:
            print("  Enforcing LIFO lane exclusivity constraints.")
    
        lane_moves = {}
        # Define locations that should not have precedence constraints (source/sink are not buffer lanes)
        excluded_locations = {'source', 'sink', self.source_lane_id, self.sink_lane_id}
        excluded_locations.discard(None)

        for move in vrp_moves:
            # Skip direct retrievals (source->sink) as they don't use buffer lanes
            # and shouldn't block buffer operations
            if move.move_type == 'direct_retrieve':
                continue
                
            # A move 'uses' a lane if it stores into it or retrieves/reshuffles from it.
            locations_used = {move.from_location, move.to_location}
            for lane_id in locations_used:
                if lane_id not in excluded_locations:
                    if lane_id not in lane_moves:
                        lane_moves[lane_id] = []
                    lane_moves[lane_id].append(move)

        for lane_id, moves in lane_moves.items():
            # Remove duplicates if a move uses the same lane as from and to (e.g., reshuffle within the same lane)
            unique_moves = list({m.move_id: m for m in moves}.values())
            if len(unique_moves) > 1:
                sorted_moves = sorted(unique_moves, key=lambda m: m.move_id)
                
                # Add precedence constraints for LIFO ordering
                for i in range(len(sorted_moves) - 1):
                    earlier_move = sorted_moves[i]
                    later_move = sorted_moves[i+1]
                    earlier_node_idx = manager.NodeToIndex(earlier_move.move_id + 1)
                    later_node_idx = manager.NodeToIndex(later_move.move_id + 1)
                    
                    # We need the duration 'earlier_move' occupies 'lane_id'.
                    occupancy_loc = lane_id
                    occupancy_tier = 1
                    
                    if earlier_move.from_location == lane_id:
                        occupancy_tier = earlier_move.from_tier
                    elif earlier_move.to_location == lane_id:
                        occupancy_tier = earlier_move.to_tier
                    
                    # Calculate the *actual* time the lane is busy
                    lane_occupancy_duration = self._get_lane_occupancy_duration(
                        occupancy_loc, occupancy_tier
                    )
                    
                    # The constraint is:
                    # Start(B) >= Start(A) + lane_occupancy_duration(A)
                    solver.Add(
                        time_dimension.CumulVar(later_node_idx) >=
                        time_dimension.CumulVar(earlier_node_idx) + lane_occupancy_duration
                    )
                    
                    if self.verbose:
                        print(f"    LIFO CONSTRAINT (Lane {lane_id}): "
                              f"Move {later_move.move_id} (UL {later_move.ul_id}) must start after "
                              f"Move {earlier_move.move_id} (UL {earlier_move.ul_id}) *clears the lane* "
                              f"(duration: {lane_occupancy_duration})")



def solve_twvrp_with_ortools(buffer, moves: List[Dict], num_vehicles: int = 1, instance=None, time_limit: Optional[int] = None, verbose: bool = False) -> Dict:
    """
    Module-level helper to solve the time window VRP using OR-Tools.
    """
    solver = TWVRPORToolsSolver(buffer, num_vehicles, vehicle_capacity=1, instance=instance, verbose=verbose)
    return solver.solve_twvrp(moves, time_limit)