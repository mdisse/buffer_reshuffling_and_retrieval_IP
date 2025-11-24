"""
Time Window Vehicle Routing Problem Solver using OR-Tools CP-SAT Scheduling

This module solves the vehicle routing problem as a job-shop scheduling problem
with time windows, precedence constraints, and resource constraints (vehicles).

Key differences from VRP approach:
- Uses CP-SAT (constraint programming) instead of routing solver
- Models moves as jobs/tasks with time intervals
- Enforces precedence and resource constraints explicitly
- Better suited for tight time windows and complex precedence chains
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from math import ceil
from functools import lru_cache

from ortools.sat.python import cp_model

from src.bay import tier

# [NEW] Change 1: Import the collision repairer and traceback
from .vrp_collision_repair import VRPCollisionRepairer
import traceback


@dataclass
class SchedulingMove:
    """Represents a move/task in the scheduling problem."""
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
    # For direct_retrieve: separate windows for pickup and delivery
    pickup_earliest: Optional[int] = None  # Earliest pickup time from source (arrival window start)
    pickup_latest: Optional[int] = None    # Latest pickup time from source (arrival window end)
    delivery_earliest: Optional[int] = None  # Earliest delivery time to sink (retrieval window start)
    delivery_latest: Optional[int] = None    # Latest delivery time to sink (retrieval window end)
    # HARD constraint bounds (cannot be violated)
    hard_start_lower_bound: Optional[int] = None  # Cannot start before this time
    hard_start_upper_bound: Optional[int] = None  # Cannot start after this time
    hard_end_lower_bound: Optional[int] = None    # Cannot end before this time
    hard_end_upper_bound: Optional[int] = None    # Cannot end after this time


class TWVRPSchedulingSolver:
    """
    Solves the TWVRP as a scheduling problem using OR-Tools CP-SAT.
    
    The problem is modeled as:
    - Tasks (moves) with time windows [earliest_start, latest_finish]
    - Resources (vehicles) with capacity constraints
    - Precedence constraints between tasks
    - Lane occupancy constraints (only one vehicle per lane at a time)
    """
    
    def __init__(self, buffer, num_vehicles: int = 1, vehicle_capacity: int = 1, 
                 instance=None, verbose: bool = False):
        """
        Initialize the scheduling solver.
        
        Args:
            buffer: Buffer/warehouse layout
            num_vehicles: Number of AMRs available
            vehicle_capacity: Capacity of each AMR (typically 1)
            instance: Problem instance with unit loads and time windows
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.buffer = buffer
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.instance = instance
        
        # Get source and sink lanes
        self.source_lane_id = str(self.buffer.get_source().get_ap_id()) if self.buffer.get_source() else None
        self.sink_lane_id = str(self.buffer.get_sink().get_ap_id()) if self.buffer.get_sink() else None
        self.lane_map = {lane.get_ap_id(): lane for lane in self.buffer.get_virtual_lanes()}
        
        # Cache lane slots for occupancy calculations
        self.lane_n_slots = {}
        for lane in self.buffer.get_virtual_lanes():
            n_slots = len(lane.get_tiers())
            # Store with string keys for consistency with parsed locations
            self.lane_n_slots[str(lane.get_ap_id())] = n_slots
        
        # Initialize caches
        self._location_parse_cache = {}
        self._location_tier_parse_cache = {}
        self._travel_time_cache = {}
    
    def solve_twvrp(self, moves: List[Dict], time_limit: Optional[int] = None) -> Dict:
        """
        Solve the time window VRP for the given move sequence using scheduling.
        
        Args:
            moves: List of move dicts (from A* or heuristic).
            time_limit: Max time in seconds for solver.
        
        Returns:
            Dict with vehicle assignments, total distance, total time, and status.
        """
        if not moves:
            return {
                'vehicles': [], 
                'total_distance': 0, 
                'total_time': 0, 
                'solver_status': 'no_moves'
            }
        
        # Convert moves to scheduling tasks
        scheduling_moves = self._convert_moves_to_scheduling_jobs(moves)
        
        # Solve using CP-SAT
        return self._solve_with_cpsat(scheduling_moves, time_limit)
    
    def _convert_moves_to_scheduling_jobs(self, moves: List[Dict]) -> List[SchedulingMove]:
        """
        Convert A* moves to scheduling jobs with HARD time windows and precedence.
        
        Time windows are enforced as HARD CONSTRAINTS through CP-SAT variable domains:
        - For storage: pickup must occur within arrival time window
        - For retrieval: delivery must occur within retrieval window
        - For direct retrieval: pickup in arrival window, delivery in retrieval window
        
        The CP-SAT solver cannot violate these bounds - if no solution exists
        within the time windows, the solver will return INFEASIBLE.
        """
        scheduling_moves = []
        
        for i, move in enumerate(moves):
            ul_id = move.get('ul_id', 0)
            move_type = move.get('type', 'unknown')
            
            from_pos, from_tier_parsed = self._parse_location_and_tier_str(move.get('from_pos', 'source'))
            to_pos, to_tier_parsed = self._parse_location_and_tier_str(move.get('to_pos', 'sink'))
            from_tier = move.get('from_tier', from_tier_parsed)
            to_tier = move.get('to_tier', to_tier_parsed)
            
            earliest_start, latest_finish = 0, 99999
            
            # Get unit load for time window information
            ul = next((u for u in self.instance.get_unit_loads() if u.get_id() == ul_id), None)
            if ul:
                service_time = self._calculate_gurobi_style_travel_time(
                    from_pos, to_pos, from_tier, to_tier, handling_time_enabled=True
                )
                
                if move_type == 'store':
                    # Store: pickup from source must happen during arrival window
                    # Start time = pickup time from source (must be in [arrival_start, arrival_end])
                    # End time = dropoff time at storage location
                    # 
                    # IMPORTANT: Time windows from instance are in Gurobi format (1-based).
                    # CP-SAT uses 0-based times, so we subtract 1 to convert.
                    # After solving, we'll add 1 back when extracting the solution.
                    arrival_start = (ul.get_arrival_start() or 0) - 1
                    arrival_end = (ul.get_arrival_end() or 99999) - 1
                    
                    # CRITICAL: The constraint is on the PICKUP TIME (start time)
                    # earliest_start/latest_finish define the soft window for tardiness calculation
                    # For store operations, this window applies to the START time (pickup)
                    earliest_start = max(0, arrival_start)  # Earliest pickup at arrival_start
                    latest_finish = arrival_end  # Latest pickup at arrival_end
                    # Note: The END time will be start_time + service_time
                
                elif move_type == 'retrieve':
                    # Retrieve: pickup from storage, deliver to sink during retrieval window
                    # Start time = pickup time from storage location
                    # End time = delivery time at sink (must be in [retrieval_start, retrieval_end])
                    # 
                    # IMPORTANT: Convert from Gurobi (1-based) to CP-SAT (0-based)
                    retrieval_start = (ul.get_retrieval_start() or 0) - 1
                    retrieval_end = (ul.get_retrieval_end() or 99999) - 1
                    
                    # Must arrive at sink within retrieval window
                    earliest_start = max(0, retrieval_start - service_time)  # Can start early but must arrive in window
                    latest_finish = retrieval_end  # Must arrive at sink by retrieval_end
                
                elif move_type == 'direct_retrieve':
                    # Direct retrieve: pickup from source during arrival window, deliver to sink during retrieval window
                    # Start time = pickup from source (must be in [arrival_start, arrival_end])
                    # End time = delivery at sink (must be in [retrieval_start, retrieval_end])
                    # 
                    # IMPORTANT: Convert from Gurobi (1-based) to CP-SAT (0-based)
                    arrival_start = (ul.get_arrival_start() or 0) - 1
                    arrival_end = (ul.get_arrival_end() or 99999) - 1
                    retrieval_start = (ul.get_retrieval_start() or 0) - 1
                    retrieval_end = (ul.get_retrieval_end() or 99999) - 1
                    
                    # For direct_retrieve, we need to track both windows separately
                    # Store them as attributes on the move object for later constraint creation
                    earliest_start = max(0, arrival_start)  # Can pick up from source starting at arrival_start
                    latest_finish = retrieval_end  # Must deliver to sink by retrieval_end
                    
                    # We'll add explicit constraints in the CP-SAT model to enforce:
                    # 1. start_var in [arrival_start, arrival_end]
                    # 2. end_var in [retrieval_start, retrieval_end]
                    
                    # Check feasibility: can we pick up in arrival window and deliver in retrieval window?
                    if arrival_end + service_time > retrieval_end:
                        if self.verbose:
                            print(f"  WARNING: Potentially infeasible direct_retrieve for UL {ul_id}: "
                                  f"latest pickup at {arrival_end} + service {service_time} = {arrival_end + service_time} > retrieval_end {retrieval_end}")
                
                elif move_type == 'reshuffle':
                    # Reshuffle: internal move, no hard time constraints from UL windows
                    # Use default wide window
                    earliest_start = 0
                    latest_finish = 99999
                
                # Ensure window is feasible
                if latest_finish < earliest_start:
                    if self.verbose:
                        print(f"  WARNING: Infeasible time window for move {i} (UL {ul_id}, {move_type}). "
                              f"Adjusting [{earliest_start}, {latest_finish}] -> [{earliest_start}, {earliest_start}]")
                    latest_finish = earliest_start
            
            # Apply bounds to ensure non-negative times
            final_earliest_start = max(0, earliest_start)
            final_latest_finish = max(final_earliest_start, latest_finish)
            
            # Set HARD constraint bounds (these CANNOT be violated)
            # Keep HARD bounds minimal to avoid over-constraining the problem
            hard_start_lower_bound = None
            hard_end_upper_bound = None
            hard_end_lower_bound = None
            
            # ONLY set HARD bounds for truly non-negotiable physical constraints
            # Time windows should be SOFT (enforced via tardiness penalties in objective)
            
            # For retrieval: delivery to sink MUST happen before retrieval deadline (external constraint)
            if move_type == 'retrieve' and ul:
                retrieval_end = (ul.get_retrieval_end() or 99999) - 1
                hard_end_upper_bound = retrieval_end
                # Also enforce retrieval start (cannot deliver before retrieval window starts)
                retrieval_start = (ul.get_retrieval_start() or 0) - 1
                hard_end_lower_bound = retrieval_start
            
            if move_type == 'direct_retrieve' and ul:
                # HARD: Must deliver to sink before retrieval deadline
                retrieval_end = (ul.get_retrieval_end() or 99999) - 1
                hard_end_upper_bound = retrieval_end
                # Also enforce retrieval start
                retrieval_start = (ul.get_retrieval_start() or 0) - 1
                hard_end_lower_bound = retrieval_start
            
            # For direct_retrieve, store the separate windows
            pickup_earliest = None
            pickup_latest = None
            delivery_earliest = None
            delivery_latest = None
            
            if move_type == 'direct_retrieve' and ul:
                arrival_start = (ul.get_arrival_start() or 0) - 1
                arrival_end = (ul.get_arrival_end() or 99999) - 1
                retrieval_start = (ul.get_retrieval_start() or 0) - 1
                retrieval_end = (ul.get_retrieval_end() or 99999) - 1
                
                pickup_earliest = max(0, arrival_start)
                pickup_latest = arrival_end
                delivery_earliest = retrieval_start
                delivery_latest = retrieval_end

            # Enforce hard start lower bound for store moves so AMRs cannot depart before UL arrival
            # Also enforce for direct_retrieve pickup window start
            if move_type == 'store' and ul:
                # earliest_start already converted to 0-based CP-SAT time
                # We should allow starting at the arrival time (which is earliest_start)
                hard_start_lower_bound = final_earliest_start

            if move_type == 'direct_retrieve' and pickup_earliest is not None:
                hard_start_lower_bound = pickup_earliest

            scheduling_moves.append(SchedulingMove(
                move_id=i,
                from_location=from_pos,
                to_location=to_pos,
                ul_id=ul_id,
                move_type=move_type,
                travel_distance=self._calculate_distance(from_pos, to_pos, from_tier, to_tier),
                service_time=service_time if ul else 1,
                earliest_start=final_earliest_start,
                latest_finish=final_latest_finish,
                from_tier=from_tier,
                to_tier=to_tier,
                pickup_earliest=pickup_earliest,
                pickup_latest=pickup_latest,
                delivery_earliest=delivery_earliest,
                delivery_latest=delivery_latest,
                hard_start_lower_bound=hard_start_lower_bound,
                hard_start_upper_bound=None,  # Not using hard upper bounds on start
                hard_end_lower_bound=hard_end_lower_bound,
                hard_end_upper_bound=hard_end_upper_bound
            ))
            
            if self.verbose and ul:
                print(f"  Move {i}: UL{ul_id} {move_type} {from_pos}->{to_pos}, service={service_time if ul else 1}, window=[{final_earliest_start}, {final_latest_finish}]")
                if move_type in ['retrieve', 'direct_retrieve']:
                    print(f"    (retrieval deadline: {ul.get_retrieval_end()})")
        
        return scheduling_moves
    
    def _solve_with_cpsat(self, scheduling_moves: List[SchedulingMove], 
                          time_limit: Optional[int] = None) -> Dict:
        """
        Solve the scheduling problem using OR-Tools CP-SAT solver.
        
        Uses HYBRID constraint approach:
        1. HARD constraints (enforced via variable domains):
           - Storage: start >= arrival_window_start (cannot store before item arrives)
           - Retrieval: end <= retrieval_window_end (cannot retrieve after deadline)
        
        2. SOFT constraints (minimized in objective via slack):
           - All other time window bounds are soft and penalized when violated
        
        Objective = (total_tardiness * 10000) + makespan
        
        This ensures the solver prioritizes meeting time windows,
        and among those, picks the one with the smallest makespan.
        """
        model = cp_model.CpModel()
        
        # Determine time horizon - use a large value to allow tardiness
        horizon = max(move.latest_finish + move.service_time for move in scheduling_moves)
        horizon = max(horizon * 2, 100000)  # Allow extra time for tardy solutions
        
        if self.verbose:
            print(f"\n=== Scheduling Problem (Hybrid Hard/Soft Time Windows) ===")
            print(f"Moves: {len(scheduling_moves)}")
            print(f"Vehicles: {self.num_vehicles}")
            print(f"Horizon: {horizon}")
            print(f"HARD constraints: storage start >= arrival_start, retrieval end <= retrieval_end")
            print(f"SOFT constraints: all other time window bounds (minimized in objective)")
            print(f"Objective: Minimize (tardiness * 10000 + makespan)")
        
        # Create variables for each move
        move_intervals = {}
        move_starts = {}
        move_ends = {}
        move_vehicles = {}
        move_tardiness = {}  # Track tardiness for each move
        
        for move in scheduling_moves:
            # Determine variable bounds based on HARD constraints
            # Start variable: must respect hard_start_lower_bound and hard_start_upper_bound if set
            start_lower = move.hard_start_lower_bound if move.hard_start_lower_bound is not None else 0
            start_upper = move.hard_start_upper_bound if move.hard_start_upper_bound is not None else horizon
            
            # End variable: must respect hard_end_upper_bound if set
            end_lower = start_lower + move.service_time  # End must be at least start + service
            if move.hard_end_lower_bound is not None:
                end_lower = max(end_lower, move.hard_end_lower_bound)
            
            end_upper = move.hard_end_upper_bound if move.hard_end_upper_bound is not None else horizon
            
            # Ensure valid domains
            if start_lower > start_upper:
                if self.verbose:
                    print(f"  ERROR: Infeasible HARD start constraints for move {move.move_id} (UL {move.ul_id}, {move.move_type})")
                    print(f"    start must be in [{start_lower}, {start_upper}] - INFEASIBLE")
                # Set to minimal valid range to let solver detect infeasibility
                start_upper = start_lower
                
            if end_lower > end_upper:
                if self.verbose:
                    print(f"  ERROR: Infeasible HARD constraints for move {move.move_id} (UL {move.ul_id}, {move.move_type})")
                    print(f"    start >= {start_lower}, end <= {end_upper}, service = {move.service_time}")
                    print(f"    Minimum end ({end_lower}) > Maximum end ({end_upper})")
                # Set to minimal valid range to let solver detect infeasibility
                end_upper = end_lower
            
            start_var = model.NewIntVar(start_lower, start_upper, f'start_m{move.move_id}')
            end_var = model.NewIntVar(end_lower, end_upper, f'end_m{move.move_id}')
            
            # Create interval variable for the move
            interval_var = model.NewIntervalVar(
                start_var, 
                move.service_time, 
                end_var,
                f'interval_m{move.move_id}'
            )
            
            # Vehicle assignment variable
            vehicle_var = model.NewIntVar(0, self.num_vehicles - 1, f'vehicle_m{move.move_id}')
            
            # SOFT TARDINESS CALCULATION
            # Track violations of the SOFT time window bounds (earliest_start, latest_finish)
            # The HARD bounds are already enforced in the variable domains above
            
            # For direct_retrieve moves, we have TWO separate windows to check:
            # 1. Pickup window: start_var must be in [pickup_earliest, pickup_latest]
            # 2. Delivery window: end_var must be in [delivery_earliest, delivery_latest]
            
            if move.move_type == 'direct_retrieve' and move.pickup_earliest is not None:
                # Direct retrieve has separate pickup and delivery windows
                
                # Pickup window tardiness (start time violations)
                pickup_early = model.NewIntVar(0, horizon, f'pickup_early_m{move.move_id}')
                model.AddMaxEquality(pickup_early, [0, move.pickup_earliest - start_var])
                
                pickup_late = model.NewIntVar(0, horizon, f'pickup_late_m{move.move_id}')
                model.AddMaxEquality(pickup_late, [0, start_var - move.pickup_latest])
                
                # Delivery window tardiness (end time violations)
                delivery_early = model.NewIntVar(0, horizon, f'delivery_early_m{move.move_id}')
                model.AddMaxEquality(delivery_early, [0, move.delivery_earliest - end_var])
                
                delivery_late = model.NewIntVar(0, horizon, f'delivery_late_m{move.move_id}')
                model.AddMaxEquality(delivery_late, [0, end_var - move.delivery_latest])
                
                # Total tardiness = sum of all window violations
                tardiness_var = model.NewIntVar(0, horizon * 4, f'tardiness_m{move.move_id}')
                model.Add(tardiness_var == pickup_early + pickup_late + delivery_early + delivery_late)
            else:
                # Standard move: single time window
                # For STORE operations: soft window applies to START time (pickup window)
                # For RETRIEVE operations: soft window applies to END time (delivery window)
                
                # Early penalty (starting before earliest_start)
                early_penalty = model.NewIntVar(0, horizon, f'early_m{move.move_id}')
                model.AddMaxEquality(early_penalty, [0, move.earliest_start - start_var])
                
                # Late penalty
                late_penalty = model.NewIntVar(0, horizon, f'late_m{move.move_id}')
                if move.move_type == 'store':
                    # CRITICAL FIX: For store operations, soft constraint is on START time (pickup)
                    # latest_finish already represents the latest PICKUP time (arrival_end)
                    model.AddMaxEquality(late_penalty, [0, start_var - move.latest_finish])
                else:
                    # For retrieve/empty operations: soft constraint is on END time (delivery/completion)
                    model.AddMaxEquality(late_penalty, [0, end_var - move.latest_finish])
                
                # Total tardiness for this move
                tardiness_var = model.NewIntVar(0, horizon * 2, f'tardiness_m{move.move_id}')
                model.Add(tardiness_var == early_penalty + late_penalty)
            
            move_starts[move.move_id] = start_var
            move_ends[move.move_id] = end_var
            move_intervals[move.move_id] = interval_var
            move_vehicles[move.move_id] = vehicle_var
            move_tardiness[move.move_id] = tardiness_var
            
            if self.verbose and move.ul_id > 0:
                hard_info = []
                if move.hard_start_lower_bound is not None:
                    hard_info.append(f"start>={move.hard_start_lower_bound}")
                if move.hard_start_upper_bound is not None:
                    hard_info.append(f"start<={move.hard_start_upper_bound}")
                if move.hard_end_upper_bound is not None:
                    hard_info.append(f"end<={move.hard_end_upper_bound}")
                hard_str = f" HARD:[{', '.join(hard_info)}]" if hard_info else ""
                
                print(f"  Move {move.move_id}: UL{move.ul_id} {move.move_type} "
                      f"{move.from_location}->{move.to_location}, "
                      f"soft_window=[{move.earliest_start}, {move.latest_finish}]{hard_str}, "
                      f"service={move.service_time}")
        
        # Add precedence constraints (move i must finish before move i+1 starts)
        precedence_rules = self._get_precedence_rules(scheduling_moves)
        
        if self.verbose:
            print(f"Unit load precedence constraints: {len(precedence_rules)}")
        
        for earlier_id, later_id in precedence_rules:
            model.Add(move_ends[earlier_id] <= move_starts[later_id])
        
        # Add LIFO constraints (upper tiers must be retrieved before lower tiers)
        lifo_rules = self._get_lifo_constraints(scheduling_moves)
        
        if self.verbose:
            print(f"LIFO constraints: {len(lifo_rules)}")
        
        for upper_retrieve_id, lower_retrieve_id in lifo_rules:
            model.Add(move_ends[upper_retrieve_id] <= move_starts[lower_retrieve_id])
        
        # NOTE: Lane sequencing constraints are REMOVED to avoid over-constraining
        # The lane occupancy (no-overlap) constraint already ensures moves in the same lane
        # don't physically conflict. Precedence constraints ensure LIFO order.
        # Lane sequencing was causing infeasibility by forcing too strict ordering.
        
        # Create vehicle-specific constraints FIRST
        # We need move_assignment_vars for the lane occupancy constraints
        # For each vehicle, create optional intervals (move is assigned to this vehicle or not)
        vehicle_intervals = {v: [] for v in range(self.num_vehicles)}
        move_assignment_vars = {}  # Store assignment bool vars: (move_id, vehicle) -> bool_var
        
        for move in scheduling_moves:
            move_id = move.move_id
            for v in range(self.num_vehicles):
                # Create a boolean variable: is this move assigned to vehicle v?
                is_assigned = model.NewBoolVar(f'assign_m{move_id}_v{v}')
                move_assignment_vars[(move_id, v)] = is_assigned
                
                # Link to vehicle assignment variable
                model.Add(move_vehicles[move_id] == v).OnlyEnforceIf(is_assigned)
                
                # Create optional interval for this vehicle
                optional_interval = model.NewOptionalIntervalVar(
                    move_starts[move_id],
                    move.service_time,
                    move_ends[move_id],
                    is_assigned,
                    f'optional_m{move_id}_v{v}'
                )
                vehicle_intervals[v].append(optional_interval)
        
        # Each move must be assigned to exactly one vehicle
        for move in scheduling_moves:
            model.Add(sum(
                move_assignment_vars[(move.move_id, v)]
                for v in range(self.num_vehicles)
            ) == 1)
        
        # No overlap constraint for each vehicle (each vehicle can only do one move at a time)
        for v in range(self.num_vehicles):
            if len(vehicle_intervals[v]) > 1:
                model.AddNoOverlap(vehicle_intervals[v])
        
        # ========================================
        # LANE OCCUPANCY CONSTRAINTS
        # ========================================
        # Add lane occupancy constraints (only one vehicle per lane at a time)
        # CRITICAL FIX: Use optional intervals tied to vehicle assignments
        # This ensures each move only occupies the lane when it's actually assigned to a vehicle
        
        lane_intervals = {}  # lane_id -> list of optional interval_vars
        
        for move in scheduling_moves:
            from_lane = self._parse_location_str(move.from_location)
            to_lane = self._parse_location_str(move.to_location)
            
            # For each vehicle, create optional intervals for this move's lane usage
            for v in range(self.num_vehicles):
                is_assigned = move_assignment_vars[(move.move_id, v)]
                
                # Add from_lane occupancy (if it's a buffer lane)
                if from_lane not in ['source', 'sink']:
                    # Block the from_lane for the entire move duration IF assigned to vehicle v
                    lane_interval = model.NewOptionalIntervalVar(
                        move_starts[move.move_id],
                        move.service_time,
                        move_ends[move.move_id],
                        is_assigned,
                        f'lane_occupancy_{from_lane}_m{move.move_id}_v{v}_from'
                    )
                    
                    if from_lane not in lane_intervals:
                        lane_intervals[from_lane] = []
                    lane_intervals[from_lane].append(lane_interval)
                
                # Add to_lane occupancy (if it's a buffer lane and different from from_lane)
                if to_lane not in ['source', 'sink'] and to_lane != from_lane:
                    # Block the to_lane for the entire move duration IF assigned to vehicle v
                    lane_interval = model.NewOptionalIntervalVar(
                        move_starts[move.move_id],
                        move.service_time,
                        move_ends[move.move_id],
                        is_assigned,
                        f'lane_occupancy_{to_lane}_m{move.move_id}_v{v}_to'
                    )
                    
                    if to_lane not in lane_intervals:
                        lane_intervals[to_lane] = []
                    lane_intervals[to_lane].append(lane_interval)
        
        # Add no-overlap constraints for each lane (only one robot at a time)
        for lane_id, intervals_list in lane_intervals.items():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)
        
        if self.verbose:
            print(f"Lane occupancy constraints: {sum(len(v) for v in lane_intervals.values())} intervals across {len(lane_intervals)} lanes")
        
        # ========================================
        # LOCATION CONTINUITY CONSTRAINTS
        # ========================================
        # Critical: A vehicle can only execute a move if it is physically at that location.
        # For each PAIR of moves (i, j), we must enforce repositioning time
        # for both possible sequences: (i -> j) and (j -> i).
        
        for v in range(self.num_vehicles):
            # For each unique pair of moves (i, j) where i < j
            for i, move_i in enumerate(scheduling_moves):
                for j, move_j in enumerate(scheduling_moves):
                    if i >= j:  # Get each unique pair (i, j) only once
                        continue
                    
                    # Get assignment bools for this vehicle
                    i_on_v = move_assignment_vars[(move_i.move_id, v)]
                    j_on_v = move_assignment_vars[(move_j.move_id, v)]
                    
                    # --- Case 1: Sequence i -> j ---
                    
                    # Repositioning time from i's end to j's start
                    reposition_time_i_to_j = self._calculate_gurobi_style_travel_time(
                        move_i.to_location, move_j.from_location,
                        move_i.to_tier, move_j.from_tier,
                        handling_time_enabled=False
                    )
                    
                    # Bool var for temporal order: end_i <= start_j
                    i_before_j = model.NewBoolVar(f'seq_m{move_i.move_id}_before_m{move_j.move_id}')
                    model.Add(move_ends[move_i.move_id] <= move_starts[move_j.move_id]).OnlyEnforceIf(i_before_j)
                    model.Add(move_ends[move_i.move_id] > move_starts[move_j.move_id]).OnlyEnforceIf(i_before_j.Not())

                    # CRITICAL: Enforce minimum separation between consecutive moves
                    # Removed +1 offset as CP-SAT end is exclusive and maps correctly to Gurobi start
                    min_separation = reposition_time_i_to_j
                    
                    # Check if we need to enforce "No Waiting" (exact abutment)
                    # This applies if we are at a buffer lane (not source/sink) and staying there (reposition=0)
                    # Gurobi model forbids waiting at buffer lanes.
                    # UPDATE: Gurobi model now allows waiting at buffer lanes (via stationary vs moving logic).
                    # So we relax this to allow waiting (>=) instead of exact abutment (==).
                    is_buffer_lane_i = move_i.to_location not in [self.source_lane_id, self.sink_lane_id, 'source', 'sink']
                    force_no_wait_i_j = False # (reposition_time_i_to_j == 0) and is_buffer_lane_i
                    
                    if force_no_wait_i_j:
                         # Enforce exact abutment: start_j == end_i
                         model.Add(
                            move_starts[move_j.move_id] == move_ends[move_i.move_id]
                        ).OnlyEnforceIf([i_on_v, j_on_v, i_before_j])
                    else:
                        # Standard precedence with travel time
                        model.Add(
                            move_starts[move_j.move_id] >= move_ends[move_i.move_id] + min_separation
                        ).OnlyEnforceIf([i_on_v, j_on_v, i_before_j])
                    
                    # --- Case 2: Sequence j -> i ---
                    
                    # Repositioning time from j's end to i's start
                    reposition_time_j_to_i = self._calculate_gurobi_style_travel_time(
                        move_j.to_location, move_i.from_location,
                        move_j.to_tier, move_i.from_tier,
                        handling_time_enabled=False
                    )
                    
                    min_separation = reposition_time_j_to_i
                    
                    is_buffer_lane_j = move_j.to_location not in [self.source_lane_id, self.sink_lane_id, 'source', 'sink']
                    force_no_wait_j_i = False # (reposition_time_j_to_i == 0) and is_buffer_lane_j
                    
                    if force_no_wait_j_i:
                        model.Add(
                            move_starts[move_i.move_id] == move_ends[move_j.move_id]
                        ).OnlyEnforceIf([i_on_v, j_on_v, i_before_j.Not()])
                    else:
                        model.Add(
                            move_starts[move_i.move_id] >= move_ends[move_j.move_id] + min_separation
                        ).OnlyEnforceIf([i_on_v, j_on_v, i_before_j.Not()])
        
        # ========================================
        # INITIAL VEHICLE POSITION CONSTRAINTS
        # ========================================
        # All vehicles start at the sink, but they don't have to depart at time 0.
        # They can wait at the sink and depart whenever needed.
        # 
        # For each vehicle, we create a "departure time" variable that represents
        # when the vehicle leaves the sink for its first move.
        
        vehicle_departure_times = {}
        for v in range(self.num_vehicles):
            # Vehicle can depart from sink at any time in [0, horizon]
            vehicle_departure_times[v] = model.NewIntVar(0, horizon, f'vehicle_{v}_departure')
        
        # The location continuity constraints will handle repositioning between subsequent moves.
        # We need to ensure the very first move on each vehicle accounts for travel from sink.
        
        for v in range(self.num_vehicles):
            for move in scheduling_moves:
                # Calculate repositioning time from sink to this move's start location
                initial_reposition_time = self._calculate_gurobi_style_travel_time(
                    self.sink_lane_id, move.from_location,
                    1, move.from_tier,
                    handling_time_enabled=False
                )
                
                # Only enforce this if the move is assigned to this vehicle
                is_assigned = move_assignment_vars[(move.move_id, v)]
                
                # Vehicle departs from sink at vehicle_departure_times[v]
                # Must travel initial_reposition_time to reach from_location  
                # Constraint: move_start >= vehicle_departure + initial_reposition_time
                # 
                # NOTE: No +1 separation here! The vehicle can start the move immediately
                # upon arriving at the location. The +1 separation only applies BETWEEN
                # consecutive moves (handled by location continuity constraints).
                
                # CRITICAL: This constraint should only apply to the FIRST move of the vehicle.
                # For subsequent moves, the location continuity constraints handle repositioning.
                # However, we can't easily identify which move is "first" before solving.
                # 
                # Solution: We enforce this for ALL moves, but also ensure vehicle_departure
                # is set to the minimum value that satisfies all assigned moves.
                # This works because:
                # - If move M is first: constraint is active and correct
                # - If move M is not first: the location continuity constraint from the
                #   previous move will be stricter, making this constraint redundant
                
                model.Add(
                    move_starts[move.move_id] >= vehicle_departure_times[v] + initial_reposition_time
                ).OnlyEnforceIf(is_assigned)
        
        # NOTE: Source/sink non-blocking constraints removed - they were over-constraining
        # the model and causing infeasibility. The core constraints (precedence, lane occupancy,
        # location continuity) are sufficient to ensure valid schedules.
        
        # NOTE: Lane cooldown constraints also removed - the lane no-overlap constraint
        # already ensures only one robot in a lane at a time, which is the critical requirement.
        
        # NOTE: LIFO lane filling constraints REMOVED - they were too strict and causing infeasibility
        # The A* solution already respects LIFO in tier assignments, and the following constraints
        # are sufficient to ensure valid execution:
        # 1. Precedence constraints (same unit load moves maintain order)
        # 2. Lane occupancy constraints (no physical conflicts in lanes)
        # 3. Location continuity constraints (robots must travel between locations)
        # 
        # The tier assignments from A* are trusted - if A* says store in tier 2, we trust that
        # tier 1 is already occupied (or will be by the time this move executes, enforced by precedence).
        
        # ========================================
        # HIERARCHICAL OBJECTIVE
        # ========================================
        # Primary: Minimize total tardiness (time window violations)
        # Secondary: Minimize makespan (completion time)
        # 
        # We use a weighted sum with a large multiplier (10000) to create
        # a hierarchical effect: tardiness is minimized first, then makespan
        
        # Calculate total tardiness across all moves
        total_tardiness = model.NewIntVar(0, horizon * len(scheduling_moves) * 2, 'total_tardiness')
        model.Add(total_tardiness == sum(move_tardiness[m.move_id] for m in scheduling_moves))
        
        # Calculate sum of completion times (total flow time)
        sum_of_end_times = model.NewIntVar(0, horizon * len(scheduling_moves), 'sum_of_end_times')
        model.Add(sum_of_end_times == sum(move_ends[m.move_id] for m in scheduling_moves))
        
        # Hierarchical objective: prioritize tardiness, then sum of end times
        # Using multiplier of 10000 ensures 1 unit of tardiness is worse than 10000 units of flow time
        objective = model.NewIntVar(0, horizon * len(scheduling_moves) * 20000, 'objective')
        model.Add(objective == total_tardiness * 10000 + sum_of_end_times)
        
        model.Minimize(objective)
        
        if self.verbose:
            print(f"\n🎯 Objective: Minimize (tardiness * 10000 + sum of end times)")
            print(f"   This prioritizes meeting time windows over minimizing total flow time")
        
        # Solve the model
        solver = cp_model.CpSolver()
        
        # Configure solver for better solutions with soft constraints
        if time_limit:
            solver.parameters.max_time_in_seconds = time_limit
        else:
            # Default to 60 seconds for soft constraint solving
            solver.parameters.max_time_in_seconds = 60
        
        # Improve search quality
        solver.parameters.num_search_workers = 8  # Use multiple threads
        solver.parameters.log_search_progress = self.verbose
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2  # More aggressive linearization
        solver.parameters.cp_model_probing_level = 2  # More probing
        
        if self.verbose:
            print(f"\nSolving scheduling problem with hard time window constraints...")
            print(f"  Time limit: {solver.parameters.max_time_in_seconds}s")
            print(f"  Search workers: {solver.parameters.num_search_workers}")
        
        status = solver.Solve(model)
        
        # Debug: print objective value from solver
        if self.verbose and (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE):
            obj_value = solver.ObjectiveValue()
            # Extract tardiness and sum of end times from the objective
            tardiness_value = solver.Value(total_tardiness)
            sum_of_end_times_value = solver.Value(sum_of_end_times)
            
            print(f"\n🎯 CP-SAT Solution Found!")
            print(f"   Status: {solver.StatusName(status)}")
            print(f"   Total Tardiness: {tardiness_value} time units")
            print(f"   Sum of End Times: {sum_of_end_times_value} time units")
            print(f"   Objective Value: {obj_value} ({tardiness_value} * 10000 + {sum_of_end_times_value})")
            
            if tardiness_value == 0:
                print(f"   ✓ All time windows satisfied!")
            else:
                print(f"   ⚠ Time windows violated (tardiness > 0)")
        
        # Extract solution
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # [NEW] Start of modification (Change 2)
            # 1. Get the naive solution from the solver
            naive_solution = self._extract_solution(solver, scheduling_moves, move_starts, 
                                                     move_ends, move_vehicles, move_tardiness, 
                                                     total_tardiness, status)
            
            if self.verbose:
                print(f"  CP-SAT solution found (Status: {naive_solution.get('solver_status')}). "
                      f"Running collision repair logic...")
            
            # 2. Run the collision repair logic on the naive solution
            try:
                repairer = VRPCollisionRepairer(instance=self.instance, verbose=self.verbose)
                repaired_solution = repairer.repair_solution(naive_solution)
                repaired_solution['solver_status'] = 'repaired_solution'
                if self.verbose:
                    print("  Collision repair logic completed.")
                return repaired_solution
            except Exception as e:
                if self.verbose:
                    print(f"  ERROR during collision repair: {e}")
                    print(traceback.format_exc())
                    print("  Returning naive (un-repaired) solution.")
                naive_solution['solver_status'] = 'repair_failed'
                naive_solution['error_message'] = f'Repair logic failed: {e}'
                return naive_solution # Return the original solution as a fallback
            # [NEW] End of modification

        else:
            if self.verbose:
                print(f"❌ Scheduling solver failed with status: {solver.StatusName(status)}")
            return {
                'vehicles': [],
                'total_distance': 0,
                'total_time': 0,
                'solver_status': 'infeasible',
                'error': f'CP-SAT status: {solver.StatusName(status)}'
            }
    
    def _extract_solution(self, solver, scheduling_moves: List[SchedulingMove],
                         move_starts, move_ends, move_vehicles, move_tardiness, 
                         total_tardiness, status) -> Dict:
        """
        Extract the solution from the CP-SAT solver.
        Includes tardiness information for soft time window violations.
        """
        # Group moves by vehicle
        vehicle_schedules = {v: [] for v in range(self.num_vehicles)}
        
        # Track tardiness details
        tardiness_details = []
        total_tardiness_value = solver.Value(total_tardiness)
        
        for move in scheduling_moves:
            vehicle = solver.Value(move_vehicles[move.move_id])
            start_time = solver.Value(move_starts[move.move_id])
            end_time = solver.Value(move_ends[move.move_id])
            tardiness = solver.Value(move_tardiness[move.move_id])
            
            vehicle_schedules[vehicle].append({
                'move': move,
                'start_time': start_time,
                'end_time': end_time,
                'ul_id': move.ul_id,
                'move_type': move.move_type,
                'from_pos': move.from_location,
                'to_pos': move.to_location,
                'from_tier': move.from_tier,
                'to_tier': move.to_tier,
                'tardiness': tardiness
            })
            
            # Record tardiness details for moves with violations
            if tardiness > 0:
                violation_type = []
                
                # For STORE operations: the critical constraint is on START time (pickup window)
                # For RETRIEVE operations: the critical constraint is on END time (delivery window)
                # The reported window should reflect the actual constrained time point
                
                if move.move_type == 'store':
                    # Store: pickup from source must be within arrival window
                    # CRITICAL: latest_finish already represents the latest PICKUP time (arrival_end)
                    # Do NOT subtract service_time - that was the old bug!
                    # Check if start_time violates the arrival window
                    if start_time < move.earliest_start:
                        violation_type.append(f"pickup {move.earliest_start - start_time} units early")
                    if start_time > move.latest_finish:
                        violation_type.append(f"pickup {start_time - move.latest_finish} units late")
                    
                    # Report the window for the PICKUP (start) time
                    window_start = move.earliest_start
                    window_end = move.latest_finish  # Latest pickup time (arrival_end in CP-SAT time)
                    tardiness_details.append({
                        'move_id': move.move_id,
                        'ul_id': move.ul_id,
                        'move_type': move.move_type,
                        'tardiness': tardiness,
                        'violation': ', '.join(violation_type) if violation_type else f'time window deviation of {tardiness} units',
                        'window': [window_start, window_end],
                        'window_type': 'pickup',  # Indicates this is the pickup window
                        'actual': [start_time, end_time],
                        'actual_pickup': start_time
                    })
                    
                elif move.move_type in ['retrieve', 'direct_retrieve']:
                    # Retrieve: delivery to sink must be within retrieval window
                    # Check if end_time violates the retrieval window
                    if start_time < move.earliest_start:
                        violation_type.append(f"started {move.earliest_start - start_time} units early")
                    if end_time > move.latest_finish:
                        violation_type.append(f"finished {end_time - move.latest_finish} units late")
                    
                    # Report the window for the DELIVERY (end) time
                    tardiness_details.append({
                        'move_id': move.move_id,
                        'ul_id': move.ul_id,
                        'move_type': move.move_type,
                        'tardiness': tardiness,
                        'violation': ', '.join(violation_type) if violation_type else f'time window deviation of {tardiness} units',
                        'window': [move.earliest_start, move.latest_finish],
                        'window_type': 'delivery',  # Indicates this is the delivery window
                        'actual': [start_time, end_time]
                    })
                    
                else:
                    # For other move types (reshuffle, etc.)
                    if start_time < move.earliest_start:
                        violation_type.append(f"started {move.earliest_start - start_time} units early")
                    if end_time > move.latest_finish:
                        violation_type.append(f"finished {end_time - move.latest_finish} units late")
                    
                    tardiness_details.append({
                        'move_id': move.move_id,
                        'ul_id': move.ul_id,
                        'move_type': move.move_type,
                        'tardiness': tardiness,
                        'violation': ', '.join(violation_type) if violation_type else f'time window deviation of {tardiness} units',
                        'window': [move.earliest_start, move.latest_finish],
                        'actual': [start_time, end_time]
                    })
        
        # Sort each vehicle's schedule by start time
        for v in vehicle_schedules:
            vehicle_schedules[v].sort(key=lambda x: x['start_time'])
        
        # Build vehicle objects for output
        vehicles = []
        total_distance = 0
        makespan = 0
        
        for v in range(self.num_vehicles):
            schedule = vehicle_schedules[v]
            if not schedule:
                continue
            
            route_moves = []
            vehicle_distance = 0
            
            # Track current position to add empty repositioning moves
            current_location = 'sink'  # Vehicles start at sink
            current_tier = 1
            last_end_time_1based = 0
            
            if self.verbose and schedule:
                print(f"\n=== Raw CP-SAT Schedule for Vehicle {v+1} ===")
                for task in schedule[:5]:  # Show first 5 moves
                    print(f"  Move {task['move'].move_id}: UL{task['ul_id']} {task['move_type']} "
                          f"{task['from_pos']}->{task['to_pos']}, start={task['start_time']}, end={task['end_time']}")
            
            for task_idx, task in enumerate(schedule):
                move = task['move']
                # CP-SAT model uses 0-based times, but we need 1-based for Gurobi
                # Nothing can happen at t=0, so add 1 to convert
                start_time_1based = task['start_time'] + 1
                end_time_1based = task['end_time'] + 1
                
                # Add empty repositioning move if needed (for distance calculation)
                # The CP-SAT solver has already accounted for the TIME, we just need to add the move for distance
                if current_location != move.from_location or current_tier != move.from_tier:
                    empty_dist = self._calculate_distance(current_location, move.from_location, 
                                                          current_tier, move.from_tier)
                    if empty_dist > 0:
                        empty_travel_time = self._calculate_gurobi_style_travel_time(
                            current_location, move.from_location, 
                            current_tier, move.from_tier, 
                            handling_time_enabled=False
                        )
                        
                        # CRITICAL: In Gurobi model, empty moves take at least 1 timestep
                        # Even if travel_time=0 (same location), the move occupies 1 timestep
                        empty_duration = max(1, empty_travel_time)
                        
                        # CRITICAL FIX FOR SIMULTANEITY CONFLICTS:
                        # Empty move must complete BEFORE the loaded move starts
                        # In Gurobi: a move DEPARTS at time t, state updated at t+1
                        #           a move ARRIVES at time t, state updated at t
                        # So if empty departs at t=X and loaded arrives at t=X, CONFLICT!
                        # We need: empty_end < start_time_1based
                        # 
                        # In Gurobi, a move starting at time t with duration d:
                        # - Occupies times [t, t+1, ..., t+d-1] (d timesteps total)
                        # - "Ends" at time t+d (first time NOT occupying)
                        # 
                        # We want: empty_end < start_time_1based
                        # So: empty_start + empty_duration < start_time_1based
                        # Therefore: empty_start < start_time_1based - empty_duration
                        # 
                        # Working backward from the constraint empty_end < start_time_1based:
                        empty_end = start_time_1based - 1  # Must end strictly before next move
                        empty_start = empty_end - empty_duration + 1  # Move occupies [start, start+1, ..., end]
                        
                        # CRITICAL: Nothing can happen at t=0, earliest possible time is t=1
                        if empty_start < 1:
                            if task_idx == 0 and self.verbose:
                                print(f"  ⚠️  empty_start={empty_start} < 1, clamping to 1")
                            empty_start = 1
                            # Recalculate end: end = start + duration - 1
                            empty_end = empty_start + empty_duration - 1
                        
                        # Ensure empty move doesn't start before previous move ends
                        # In Gurobi: previous move ends at last_end means it DEPARTS at last_end
                        # Next move can start at last_end+1 (the earliest it can DEPART)
                        # UPDATE: With relaxed constraints, we allow start == last_end if reposition_time == 0
                        # But here we are in the "empty_dist > 0" block, so reposition_time >= 1.
                        # So start >= last_end + 1 is correct for non-zero moves.
                        
                        if empty_start <= last_end_time_1based:
                            if self.verbose:
                                print(f"  ⚠️  WARNING: Empty move start ({empty_start}) <= last_end ({last_end_time_1based})")
                            # Start after previous ends
                            empty_start = last_end_time_1based # + 1 REMOVED: Allow abutment if CP-SAT allowed it
                            
                            # If CP-SAT allowed it, it means the constraints were satisfied.
                            # The constraints say: start_next >= end_prev + travel_time
                            # Here: start_loaded >= end_prev + empty_travel_time
                            # 
                            # If we force empty_start = last_end_time_1based + 1, we are adding an extra gap.
                            # Let's trust CP-SAT's timing.
                            # 
                            # The empty move fills the gap between last_end and next_start.
                            # empty_start = next_start - empty_duration
                            # 
                            # If empty_start < last_end, THEN we have a problem.
                            # But if empty_start == last_end, it might be valid if the previous move
                            # "ended" at t, meaning it occupied up to t-1.
                            # 
                            # Wait, Gurobi semantics:
                            # Move starts at t, duration d. Occupies [t, t+d-1].
                            # Next move can start at t+d.
                            # So if prev ends at 47 (occupies up to 46), next can start at 47.
                            # 
                            # So empty_start = last_end_time_1based is VALID.
                            # It means 0 gap.
                            
                            empty_start = max(empty_start, last_end_time_1based)
                            empty_end = empty_start + empty_duration # - 1 REMOVED: Gurobi end is start + duration
                            
                            # Now check if this pushes the empty move's end to conflict with the next loaded move
                            if empty_end > start_time_1based: # Changed >= to > because end is exclusive in some contexts? No, Gurobi end is start of next.
                                # If empty_end (arrival) > start_time (departure of next), conflict.
                                # If empty_end == start_time, it's perfect abutment.
                                
                                if self.verbose:
                                    print(f"  ⚠️  CRITICAL CP-SAT CONSTRAINT VIOLATION:")
                                    print(f"      Previous loaded move ends at t={last_end_time_1based}")
                                    print(f"      Empty reposition takes {empty_travel_time} time units")
                                    print(f"      Earliest next move can start: t={last_end_time_1based + empty_travel_time}")
                                    print(f"      But CP-SAT scheduled it at: t={start_time_1based}")
                                    print(f"      This indicates the location continuity constraints are not working correctly!")
                                
                                # For now, we have no choice but to push the loaded move forward
                                # This will make the solution invalid, but at least avoid the simultaneity conflict
                                empty_end = start_time_1based # - 1 REMOVED
                                # Recalculate start: end = start + duration, so start = end - duration
                                empty_start = max(last_end_time_1based, empty_end - empty_duration)
                            
                            # Check if this creates a conflict with the next move
                            if empty_end > start_time_1based:
                                if self.verbose:
                                    print(f"  ⚠️  CRITICAL: Adjusted empty move (end={empty_end}) conflicts with next move (start={start_time_1based})")
                                # This indicates a bug in the CP-SAT constraints!
                                # For now, just report it and let validation catch it
                        
                        if task_idx == 0 and self.verbose:
                            print(f"  Final: empty_start={empty_start}, empty_end={empty_end}")
                            print(f"  Recording empty move from t={empty_start} to t={empty_end}\n")
                        
                        route_moves.append({
                            'ul_id': 0,
                            'move_type': 'empty',
                            'from_location': current_location,
                            'to_location': move.from_location,
                            'from_pos': current_location,
                            'to_pos': move.from_location,
                            'from_tier': current_tier,
                            'to_tier': move.from_tier,
                            'start_time': empty_start,
                            'end_time': empty_start + empty_duration,  # Gurobi semantics: arrival time = start + duration
                            'distance': empty_dist,
                            'travel_distance': empty_dist,
                            'service_time': empty_duration,  # Use clamped duration (min 1)
                            'empty_travel_distance': empty_dist
                        })
                        vehicle_distance += empty_dist
                        
                        # CRITICAL: Update last_end_time after empty move for next iteration
                        last_end_time_1based = empty_start + empty_duration
                        
                        # CRITICAL: Update current position after empty repositioning
                        current_location = move.from_location
                        current_tier = move.from_tier
                
                # Add the actual loaded move
                route_moves.append({
                    'ul_id': move.ul_id,
                    'move_type': move.move_type,
                    'from_location': move.from_location,
                    'to_location': move.to_location,
                    'from_pos': move.from_location,
                    'to_pos': move.to_location,
                    'from_tier': move.from_tier,
                    'to_tier': move.to_tier,
                    'start_time': start_time_1based,
                    'end_time': end_time_1based,
                    'distance': move.travel_distance,
                    'travel_distance': move.travel_distance,
                    'service_time': move.service_time,
                    'empty_travel_distance': 0
                })
                vehicle_distance += move.travel_distance
                makespan = max(makespan, task['end_time'])
                
                # Update position for next iteration
                current_location = move.to_location
                current_tier = move.to_tier
                last_end_time_1based = end_time_1based
            
            # Note: Vehicles do NOT need to return to sink at the end
            
            # Convert to 1-based vehicle ID for Gurobi compatibility
            vehicle_id_1based = v + 1
            
            vehicles.append({
                'vehicle_id': vehicle_id_1based,           # Use 1-based vehicle ID
                'moves': route_moves,
                'total_distance': vehicle_distance,
                'total_time': schedule[-1]['end_time'] + 1 if schedule else 0,  # 1-based time
                'completion_time': schedule[-1]['end_time'] + 1 if schedule else 0  # 1-based time
            })
            total_distance += vehicle_distance
        
        # Calculate final makespan (convert to 1-based)
        makespan = max((schedule[-1]['end_time'] + 1 for schedule in vehicle_schedules.values() if schedule), default=0)
        
        # Time windows are now SOFT constraints with penalties in the objective
        # If tardiness > 0, some time windows were violated
        solver_status = 'optimal' if status == cp_model.OPTIMAL else 'feasible'
        
        if self.verbose:
            print(f"\n✓ Solution found: {solver_status}")
            print(f"  Total Tardiness: {total_tardiness_value} time units")
            print(f"  Makespan: {makespan}")
            print(f"  Total distance: {total_distance}")
            print(f"  Active vehicles: {len([v for v in vehicles if v['moves']])}")
            
            if total_tardiness_value == 0:
                print(f"  ✓ All time windows satisfied!")
            else:
                print(f"  ⚠ {len(tardiness_details)} move(s) violated time windows:")
                for detail in tardiness_details[:5]:  # Show first 5
                    print(f"    - Move {detail['move_id']} (UL{detail['ul_id']} {detail['move_type']}): "
                          f"{detail['violation']}, tardiness={detail['tardiness']}")
        
        result = {
            'vehicles': vehicles,
            'total_distance': total_distance,
            'total_time': makespan,
            'solver_status': solver_status,
            'tardiness': {
                'total': total_tardiness_value,
                'violations': tardiness_details,
                'num_violations': len(tardiness_details)
            }
        }
        
        return result
    
    def _get_precedence_rules(self, scheduling_moves: List[SchedulingMove]) -> List[Tuple[int, int]]:
        """
        Generate precedence constraints based on A* move sequence.
        Sequential moves for the same unit load must maintain order.
        """
        precedence_rules = []
        
        # Group moves by unit load
        ul_moves = {}
        for move in scheduling_moves:
            if move.ul_id not in ul_moves:
                ul_moves[move.ul_id] = []
            ul_moves[move.ul_id].append(move.move_id)
        
        # Add precedence between sequential moves of same unit load
        for ul_id, move_ids in ul_moves.items():
            for i in range(len(move_ids) - 1):
                precedence_rules.append((move_ids[i], move_ids[i + 1]))
        
        return precedence_rules
    
    def _get_lifo_constraints(self, scheduling_moves: List[SchedulingMove]) -> List[Tuple[int, int]]:
        """
        Generate LIFO constraints for storage locations.
        
        If unit load A is stored at (lane, tier_upper) and unit load B is stored at (lane, tier_lower)
        where tier_upper > tier_lower (A is on top of B), then A must be retrieved before B.
        
        Returns:
            List of (upper_retrieve_move_id, lower_retrieve_move_id) tuples
        """
        lifo_rules = []
        
        # Track storage and retrieval moves by (lane, tier)
        storage_moves = {}  # (lane, tier) -> list of (ul_id, store_move_id, retrieve_move_id)
        
        for move in scheduling_moves:
            if move.move_type in ['store', 'reshuffle']:
                # Track where this UL is stored
                lane = self._parse_location_str(move.to_location)
                if lane not in ['source', 'sink']:
                    key = (lane, move.to_tier)
                    if key not in storage_moves:
                        storage_moves[key] = []
                    
                    # Find the corresponding retrieval move for this unit load
                    retrieve_move = None
                    for other_move in scheduling_moves:
                        if (other_move.ul_id == move.ul_id and 
                            other_move.move_type in ['retrieve', 'reshuffle', 'direct_retrieve'] and
                            self._parse_location_str(other_move.from_location) == lane and
                            other_move.from_tier == move.to_tier):
                            retrieve_move = other_move
                            break
                    
                    # Always track storage moves, even if not retrieved
                    storage_moves[key].append({
                        'ul_id': move.ul_id,
                        'store_move_id': move.move_id,
                        'retrieve_move_id': retrieve_move.move_id if retrieve_move else None
                    })
        
        # For each lane, enforce LIFO between tiers
        # Group by lane
        lanes = {}
        for (lane, tier), moves in storage_moves.items():
            if lane not in lanes:
                lanes[lane] = {}
            lanes[lane][tier] = moves
        
        # For each lane, check all tier pairs
        for lane, tiers_dict in lanes.items():
            tier_list = sorted(tiers_dict.keys())
            
            # For each pair of tiers where upper > lower
            # tier_list is sorted, so lower indices come first.
            # But wait, tier numbering: 1 is Back, 2 is Front.
            # So 1 < 2.
            # We want Store(1) before Store(2).
            # And Retrieve(2) before Retrieve(1).
            
            for i, upper_tier in enumerate(tier_list):
                for lower_tier in tier_list[:i]:  # lower_tier < upper_tier
                    # All ULs in upper_tier (Front) must be retrieved before any UL in lower_tier (Back)
                    # All ULs in lower_tier (Back) must be stored before any UL in upper_tier (Front)
                    
                    upper_moves = tiers_dict[upper_tier]
                    lower_moves = tiers_dict[lower_tier]
                    
                    for upper_entry in upper_moves:
                        for lower_entry in lower_moves:
                            # Upper tier retrieval must complete before lower tier retrieval starts
                            if upper_entry['retrieve_move_id'] and lower_entry['retrieve_move_id']:
                                lifo_rules.append((
                                    upper_entry['retrieve_move_id'],
                                    lower_entry['retrieve_move_id']
                                ))
                            
                            # Lower tier storage must complete before upper tier storage starts
                            lifo_rules.append((
                                lower_entry['store_move_id'],
                                upper_entry['store_move_id']
                            ))
        
        return lifo_rules
    
    def _get_lane_sequencing_rules(self, scheduling_moves: List[SchedulingMove]) -> List[Tuple[int, int]]:
        """
        Generate lane sequencing constraints to ensure moves follow A* order
        for the same lane-tier combination.
        
        Rationale:
        - A* determines a specific order for accessing each storage location
        - We enforce this order to respect the plan's intent
        - LIFO constraints (precedence) already prevent wrong retrieval order
        - Cooldown constraints already prevent physical collisions
        - So we only need light sequencing to maintain A*'s intended ordering
        
        We track moves by (lane, tier) to enforce ordering at the storage location level.
        """
        lane_sequencing_rules = []
        
        # Group moves by (lane, tier) combination
        lane_tier_moves = {}  # (lane_id, tier) -> list of move_ids
        
        for move in scheduling_moves:
            move_type = move.move_type
            
            # For stores: track destination (to_lane, to_tier)
            if move_type == 'store':
                to_lane = self._parse_location_str(move.to_location)
                if to_lane not in ['source', 'sink']:
                    key = (to_lane, move.to_tier)
                    if key not in lane_tier_moves:
                        lane_tier_moves[key] = []
                    lane_tier_moves[key].append(move.move_id)
            
            # For retrieves: track source (from_lane, from_tier)
            elif move_type == 'retrieve':
                from_lane = self._parse_location_str(move.from_location)
                if from_lane not in ['source', 'sink']:
                    key = (from_lane, move.from_tier)
                    if key not in lane_tier_moves:
                        lane_tier_moves[key] = []
                    lane_tier_moves[key].append(move.move_id)
        
        # For each lane-tier combination, enforce A* ordering
        for (lane_id, tier), move_ids in lane_tier_moves.items():
            # Remove duplicates while preserving A* order
            seen = set()
            ordered_move_ids = []
            for mid in move_ids:
                if mid not in seen:
                    seen.add(mid)
                    ordered_move_ids.append(mid)
            
            # Add precedence constraints for consecutive moves at same location
            # Note: This is relaxed compared to forcing end-before-start
            # The cooldown constraints handle physical separation
            for i in range(len(ordered_move_ids) - 1):
                lane_sequencing_rules.append((ordered_move_ids[i], ordered_move_ids[i + 1]))
        
        return lane_sequencing_rules
    
    # ============= Helper Methods (same as VRP solver) =============
    
    def _parse_location_str(self, loc_str: any) -> str:
        """Parse a location string to extract the AP id or keyword."""
        s = str(loc_str)
        
        if s in self._location_parse_cache:
            return self._location_parse_cache[s]
        
        if s.lower() in ['source', 'sink']:
            result = s.lower()
        else:
            # Correct: Just finds the number
            ap_id = re.search(r'\d+', s)
            result = ap_id.group(0) if ap_id else s
            
            # Normalize: location '0' is the source
            if result == '0' or result == str(self.source_lane_id):
                result = 'source'
            # Normalize: location for sink
            elif result == str(self.sink_lane_id):
                result = 'sink'
        
        self._location_parse_cache[s] = result
        return result
    
    def _parse_location_and_tier_str(self, loc_str: any) -> tuple[str, int]:
        """Parse a location string to extract the AP id and tier."""
        s = str(loc_str)
        
        if s in self._location_tier_parse_cache:
            return self._location_tier_parse_cache[s]
        
        if s.lower() in ['source', 'sink']:
            return (s.lower(), 1)
        
        numbers = re.findall(r'\d+', s)
        if len(numbers) >= 2:
            result = (numbers[0], int(numbers[1]))
        elif len(numbers) == 1:
            result = (numbers[0], 1)
        else:
            result = (s, 1)
        
        self._location_tier_parse_cache[s] = result
        return result
    
    @lru_cache(maxsize=None)
    def _get_in_lane_travel_time(self, loc: str, tier: int) -> int:
        """
        Calculates the one-way travel time from the lane's AP to the specified tier.
        """
        if loc in {self.source_lane_id, self.sink_lane_id, 'source', 'sink', None}:
            return 0
            
        n_slots = self.lane_n_slots.get(loc)
        if not n_slots:
            return 0  # Fallback

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
        
        return max(1, total_time)  # Ensure at least 1 time unit
    
    def _calculate_distance(self, from_loc: str, to_loc: str, 
                           from_tier: int = 1, to_tier: int = 1) -> float:
        """Calculate distance between two locations."""
        def get_lane_or_str(loc: any):
            if isinstance(loc, str):
                # Parse location first (this normalizes '0' -> 'source', etc.)
                parsed = self._parse_location_str(loc)
                
                # Check if it's source or sink
                if parsed.lower() in ['source', 'sink']:
                    return parsed.lower()
                
                # Otherwise try to convert to access point ID
                try:
                    ap_id = int(parsed)
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

        if lane1_obj == lane2_obj:
            # If same lane and same tier (or both are strings like 'source'), distance is 0
            if isinstance(lane1_obj, str):
                return 0.0
            
            # For Lane objects, check if tiers are same
            t1_id = tier1_obj.get_id() if hasattr(tier1_obj, 'get_id') else tier1_obj
            t2_id = tier2_obj.get_id() if hasattr(tier2_obj, 'get_id') else tier2_obj
            
            if t1_id == t2_id:
                return 0.0

        return self.instance.calculate_distance(lane1_obj, tier1_obj, lane2_obj, tier2_obj)
    
    def _calculate_gurobi_style_travel_time(self, from_loc: str, to_loc: str, 
                                           from_tier: int, to_tier: int, 
                                           handling_time_enabled: bool) -> int:
        """
        OPTIMIZED: Cached travel time calculation.
        """
        cache_key = (from_loc, to_loc, from_tier, to_tier, handling_time_enabled)
        if cache_key in self._travel_time_cache:
            return self._travel_time_cache[cache_key]
        
        distance = self._calculate_distance(from_loc, to_loc, from_tier, to_tier)
        travel_time = distance / self.instance.vehicle_speed
        
        if handling_time_enabled: 
            # Correct: Adds 2x handling time (load + unload) and applies ceil() only to travel time.
            result = max(1, ceil(travel_time) + 2*self.instance.get_handling_time())
        else:
            # Allow 0 time for 0 distance (same location) to enable immediate task sequencing
            # Also check if distance is very close to 0 (float precision)
            if distance < 1e-6:
                result = 0
            else:
                result = max(1, ceil(travel_time))
        
        self._travel_time_cache[cache_key] = result
        return result


def solve_twvrp_with_scheduling(buffer, moves: List[Dict], num_vehicles: int = 1, 
                                instance=None, time_limit: Optional[int] = None, 
                                verbose: bool = False) -> Dict:
    """
    Module-level helper to solve the time window VRP using CP-SAT scheduling.
    
    Args:
        buffer: Buffer object for distance calculations
        moves: List of moves from A* search
        num_vehicles: Number of available vehicles
        instance: Instance object for handling time access
        time_limit: Maximum time in seconds for solving
        verbose: Enable verbose output
    
    Returns:
        Dictionary containing the scheduling solution with hard time window constraints
    """
    solver = TWVRPSchedulingSolver(
        buffer, num_vehicles, vehicle_capacity=1, 
        instance=instance, verbose=verbose
    )
    return solver.solve_twvrp(moves, time_limit)