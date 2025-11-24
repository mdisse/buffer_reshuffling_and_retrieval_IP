"""
This module implements a post-processing repair logic for VRP schedules
to resolve lane collisions based on a 3-priority strategy.

As requested:
1.  **Priority 1 (Reschedule):** If a 'park' blocks an intruder and the
    parked vehicle's next move is 'empty', reschedule the empty move
    to happen earlier, if safe. This logic respects all future
    hard time-window constraints.
2.  **Priority 2 (Eviction):** If a 'park' blocks and the next move is
    'loaded', OR if Priority 1 fails, attempt to "evict" the
    parked vehicle.
    - **Smart Eviction:** If the next move is 'empty' to Source/Sink,
      execute that move now.
    - **Standard Eviction:** Otherwise, move to a free lane
      (preferring the Sink) and back.
3.  **Priority 3 (Fallback/Delay):** If the blocker is a 'move' (not a
    'park') or if P1 and P2 fail, delay the *intruding* vehicle and
    all its subsequent moves.
"""

import sys
import os
import copy
import math
from typing import List, Dict, Optional, Any

# Add project root to path
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

# We can import these types for clarity, but the code relies on
# the instance being passed in, which has the concrete classes.
from src.instance.instance import Instance
from src.bay.buffer import Buffer
from src.bay.virtual_lane import VirtualLane


class VRPCollisionRepairer:
    """
    Implements the 3-priority repair logic to fix lane collisions
    in a "naive" VRP solution.
    """

    def __init__(self, instance: Instance, verbose: bool = False):
        self.instance = instance
        self.verbose = verbose
        self.buffer: Buffer = self.instance.get_buffer()
        # Handle potential attribute errors if sink/source don't exist
        self.source_ap_id = self.buffer.get_source().get_ap_id() if self.buffer.has_sources() else -1
        self.sink_ap_id = self.buffer.get_sink().get_ap_id() if self.buffer.has_sinks() else -2
        # Cache lane objects for quick lookup
        self.lane_map = {lane.get_ap_id(): lane for lane in self.buffer.get_virtual_lanes()}

    def repair_solution(self, naive_solution: Dict) -> Dict:
        """
        Main entry point for the repair logic.
        Operates as a while loop, finding and fixing one collision at a time.
        """
        repaired_solution = copy.deepcopy(naive_solution)
        loops = 0
        max_loops = 2000  # Safety break

        if self.verbose:
            print(f"  Repair logic starting...")

        while loops < max_loops:
            loops += 1
            collision_found = False

            # 1. Build the full timeline of *all* blocking events
            timeline = self._build_full_lane_timeline(repaired_solution)

            # 2. Sort by lane, then by start time
            timeline.sort(key=lambda x: (x['lane_id'], x['start']))

            # 3. Scan for the first collision
            # Check ALL pairs of events in the same lane, not just consecutive ones
            for i in range(len(timeline)):
                event_A = timeline[i]
                for j in range(i + 1, len(timeline)):
                    event_B = timeline[j]
                    
                    # Skip if different lanes
                    if event_A['lane_id'] != event_B['lane_id']:
                        break  # Timeline is sorted by lane, so no more matches
                    
                    # Skip if same vehicle
                    if event_A['v_idx'] == event_B['v_idx']:
                        continue
                    
                    # Check for temporal overlap
                    if event_A['end'] > event_B['start']:
                        collision_found = True
                        # 4. Check for Vehicle Swap pattern FIRST (new optimization)
                        swap_executed = self._attempt_vehicle_swap(repaired_solution, event_A, event_B, timeline)
                        
                        if not swap_executed:
                            # 5. Resolve the collision using standard logic
                            self._resolve_collision(repaired_solution, event_A, event_B, timeline)

                        # 6. Break from both loops to rebuild the timeline
                        break
                
                if collision_found:
                    break

            if not collision_found:
                if self.verbose:
                    print(f"  Repair logic finished: No collisions found after {loops} repair loops.")
                break  # Exit while loop

        if loops == max_loops:
            if self.verbose:
                print(f"  Repair logic FAILED: Max loops ({max_loops}) reached. Returning potentially broken solution.")

        # 7. Enforce dependencies (Store < Retrieve)
        self._enforce_dependencies(repaired_solution)

        # Recalculate makespan and return
        return self._recalculate_solution_metrics(repaired_solution)

    def _attempt_vehicle_swap(self, solution: Dict, event_A: Dict, event_B: Dict, timeline: List[Dict]) -> bool:
        """
        Detects if two vehicles are essentially swapping positions and would benefit from
        swapping their remaining schedules. This avoids wasteful repositioning moves.
        
        Pattern to detect:
        - Vehicle A is at/moving to location X
        - Vehicle B is at/moving to location Y
        - Vehicle A's next work requires being at location Y
        - Vehicle B's next work requires being at location X
        
        Returns True if swap was executed, False otherwise.
        """
        v_idx_A = event_A['v_idx']
        v_idx_B = event_B['v_idx']
        
        vehicle_A = solution['vehicles'][v_idx_A]
        vehicle_B = solution['vehicles'][v_idx_B]
        
        moves_A = vehicle_A['moves']
        moves_B = vehicle_B['moves']
        
        # Find where each vehicle will be after their current collision point
        # For event_A (the blocker), use the move after the current park/move
        if event_A['event_type'] == 'park':
            next_move_idx_A = event_A['move_idx'] + 1
        else:
            # For a 'move' event, the vehicle is currently executing this move
            next_move_idx_A = event_A['move_idx'] + 1
        
        # For event_B (the intruder)
        if event_B['event_type'] == 'park':
            next_move_idx_B = event_B['move_idx'] + 1
        else:
            # For a 'move' event, this is the current move being executed
            next_move_idx_B = event_B['move_idx'] + 1
        
        # Safety checks
        if next_move_idx_A >= len(moves_A) or next_move_idx_B >= len(moves_B):
            return False  # One vehicle has no more moves
        
        # Get the locations each vehicle is AT or HEADING TO
        # For move events, the vehicle is heading to the destination
        if event_A['event_type'] == 'move':
            location_A_current = str(moves_A[event_A['move_idx']]['to_location'])
        else:
            location_A_current = str(moves_A[event_A['move_idx']]['to_location']) if event_A['move_idx'] < len(moves_A) else None
        
        if event_B['event_type'] == 'move':
            location_B_current = str(moves_B[event_B['move_idx']]['to_location'])
        else:
            location_B_current = str(moves_B[event_B['move_idx']]['to_location']) if event_B['move_idx'] < len(moves_B) else None
        
        # Get where each vehicle's next LOADED move starts from
        next_loaded_idx_A = self._find_next_loaded_move(moves_A, next_move_idx_A)
        next_loaded_idx_B = self._find_next_loaded_move(moves_B, next_move_idx_B)
        
        if next_loaded_idx_A is None or next_loaded_idx_B is None:
            return False  # One vehicle has no more loaded work
        
        next_loaded_A = moves_A[next_loaded_idx_A]
        next_loaded_B = moves_B[next_loaded_idx_B]
        
        location_A_needs = str(next_loaded_A['from_location'])
        location_B_needs = str(next_loaded_B['from_location'])
        
        # If both need the same location, swapping won't help congestion
        if location_A_needs == location_B_needs:
            return False
        
        # Check for the swap pattern:
        # A is at/heading to X, needs to be at Y
        # B is at/heading to Y, needs to be at X
        # This means they're crossing paths wastefully
        
        is_swap_pattern = (
            location_A_current == location_B_needs and 
            location_B_current == location_A_needs
        )
        
        if not is_swap_pattern:
            # Try alternate pattern: check if their next loaded moves are in opposite locations
            # and they're currently trying to reposition there
            next_move_A = moves_A[next_move_idx_A] if next_move_idx_A < len(moves_A) else None
            next_move_B = moves_B[next_move_idx_B] if next_move_idx_B < len(moves_B) else None
            
            if next_move_A and next_move_B:
                # Are they both doing empty repositioning moves?
                if (next_move_A.get('move_type') in ('empty', 'e') and 
                    next_move_B.get('move_type') in ('empty', 'e')):
                    
                    dest_A = str(next_move_A['to_location'])
                    dest_B = str(next_move_B['to_location'])
                    
                    # Check if A is going where B needs to be, and vice versa
                    is_swap_pattern = (
                        dest_A == location_B_needs and 
                        dest_B == location_A_needs
                    )
            
            # Also check if they're currently executing moves to each other's positions
            if not is_swap_pattern and event_A['event_type'] == 'move' and event_B['event_type'] == 'move':
                current_move_A = moves_A[event_A['move_idx']]
                current_move_B = moves_B[event_B['move_idx']]
                
                # Check if they're moving to where the other needs to be
                dest_A = str(current_move_A['to_location'])
                dest_B = str(current_move_B['to_location'])
                
                is_swap_pattern = (
                    dest_A == location_B_needs and 
                    dest_B == location_A_needs
                )
        
        if not is_swap_pattern:
            return False  # Not a swap pattern
        
        if self.verbose:
            print(f"  Repair (VEHICLE SWAP): Detected swap pattern between V{vehicle_A['vehicle_id']} "
                  f"and V{vehicle_B['vehicle_id']}!")
            print(f"    V{vehicle_A['vehicle_id']} at {location_A_current}, needs {location_A_needs}")
            print(f"    V{vehicle_B['vehicle_id']} at {location_B_current}, needs {location_B_needs}")
            print(f"    Swapping remaining schedules from idx {next_loaded_idx_A} and {next_loaded_idx_B}")
        
        # Execute the swap: exchange all moves from the next loaded move onwards
        remaining_A = moves_A[next_loaded_idx_A:]
        remaining_B = moves_B[next_loaded_idx_B:]
        
        # Truncate the current schedules
        moves_A[next_loaded_idx_A:] = []
        moves_B[next_loaded_idx_B:] = []
        
        # Swap the remaining schedules
        moves_A.extend(remaining_B)
        moves_B.extend(remaining_A)
        
        # Update vehicle IDs in the swapped moves (if stored)
        for move in moves_A[next_loaded_idx_A:]:
            if 'vehicle_id' in move:
                move['vehicle_id'] = vehicle_A['vehicle_id']
        
        for move in moves_B[next_loaded_idx_B:]:
            if 'vehicle_id' in move:
                move['vehicle_id'] = vehicle_B['vehicle_id']
        
        if self.verbose:
            print(f"  Repair (VEHICLE SWAP): Swap completed successfully!")
        
        return True
    
    def _find_next_loaded_move(self, moves: List[Dict], start_idx: int) -> Optional[int]:
        """Helper to find the index of the next loaded (non-empty) move."""
        for i in range(start_idx, len(moves)):
            if moves[i].get('move_type') not in ('empty', 'e'):
                return i
        return None

    def _build_full_lane_timeline(self, solution: Dict) -> List[Dict]:
        """
        Builds a complete timeline of all lane-blocking events (moves and parks).
        """
        timeline = []
        for v_idx, vehicle in enumerate(solution.get('vehicles', [])):
            moves = vehicle.get('moves', [])
            for m_idx, move in enumerate(moves):
                
                # --- 1. Create 'move' events ---
                
                # A 'move' blocks its *destination* lane
                lane_ap_id_to = self._get_lane_id_for_repair(move.get('to_location'))
                if lane_ap_id_to is not None:
                    duration = self._get_lane_blocking_duration(move, 'to')
                    end_time = move['end_time']
                    start_time = end_time - duration

                    timeline.append({
                        'event_type': 'move',
                        'lane_id': lane_ap_id_to,
                        'start': start_time,
                        'end': end_time,
                        'v_idx': v_idx,
                        'move_idx': m_idx,
                        'vehicle_id': vehicle['vehicle_id'],
                        'move_type': move.get('move_type', 'unknown')
                    })
                
                # A loaded move also blocks its *source* lane
                move_type = move.get('move_type')
                if move_type in ('retrieve', 'reshuffle', 'y', 'x', 'empty', 'e'):
                    lane_ap_id_from = self._get_lane_id_for_repair(move.get('from_location'))
                    if lane_ap_id_from is not None and lane_ap_id_from != lane_ap_id_to:
                        # Get blocking duration for the *source*
                        duration = self._get_lane_blocking_duration(move, 'from')
                        # A 'retrieve'/'reshuffle' blocks the source
                        # lane from the *start* of the move.
                        start_time = move['start_time']
                        end_time = start_time + duration
                        
                        timeline.append({
                            'event_type': 'move',
                            'lane_id': lane_ap_id_from,
                            'start': start_time,
                            'end': end_time,
                            'v_idx': v_idx,
                            'move_idx': m_idx,
                            'vehicle_id': vehicle['vehicle_id'],
                            'move_type': move.get('move_type', 'unknown')
                        })


                # --- 2. Create 'park' events ---
                # A 'park' event is the gap *before* the *next* move
                if m_idx + 1 < len(moves):
                    next_move = moves[m_idx + 1]
                    park_start = move['end_time']
                    park_end = next_move['start_time']

                    if park_end > park_start:
                        # The vehicle is parked at the 'to_location' of the *current* move
                        parked_lane_ap_id = self._get_lane_id_for_repair(move.get('to_location'))

                        if parked_lane_ap_id is not None:
                            timeline.append({
                                'event_type': 'park',
                                'lane_id': parked_lane_ap_id,
                                'start': park_start,
                                'end': park_end,
                                'v_idx': v_idx,
                                'move_idx': m_idx,  # This park event is *after* move_idx
                                'vehicle_id': vehicle['vehicle_id'],
                                'move_type': 'park'
                            })
                else:
                    # This is the LAST move for this vehicle.
                    # If it ends at a blocking lane, the vehicle stays parked there indefinitely.
                    # We need to create a "final park" event that extends far into the future.
                    parked_lane_ap_id = self._get_lane_id_for_repair(move.get('to_location'))
                    
                    if parked_lane_ap_id is not None:
                        park_start = move['end_time']
                        # Use a very large end time to represent "indefinitely parked"
                        # We can use 10000 or calculate the maximum time in the solution
                        max_time = max(
                            (m['end_time'] for v in solution.get('vehicles', []) for m in v.get('moves', [])),
                            default=park_start
                        )
                        park_end = max_time + 1000  # Extend well beyond the solution
                        
                        timeline.append({
                            'event_type': 'park',
                            'lane_id': parked_lane_ap_id,
                            'start': park_start,
                            'end': park_end,
                            'v_idx': v_idx,
                            'move_idx': m_idx,  # This park event is *after* move_idx
                            'vehicle_id': vehicle['vehicle_id'],
                            'move_type': 'park_final'  # Mark it as a final park for debugging
                        })
        return timeline

    def _get_lane_id_for_repair(self, location: Any) -> Optional[int]:
        """Helper to get a lane AP ID, skipping non-blocking locations."""
        try:
            lane_ap_id = int(location)
        except (ValueError, TypeError):
            return None  # Is 'source', 'sink', or other string

        if lane_ap_id == self.source_ap_id or lane_ap_id == self.sink_ap_id:
            return None  # Non-blocking locations

        return lane_ap_id

    def _find_virtual_lane_by_ap_id(self, ap_id: int) -> Optional[VirtualLane]:
        """Helper to find the VirtualLane object from the cache."""
        return self.lane_map.get(ap_id)

    def _get_lane_blocking_duration(self, move: Dict, loc_type: str = 'to') -> int:
        """
        Calculates the true tier-based blocking duration for a move.
        'loc_type' specifies whether we are blocking the 'to' or 'from' lane.
        """
        if loc_type == 'to':
            lane_ap_id = self._get_lane_id_for_repair(move.get('to_location'))
            tier = move.get('to_tier', 1)
        else: # 'from'
            lane_ap_id = self._get_lane_id_for_repair(move.get('from_location'))
            tier = move.get('from_tier', 1)

        if lane_ap_id is None:
            return 0  # Not a buffer lane

        lane_obj = self._find_virtual_lane_by_ap_id(lane_ap_id)
        if not lane_obj:
            if self.verbose:
                print(f"  Repair Warning: Could not find virtual lane object for AP {lane_ap_id}")
            return 1  # Safe default

        J_i = 1 # Safe default
        if hasattr(lane_obj, 'get_tiers') and lane_obj.get_tiers():
            J_i = len(lane_obj.get_tiers())
        elif hasattr(lane_obj, 'stacks') and lane_obj.stacks is not None:
            J_i = len(lane_obj.stacks)

        j = tier  # Slot index (1=deep, J_i=shallow)
        h = self.instance.get_handling_time()

        is_loaded = move.get('move_type') not in ('empty', 'e')
        h_component = h if is_loaded else 0
        
        # Extra buffer to match Gurobi's conservative constraints (vehicle_update adds 2h)
        safety_buffer = 2 * h if is_loaded else 0

        # Per Eq 4.70: D_j = 2*(J_i - j) + h
        duration_in_out = 2 * (J_i - j) + h_component + safety_buffer

        return int(math.ceil(max(1, duration_in_out)))

    def _resolve_collision(self, solution: Dict, event_A: Dict, event_B: Dict, timeline: List[Dict]):
        """
        Implements the 3-priority logic.
        P1 failure now escalates to P2.
        """
        # Special case: if BOTH are 'move' events, delay the one that can be delayed more easily
        if event_A['event_type'] == 'move' and event_B['event_type'] == 'move':
            if self.verbose:
                print(f"  Repair (P3: Move-Move Collision): Both V{event_A['vehicle_id']} and V{event_B['vehicle_id']} are MOVING. "
                      f"Delaying the intruder V{event_B['vehicle_id']}.")
            self._delay_intruder(solution, event_A, event_B)
            return
        
        if event_A['event_type'] == 'move':
            # --- Priority 3: Fallback (Unchanged) ---
            if self.verbose:
                print(f"  Repair (P3: Fallback): V{event_A['vehicle_id']} is MOVING (blocks {event_A['start']}-{event_A['end']}). "
                      f"Delaying intruder V{event_B['vehicle_id']}.")
            self._delay_intruder(solution, event_A, event_B)
            return

        elif event_A['event_type'] == 'park':
            # --- Event A is a PARK. Check its *next* move. ---
            vehicle_A = solution['vehicles'][event_A['v_idx']]
            next_move_idx = event_A['move_idx'] + 1

            if next_move_idx >= len(vehicle_A['moves']):
                # Parked and has no more moves.
                # Check if we can "steal" the retrieval task from the intruding vehicle
                if self._attempt_retrieval_steal(solution, event_A, event_B, timeline):
                    if self.verbose:
                        print(f"  Repair (P-Steal): Successfully stole retrieval task from V{event_B['vehicle_id']} to V{event_A['vehicle_id']}.")
                    return
                
                # Stealing failed, B must wait.
                if self.verbose:
                    print(f"  Repair (P3: Fallback): V{event_A['vehicle_id']} is PARKED (blocks {event_A['start']}-{event_A['end']}) and has no more moves. "
                          f"Delaying intruder V{event_B['vehicle_id']}.")
                self._delay_intruder(solution, event_A, event_B)
                return

            p1_success = False
            if vehicle_A['moves'][next_move_idx]['move_type'] in ('empty', 'e'):
                # --- Priority 1: Reschedule Empty Move ---
                if self.verbose:
                    print(f"  Repair (P1: Attempt): V{event_A['vehicle_id']}'s next move is empty. Attempting P1 Reschedule.")
                # Try P1 first. It will return True on success, False on failure.
                p1_success = self._reschedule_empty_move(solution, event_A, event_B, next_move_idx)

            if not p1_success:
                # --- Priority 2: Eviction ---
                # P1 failed or wasn't applicable (next move is loaded)
                if self.verbose:
                    print(f"  Repair (P2: Attempt): P1 failed or not applicable. Escalating to P2 (Eviction).")
                evicted = self._attempt_eviction(solution, event_A, event_B, timeline)
                
                if not evicted:
                    # P1 and P2 FAILED. Must use P3 (Fallback).
                    if self.verbose:
                        print(f"  Repair (P1 & P2 FAILED): No eviction possible. "
                              f"Delaying intruder V{event_B['vehicle_id']}.")
                    self._delay_intruder(solution, event_A, event_B)

    def _get_empty_move_block_info(self, vehicle_moves: List[Dict], start_move_idx: int) -> tuple[int, int, int]:
        """
        Finds the info for the contiguous block of empty moves starting at start_move_idx.
        This block ends at the next (non-empty) loaded move.

        Returns:
            tuple: (
                earliest_safe_start,  # Earliest time this block can start, respecting future loaded moves
                total_empty_duration, # Total duration of all empty moves in this block
                next_loaded_move_idx  # Index of the next loaded move (or len(moves))
            )
        """
        total_empty_duration = 0
        next_loaded_move_idx = len(vehicle_moves) # Default to end

        # Find the next loaded move and sum durations
        for i in range(start_move_idx, len(vehicle_moves)):
            move = vehicle_moves[i]
            if move['move_type'] not in ('empty', 'e'):
                next_loaded_move_idx = i
                break
            
            duration = move['end_time'] - move['start_time']
            total_empty_duration += max(1, duration) # Ensure at least 1 unit per move

        earliest_safe_start = 0
        if next_loaded_move_idx < len(vehicle_moves):
            # We have a future loaded move.
            # The *earliest* this block of empty moves can start is
            # (start_time_of_loaded_move) - (total_duration_of_empty_moves)
            # This respects the CP-SAT solver's (valid) schedule for the loaded move,
            # which in turn respects the hard arrival/retrieval constraints.
            scheduled_start_of_loaded_move = vehicle_moves[next_loaded_move_idx]['start_time']
            earliest_safe_start = scheduled_start_of_loaded_move - total_empty_duration
        
        # Also, it can't start before the previous move (the park) ended
        if start_move_idx > 0:
            # The previous move is at start_move_idx - 1
            earliest_safe_start = max(earliest_safe_start, vehicle_moves[start_move_idx - 1]['end_time'])
        else:
            earliest_safe_start = max(0, earliest_safe_start) # Can't start before t=0

        return earliest_safe_start, total_empty_duration, next_loaded_move_idx

    def _reschedule_empty_move(self, solution: Dict, event_A: Dict, event_B: Dict, next_move_A_idx: int) -> bool:
        """
        Implements Priority 1 logic: Try to move an empty move earlier,
        respecting all future hard constraints.
        Returns True on success, False on failure.
        """
        vehicle_A_moves = solution['vehicles'][event_A['v_idx']]['moves']

        # 1. Get info about the block of empty moves we need to reschedule
        (earliest_safe_start_allowed, 
         total_empty_duration, 
         next_loaded_move_idx) = self._get_empty_move_block_info(vehicle_A_moves, next_move_A_idx)

        # 2. Determine the new target start time.
        # It must be *at least* when the park started (event_A['start'])
        # AND *at least* as early as it's allowed to (earliest_safe_start_allowed).
        target_new_start = max(event_A['start'], earliest_safe_start_allowed)

        # 3. Calculate when this block would *finish*
        target_new_end = target_new_start + total_empty_duration

        # 4. Final Feasibility Check:
        # Does this new, safe schedule *actually* solve the collision?
        # It solves it *only* if it finishes before the intruder (B) arrives.
        if target_new_end <= event_B['start']:
            # FEASIBLE: Reschedule the *entire block* of empty moves
            old_start_time = vehicle_A_moves[next_move_A_idx]['start_time']
            delay = target_new_start - old_start_time  # This is a (likely) negative delay
            
            if self.verbose:
                print(f"  Repair (P1: Reschedule Empty): V-{event_A['vehicle_id']} in L{event_A['lane_id']} "
                      f"is moving empty block (idx {next_move_A_idx} to {next_loaded_move_idx-1}) "
                      f"from t={old_start_time} to t={target_new_start}.")

            # Propagate this shift to all empty moves in the block
            self._propagate_schedule_shift(solution, event_A['v_idx'],
                                         next_move_A_idx, next_loaded_move_idx, 
                                         delay, target_new_start)
            return True # Signal Success
        else:
            # INFEASIBLE: Even the earliest safe reschedule time is
            # not enough to avoid the collision.
            if self.verbose:
                print(f"  Repair (P1: Reschedule FAILED): V-{event_A['vehicle_id']} in L{event_A['lane_id']} "
                      f"CANNOT reschedule. Earliest safe end ({target_new_end}) is "
                      f"after intruder start ({event_B['start']}).")
            return False # Signal Failure

    def _attempt_eviction(self, solution: Dict, event_A: Dict, event_B: Dict, timeline: List[Dict]) -> bool:
        """
        Implements Priority 2 logic: Evict a stuck vehicle.
        Tries "Smart Eviction" first, then "Standard Eviction".
        """
        v_idx_A = event_A['v_idx']
        vehicle_A_moves = solution['vehicles'][v_idx_A]['moves']
        next_move_A_idx = event_A['move_idx'] + 1
        
        # Find the *first* empty move in the block (if any)
        first_empty_move = None
        first_empty_move_idx = -1
        next_loaded_move_idx = len(vehicle_A_moves) # Default to end
        
        for i in range(next_move_A_idx, len(vehicle_A_moves)):
            if vehicle_A_moves[i]['move_type'] not in ('empty', 'e'):
                next_loaded_move_idx = i
                break
            elif first_empty_move is None:
                first_empty_move = vehicle_A_moves[i]
                first_empty_move_idx = i
        
        # --- "Smart Eviction" Check ---
        # Instead of a wasteful 2-move eviction, try to move directly to
        # where the vehicle needs to be next.
        if first_empty_move is not None or next_loaded_move_idx < len(vehicle_A_moves):
            # Determine the best target location:
            # 1. If there's a loaded move, go to its 'from_location'
            # 2. Otherwise, if the next empty move goes to source/sink, go there
            # 3. Otherwise, try the empty move's destination
            
            target_ap_id = None
            target_reason = ""
            
            # Option 1: Go directly to where the next LOADED move starts
            if next_loaded_move_idx < len(vehicle_A_moves):
                next_loaded_move = vehicle_A_moves[next_loaded_move_idx]
                target_location_str = str(next_loaded_move['from_location'])
                
                # Check if this is a valid lane (not the current blocked lane)
                try:
                    potential_ap_id = int(target_location_str)
                    if potential_ap_id != event_A['lane_id']:  # Don't "evict" to the same lane
                        target_ap_id = potential_ap_id
                        target_reason = f"next loaded move starts from lane {target_ap_id}"
                except (ValueError, TypeError):
                    # It's source/sink - handle below
                    if target_location_str in ('source', str(self.source_ap_id)):
                        target_ap_id = self.source_ap_id
                        target_reason = "next loaded move starts from source"
                    elif target_location_str in ('sink', str(self.sink_ap_id)):
                        target_ap_id = self.sink_ap_id
                        target_reason = "next loaded move starts from sink"
            
            # Option 2: Fall back to first empty move's destination (if it's source/sink)
            if target_ap_id is None and first_empty_move is not None:
                to_loc_str = str(first_empty_move['to_location'])
                is_dest_source = (to_loc_str == str(self.source_ap_id) or to_loc_str == 'source')
                is_dest_sink = (to_loc_str == str(self.sink_ap_id) or to_loc_str == 'sink')
                
                if is_dest_source:
                    target_ap_id = self.source_ap_id
                    target_reason = "next empty move goes to source"
                elif is_dest_sink:
                    target_ap_id = self.sink_ap_id
                    target_reason = "next empty move goes to sink"
            
            if target_ap_id is not None:
                evict_to_lane_obj_str = "source" if target_ap_id == self.source_ap_id else ("sink" if target_ap_id == self.sink_ap_id else f"lane {target_ap_id}")
                
                if self.verbose:
                    print(f"  Repair (P2: Smart Eviction): {target_reason}. "
                          f"Attempting to go directly to {evict_to_lane_obj_str}.")
                
                # 1. Get info about the *entire* empty block
                (earliest_safe_start, total_empty_dur, 
                 next_loaded_idx) = self._get_empty_move_block_info(vehicle_A_moves, first_empty_move_idx)

                # 2. Calculate new move times for going to the target
                parked_lane_obj = self._find_virtual_lane_by_ap_id(event_A['lane_id'])
                evict_to_lane_obj = self._get_lane_obj_from_location(str(target_ap_id))
                evict_start = event_A['start'] # Start as soon as park began
                
                # Get current tier from the move that brought us here
                prev_move = vehicle_A_moves[event_A['move_idx']]
                current_tier = prev_move.get('to_tier', 1)
                
                # Determine the tier for the target location
                target_tier = 1  # Default
                if next_loaded_move_idx < len(vehicle_A_moves):
                    next_loaded_move = vehicle_A_moves[next_loaded_move_idx]
                    if str(next_loaded_move['from_location']) == str(target_ap_id):
                        target_tier = next_loaded_move.get('from_tier', 1)
                
                evict_move_dist = self.instance.calculate_distance(parked_lane_obj, current_tier, evict_to_lane_obj, target_tier)
                evict_move_time = int(math.ceil(max(1, evict_move_dist / self.instance.get_vehicle_speed())))
                evict_end = evict_start + evict_move_time

                # 3. Check if this *still* collides with B
                # We check evict_start, not evict_end, because we leave the source lane at evict_start.
                if evict_start > event_B['start']:
                    if self.verbose:
                        print(f"  Repair (P2: Smart Eviction FAILED): Evict start ({evict_start}) "
                              f"is after intruder start ({event_B['start']}).")
                    # This smart eviction won't work. Fall through to standard eviction.
                
                else:
                    # 3.5. NEW CHECK: Is the target lane free during our entire stay?
                    # We need to check if the target lane is available from when we arrive (evict_end)
                    # until when we need to leave for our next loaded move
                    if next_loaded_idx < len(vehicle_A_moves):
                        next_loaded_move = vehicle_A_moves[next_loaded_idx]
                        # Check if we're staying at the target lane or leaving
                        next_loaded_from_str = str(next_loaded_move['from_location'])
                        
                        if next_loaded_from_str == str(target_ap_id):
                            # We're staying at this lane until the loaded move starts
                            lane_occupation_end = next_loaded_move['start_time']
                        else:
                            # We need to leave before the loaded move to travel there
                            next_loaded_from_obj = self._get_lane_obj_from_location(next_loaded_from_str)
                            travel_from_wait_spot = self.instance.calculate_distance(
                                evict_to_lane_obj, target_tier,
                                next_loaded_from_obj, next_loaded_move.get('from_tier', 1)
                            )
                            travel_time_from_wait = int(math.ceil(max(1, travel_from_wait_spot / self.instance.get_vehicle_speed())))
                            lane_occupation_end = next_loaded_move['start_time'] - travel_time_from_wait
                    else:
                        # No more loaded moves, we'll stay indefinitely
                        lane_occupation_end = evict_end + 100000  # Very large number
                    
                    # Check if target lane is a buffer lane (not source/sink) and when it becomes free
                    target_lane_id = self._get_lane_id_for_repair(str(target_ap_id))
                    need_to_wait = False
                    wait_at_current_until = None
                    
                    if target_lane_id is not None:  # It's a buffer lane, not source/sink
                        # We need the lane to be free for the arrival move only, not the entire stay
                        # Find when the lane becomes free for just the eviction move itself
                        arrival_window_end = evict_end  # Original arrival time + move duration
                        
                        # Check if lane is free during our planned arrival
                        current_evict_end = event_A['end'] + evict_move_time
                        
                        if not self._is_lane_free(timeline, target_lane_id, event_A['end'], current_evict_end):
                            # Lane is occupied during our planned arrival - find when it becomes free
                            # We need it free for just the eviction move duration
                            lane_free_from = self._find_lane_free_from(timeline, target_lane_id, event_A['end'], current_evict_end)
                            
                            if lane_free_from > event_A['end']:
                                # Lane is occupied - we need to WAIT at current location until lane is free
                                wait_at_current_until = lane_free_from
                                adjusted_evict_start = lane_free_from
                                adjusted_evict_end = lane_free_from + evict_move_time
                                need_to_wait = True
                                
                                if self.verbose:
                                    print(f"  Repair (P2: Smart Eviction): Target lane {target_lane_id} occupied until {lane_free_from}. "
                                          f"Will WAIT at L{event_A['lane_id']} until t={lane_free_from}, then move to lane {target_lane_id} at t={adjusted_evict_start}-{adjusted_evict_end}.")
                                
                                # Update for later use
                                evict_start = adjusted_evict_start
                                evict_end = adjusted_evict_end
                    
                    # 4. Check if this new schedule is safe for future moves
                    # If we're going directly to where the loaded move starts,
                    # we can just wait there (travel time = 0).
                    # Otherwise, we need to calculate travel from the waiting spot.
                    
                    earliest_new_start_for_loaded = 0
                    is_safe_for_future = True

                    if next_loaded_idx < len(vehicle_A_moves):
                        next_loaded_move = vehicle_A_moves[next_loaded_idx]
                        next_loaded_from_str = str(next_loaded_move['from_location'])
                        
                        # Are we already at the right location?
                        if next_loaded_from_str == str(target_ap_id):
                            # Perfect! We're already where we need to be.
                            # Just wait until the intruder passes.
                            travel_time_from_wait = 0
                            wait_until = max(evict_end, event_B['end'])
                            earliest_new_start_for_loaded = wait_until
                        else:
                            # We need to travel from the waiting spot to the loaded move's start
                            next_loaded_from_obj = self._get_lane_obj_from_location(next_loaded_from_str)
                            
                            travel_from_wait_spot = self.instance.calculate_distance(
                                evict_to_lane_obj, target_tier,
                                next_loaded_from_obj, next_loaded_move.get('from_tier', 1)
                            )
                            travel_time_from_wait = int(math.ceil(max(1, travel_from_wait_spot / self.instance.get_vehicle_speed())))

                            # Earliest the loaded move can start = wait_end + travel
                            wait_until = max(evict_end, event_B['end'])
                            earliest_new_start_for_loaded = wait_until + travel_time_from_wait
                        
                        is_safe_for_future = (earliest_new_start_for_loaded <= next_loaded_move['start_time'])

                    if not is_safe_for_future:
                         if self.verbose:
                            print(f"  Repair (P2: Smart Eviction FAILED): New schedule (start {earliest_new_start_for_loaded}) "
                                  f"violates future loaded move (start {vehicle_A_moves[next_loaded_idx]['start_time']}).")
                         # This smart eviction is not safe. Fall through.
                    
                    elif evict_start > event_B['start']:
                        # Eviction conflicts with intruder (after possible delay adjustment)
                        if self.verbose:
                            print(f"  Repair (P2: Smart Eviction FAILED): Adjusted evict start ({evict_start}) "
                                  f"conflicts with intruder (starts {event_B['start']}).")
                        pass
                    
                    else:
                        # 5. Apply the Smart Eviction (all checks passed!)
                        if self.verbose:
                            if need_to_wait:
                                print(f"  Repair (P2: Smart Eviction): SUCCESS with WAIT. Waiting at L{event_A['lane_id']} until t={wait_at_current_until}, then going to {evict_to_lane_obj_str} at {evict_start}-{evict_end}.")
                            else:
                                print(f"  Repair (P2: Smart Eviction): SUCCESS. Going to {evict_to_lane_obj_str} at {evict_start}-{evict_end}.")
                        
                        # Determine where to insert/replace moves
                        # Get current tier from the move that brought us here
                        prev_move = vehicle_A_moves[event_A['move_idx']]
                        current_tier = prev_move.get('to_tier', 1)

                        if need_to_wait:
                            # Create PARK move at current location
                            park_move = {
                                'move_type': 'empty', 'ul_id': 0,
                                'from_location': str(event_A['lane_id']),
                                'to_location': str(event_A['lane_id']),
                                'from_tier': current_tier, 'to_tier': current_tier,
                                'start_time': event_A['end'],
                                'end_time': wait_at_current_until,
                                'travel_distance': 0,
                                'service_time': wait_at_current_until - event_A['end'],
                                'move_type_internal': 'smart_evict_wait'
                            }
                            
                            # Create the smart eviction move
                            smart_evict_move = {
                                'move_type': 'empty', 'ul_id': 0,
                                'from_location': str(event_A['lane_id']),
                                'to_location': str(target_ap_id),
                                'from_tier': current_tier, 'to_tier': target_tier,
                                'start_time': evict_start, 'end_time': evict_end,
                                'travel_distance': evict_move_dist, 'service_time': evict_move_time,
                                'move_type_internal': 'smart_evict'
                            }
                            
                            # Replace existing empty moves with PARK + smart eviction
                            if first_empty_move is not None:
                                vehicle_A_moves[first_empty_move_idx] = park_move
                                # Delete all other empty moves except the first
                                for i in range(next_loaded_idx - 1, first_empty_move_idx, -1):
                                    if self.verbose:
                                        print(f"  Repair (P2: Smart Eviction): Deleting redundant empty move at idx {i}")
                                    del vehicle_A_moves[i]
                                # Insert smart eviction after park
                                vehicle_A_moves.insert(first_empty_move_idx + 1, smart_evict_move)
                            else:
                                # No empty moves - insert both right after the current move
                                vehicle_A_moves.insert(next_move_A_idx, park_move)
                                vehicle_A_moves.insert(next_move_A_idx + 1, smart_evict_move)
                        else:
                            # Normal case - create just the smart eviction move
                            new_evict_move = {
                                'move_type': 'empty', 'ul_id': 0,
                                'from_location': str(event_A['lane_id']),
                                'to_location': str(target_ap_id),
                                'from_tier': current_tier, 'to_tier': target_tier,
                                'start_time': evict_start, 'end_time': evict_end,
                                'travel_distance': evict_move_dist, 'service_time': evict_move_time,
                                'move_type_internal': 'smart_evict'
                            }
                            
                            # Replace or insert the new eviction move
                            if first_empty_move is not None:
                                vehicle_A_moves[first_empty_move_idx] = new_evict_move
                                # Delete all other empty moves
                                for i in range(next_loaded_idx - 1, first_empty_move_idx, -1):
                                    if self.verbose:
                                        print(f"  Repair (P2: Smart Eviction): Deleting redundant empty move at idx {i}")
                                    del vehicle_A_moves[i]
                            else:
                                # No empty moves - insert the new one right after the current move
                                vehicle_A_moves.insert(next_move_A_idx, new_evict_move)

                        return True # Success!

        # --- Fallback to 2-Move Eviction (P2) ---
        # Smart Eviction failed or wasn't applicable.
        if self.verbose:
             print(f"  Repair (P2: Standard Eviction): Smart eviction not possible or failed. "
                   f"Trying 2-move eviction, preferring Empty Lanes then Sink.")
        
        # 1. Try any free buffer lane first (better than Sink)
        free_spot_ap_id = self._find_feasible_eviction_lane(timeline, event_A, event_B, solution)
        
        evicted = False
        if free_spot_ap_id is not None:
             if self.verbose:
                print(f"  Repair (P2: Eviction): Found free lane L{free_spot_ap_id}. Evicting there.")
             evicted = self._perform_eviction(solution, event_A, event_B, free_spot_ap_id)
        
        # 2. If no free buffer lane, try Sink
        if not evicted:
            if self.verbose:
                print(f"  Repair (P2: Eviction): No free buffer lanes. Trying Sink (L{self.sink_ap_id}).")
            
            evicted = self._attempt_eviction_to_specific_lane(
                solution, event_A, event_B, timeline, self.sink_ap_id
            )

        return evicted # Return True/False result of P2

    def _attempt_eviction_to_specific_lane(self, solution: Dict, event_A: Dict, event_B: Dict, timeline: List[Dict], target_lane_ap_id: int) -> bool:
        """Implements Priority 2 logic: Evict a stuck vehicle to a *specific* lane (like the sink)."""
        # The sink is non-blocking, so we don't need to check if it's "free"
        if target_lane_ap_id == self.sink_ap_id:
            return self._perform_eviction(solution, event_A, event_B, target_lane_ap_id)

        # Calculate the ACTUAL time window we need the lane to be free
        # We need it free from the moment we start moving there (evict_start)
        # until the moment we leave (return_start).
        
        lane_A_obj = self._find_virtual_lane_by_ap_id(event_A['lane_id'])
        eviction_lane_obj = self._get_lane_obj_from_location(str(target_lane_ap_id))
        
        if not lane_A_obj or not eviction_lane_obj:
            return False

        evict_start = event_A['start']
        evict_move_dist = self.instance.calculate_distance(lane_A_obj, 1, eviction_lane_obj, 1)
        evict_move_time = int(math.ceil(max(1, evict_move_dist / self.instance.get_vehicle_speed())))
        evict_end = evict_start + evict_move_time
        
        # We stay until B passes
        return_start = max(evict_end, event_B['end'])

        # Check if the *specific* target lane is free during the ENTIRE occupation
        if not self._is_lane_free(timeline, target_lane_ap_id, evict_start, return_start):
             if self.verbose:
                print(f"  Repair (P2: Eviction): Target lane L{target_lane_ap_id} is not free during required window {evict_start}-{return_start}.")
             return False

        # Check if lane is empty of unit loads (if it's a buffer lane)
        if target_lane_ap_id != self.sink_ap_id and target_lane_ap_id != self.source_ap_id:
             if not self._is_lane_empty_of_loads(solution, target_lane_ap_id, evict_start):
                 if self.verbose:
                    print(f"  Repair (P2: Eviction): Target lane L{target_lane_ap_id} is not empty of unit loads.")
                 return False

        return self._perform_eviction(solution, event_A, event_B, target_lane_ap_id)
        
    def _is_lane_free(self, timeline: List[Dict], lane_ap_id: int, start_time: int, end_time: int) -> bool:
        """Checks if a specific lane is free during a window."""
        # A lane_ap_id of < 0 (like our sink/source defaults)
        # is never "free" in this context, as it's not a valid eviction spot.
        if lane_ap_id is None or lane_ap_id < 0:
            return False

        for event in timeline:
            if event['lane_id'] == lane_ap_id:
                # Check for overlap
                if event['start'] < end_time and event['end'] > start_time:
                    return False # Conflict
        return True # No conflicts

    def _find_lane_free_from(self, timeline: List[Dict], lane_ap_id: int, desired_start: int, desired_end: int) -> int:
        """
        Finds the earliest time when a lane becomes continuously free for the entire period [desired_start, desired_end).
        Returns desired_start if already free, otherwise returns the time when it becomes free.
        """
        if lane_ap_id is None or lane_ap_id < 0:
            return desired_start  # Non-blocking lane
        
        # Find all events in this lane
        lane_events = [e for e in timeline if e['lane_id'] == lane_ap_id]
        lane_events.sort(key=lambda x: x['start'])
        
        # Check if desired window is already free
        if self._is_lane_free(timeline, lane_ap_id, desired_start, desired_end):
            return desired_start
        
        # Find the latest event that conflicts with our desired window
        latest_conflict_end = desired_start
        for event in lane_events:
            # Check if this event overlaps with our desired period
            if event['start'] < desired_end and event['end'] > desired_start:
                latest_conflict_end = max(latest_conflict_end, event['end'])
        
        # Now check if there's a continuous free period from latest_conflict_end
        # that's long enough for our stay duration
        stay_duration = desired_end - desired_start
        
        # Try starting from latest_conflict_end
        candidate_start = latest_conflict_end
        candidate_end = candidate_start + stay_duration
        
        # Keep checking until we find a free window
        max_iterations = 100  # Safety limit
        for _ in range(max_iterations):
            if self._is_lane_free(timeline, lane_ap_id, candidate_start, candidate_end):
                return candidate_start
            
            # Find the next conflicting event
            next_conflict = None
            for event in lane_events:
                if event['start'] < candidate_end and event['end'] > candidate_start:
                    if next_conflict is None or event['end'] > next_conflict:
                        next_conflict = event['end']
            
            if next_conflict is None:
                # No more conflicts, we're free!
                return candidate_start
            
            # Move our window to after this conflict
            candidate_start = next_conflict
            candidate_end = candidate_start + stay_duration
        
        # Fallback: return far in the future
        return candidate_start

    def _find_feasible_eviction_lane(self, timeline: List[Dict], event_A: Dict, event_B: Dict, solution: Dict) -> Optional[int]:
        """Helper for P2: Finds a lane that is free during the REQUIRED eviction window."""
        occupied_lane_id = event_A['lane_id']
        all_lanes = set(self.lane_map.keys()) - {self.source_ap_id, self.sink_ap_id, occupied_lane_id}
        
        lane_A_obj = self._find_virtual_lane_by_ap_id(occupied_lane_id)
        if not lane_A_obj:
            return None

        # Sort lanes by distance from the current lane (occupied_lane_id)
        # This prioritizes "nearby" lanes as requested.
        def get_dist(target_ap_id):
            target_obj = self._get_lane_obj_from_location(str(target_ap_id))
            if not target_obj: return float('inf')
            return self.instance.calculate_distance(lane_A_obj, 1, target_obj, 1)

        sorted_lanes = sorted(list(all_lanes), key=get_dist)

        for lane_ap_id in sorted_lanes:
            eviction_lane_obj = self._get_lane_obj_from_location(str(lane_ap_id))
            if not eviction_lane_obj:
                continue

            # Calculate required window for THIS lane
            evict_start = event_A['start']
            
            # Calculate safe tier to be accurate about travel time and blocking
            evict_target_tier = self._calculate_safe_entry_tier(lane_ap_id, evict_start, solution)
            
            # Calculate exit duration (how long we block the lane while leaving)
            # We need to ensure the lane is free until we have FULLY left.
            dummy_return_move = {
                'from_location': str(lane_ap_id),
                'from_tier': evict_target_tier,
                'move_type': 'empty'
            }
            exit_duration = self._get_lane_blocking_duration(dummy_return_move, 'from')

            evict_move_dist = self.instance.calculate_distance(lane_A_obj, 1, eviction_lane_obj, evict_target_tier)
            evict_move_time = int(math.ceil(max(1, evict_move_dist / self.instance.get_vehicle_speed())))
            evict_end = evict_start + evict_move_time
            
            return_start = max(evict_end, event_B['end'])
            
            # Check if lane is free from arrival until we FULLY leave (return_start + exit_duration)
            # PLUS a safety buffer to avoid tight scheduling with future tasks
            safety_buffer = 30 
            if self._is_lane_free(timeline, lane_ap_id, evict_start, return_start + exit_duration + safety_buffer):
                # Check if lane is empty of unit loads
                if self._is_lane_empty_of_loads(solution, lane_ap_id, evict_start):
                    return lane_ap_id
                
        return None

    def _get_lane_obj_from_location(self, location_str: str) -> Any:
        """
        Gets the correct object for calculate_distance (VirtualLane, "source", or "sink").
        """
        # Handle explicit string markers
        if location_str == 'source':
            return "source"
        if location_str == 'sink':
            return "sink"
        
        # Try to parse as an AP ID
        try:
            ap_id = int(location_str)
            # Check if this AP ID is the source or sink
            if ap_id == self.source_ap_id:
                return "source"
            if ap_id == self.sink_ap_id:
                return "sink"
            # Otherwise, it's a regular lane
            return self._find_virtual_lane_by_ap_id(ap_id)
        except (ValueError, TypeError):
            return None

    def _perform_eviction(self, solution: Dict, event_A: Dict, event_B: Dict, eviction_lane_ap_id: int) -> bool:
        """
        Helper function to execute the schedule modification for a 2-move eviction.
        """
        if self.verbose:
            print(f"  Repair (P2: Standard Eviction): Evicting V{event_A['vehicle_id']} from L{event_A['lane_id']} "
                  f"to L{eviction_lane_ap_id}.")

        vehicle_A_moves = solution['vehicles'][event_A['v_idx']]['moves']
        next_move_A_idx = event_A['move_idx'] + 1
        
        # --- OPTIMIZATION: Skip intermediate empty moves ---
        # Find the next loaded move to target directly, avoiding "ping-pong" moves (e.g. 2->4->2->3)
        next_loaded_idx = self._find_next_loaded_move(vehicle_A_moves, next_move_A_idx)
        
        target_loc = None
        target_tier = 1
        
        if next_loaded_idx is not None:
            # Found a future task. Target it directly.
            target_move = vehicle_A_moves[next_loaded_idx]
            target_loc = str(target_move['from_location'])
            target_tier = target_move.get('from_tier', 1)
            
            # Remove all intermediate empty moves (we will replace them with the return move)
            # The range to remove is [next_move_A_idx, next_loaded_idx)
            if next_loaded_idx > next_move_A_idx:
                if self.verbose:
                    print(f"  Repair (P2): Optimizing eviction path. Removing {next_loaded_idx - next_move_A_idx} intermediate empty moves.")
                del vehicle_A_moves[next_move_A_idx:next_loaded_idx]
                
        else:
            # No future loaded moves. 
            # Fallback to immediate next move if it exists.
            if next_move_A_idx < len(vehicle_A_moves):
                target_move = vehicle_A_moves[next_move_A_idx]
                target_loc = str(target_move['from_location'])
                target_tier = target_move.get('from_tier', 1)
            else:
                # No moves left. Target current lane (stay put logic, though eviction forces move out)
                target_loc = str(event_A['lane_id'])
                target_tier = 1

        # --- [CRASH FIX] Get Lane Objects for distance calculation ---
        lane_A_obj = self._find_virtual_lane_by_ap_id(event_A['lane_id'])
        eviction_lane_obj = self._get_lane_obj_from_location(str(eviction_lane_ap_id))
        next_move_from_lane_obj = self._get_lane_obj_from_location(target_loc)

        if not lane_A_obj or not eviction_lane_obj or not next_move_from_lane_obj:
             if self.verbose: 
                 print(f"  Repair (P2: Eviction ABORTED): Could not find lane objects for eviction. "
                       f"(A: {lane_A_obj}, Evict: {eviction_lane_obj}, Next: {next_move_from_lane_obj})")
             return False

        # 1. Create Evict Move (Parked Lane -> Eviction Lane)
        evict_start = event_A['start']  # Starts when park began
        
        # Get current tier from the move that brought us here
        prev_move = vehicle_A_moves[event_A['move_idx']]
        current_tier = prev_move.get('to_tier', 1)
        
        # FIX: Calculate the safe tier.
        evict_target_tier = self._calculate_safe_entry_tier(eviction_lane_ap_id, evict_start, solution)
        
        evict_move_dist = self.instance.calculate_distance(lane_A_obj, current_tier, eviction_lane_obj, evict_target_tier)
        evict_move_time = int(math.ceil(max(1, evict_move_dist / self.instance.get_vehicle_speed())))
        evict_end = evict_start + evict_move_time

        evict_move = {
            'move_type': 'empty', 'ul_id': 0,
            'from_location': str(event_A['lane_id']), 'to_location': str(eviction_lane_ap_id),
            'from_tier': current_tier, 'to_tier': evict_target_tier,  # Use calculated tier
            'start_time': evict_start, 'end_time': evict_end,
            'travel_distance': evict_move_dist, 'service_time': evict_move_time,
            'move_type_internal': 'evict_out'
        }

        # 2. Create Return Move (Eviction Lane -> Next Loaded Move's Start)
        
        return_move_dist = self.instance.calculate_distance(
            eviction_lane_obj, evict_target_tier,
            next_move_from_lane_obj, target_tier
        )
        return_move_time = int(math.ceil(max(1, return_move_dist / self.instance.get_vehicle_speed())))

        # Must wait for intruder (B) to pass
        min_return_start = max(evict_end, event_B['end']) # Wait for B to *end*
        
        # OPTIMIZATION: Don't return early! Stay at sink until needed.
        # Aim to arrive at the target exactly when the next move is scheduled to start.
        desired_arrival = target_move['start_time']
        just_in_time_start = desired_arrival - return_move_time
        
        return_start = max(min_return_start, just_in_time_start)
        return_end = return_start + return_move_time

        return_move = {
            'move_type': 'empty', 'ul_id': 0,
            'from_location': str(eviction_lane_ap_id), 'to_location': target_loc,
            'from_tier': evict_target_tier, 'to_tier': target_tier,
            'start_time': return_start, 'end_time': return_end,
            'travel_distance': return_move_dist, 'service_time': return_move_time,
            'move_type_internal': 'evict_return'
        }

        # 3. Insert new moves
        vehicle_A_moves.insert(next_move_A_idx, evict_move)
        vehicle_A_moves.insert(next_move_A_idx + 1, return_move)

        # 4. Propagate delay to all subsequent moves in V-A's schedule
        if (next_move_A_idx + 2) < len(vehicle_A_moves):
             next_real_move = vehicle_A_moves[next_move_A_idx + 2]
             delay = return_end - next_real_move['start_time']
             if delay > 0:
                 if self.verbose:
                      print(f"  Repair (P2: Eviction): Eviction complete. Delaying V{event_A['vehicle_id']}'s schedule by {delay} units.")
                 # +2 because we added 2 moves
                 self._delay_intruder_and_propagate(solution, event_A['v_idx'], next_move_A_idx + 2, delay)

        return True

    def _delay_intruder(self, solution: Dict, event_A: Dict, event_B: Dict):
        """Implements P3: Delay the intruding vehicle (B)."""
        # Delay must be at least 1 time unit to resolve start/end overlap
        delay = max(1, (event_A['end'] - event_B['start']))
        
        if self.verbose:
            print(f"  Repair (P3: Fallback): Delaying V{event_B['vehicle_id']}'s schedule by {delay} units.")

        self._delay_intruder_and_propagate(solution, event_B['v_idx'], event_B['move_idx'], delay)

    def _propagate_schedule_shift(self, solution: Dict, v_idx: int, 
                                start_move_idx: int, end_move_idx: int,
                                delay: int, start_time: int):
        """
        Propagation for P1: ONLY shifts empty moves in the block
        [start_move_idx, end_move_idx). 'delay' is (likely) negative.
        'start_time' is the new, safe start time for the *first* move in the block.
        """
        moves = solution['vehicles'][v_idx]['moves']
        
        current_time = start_time
        for i in range(start_move_idx, end_move_idx): # Iterate *only* over the block
            move = moves[i]
            
            duration = move['end_time'] - move['start_time']
            new_start = current_time
            new_end = new_start + max(1, duration) # Ensure at least 1 unit

            move['start_time'] = new_start
            move['end_time'] = new_end
            
            # Next move must start *after* this one ends
            current_time = new_end

    def _delay_intruder_and_propagate(self, solution: Dict, v_idx: int, start_move_idx: int, delay: int):
        """
        Propagation for P2/P3: Shifts ALL subsequent moves by a
        positive 'delay'.
        """
        moves = solution['vehicles'][v_idx]['moves']
        for i in range(start_move_idx, len(moves)):
            moves[i]['start_time'] += delay
            moves[i]['end_time'] += delay

    def _attempt_retrieval_steal(self, solution: Dict, event_A: Dict, event_B: Dict, timeline: List[Dict]) -> bool:
        """
        Attempt to steal a retrieval task from the intruding vehicle.
        
        This is used when:
        - Vehicle A is parked at a lane with no more moves (likely after a storage move)
        - Vehicle B tries to access that same lane (likely with an empty move followed by a retrieval)
        
        Strategy:
        1. Check if A's last move was a storage move to this lane
        2. Check if B's current move is an empty move to this lane
        3. Check if B's next move is a retrieval from this lane
        4. If all true, give the retrieval task to A and remove the empty+retrieval moves from B
        
        Returns True if successful, False otherwise.
        """
        vehicle_A = solution['vehicles'][event_A['v_idx']]
        vehicle_B = solution['vehicles'][event_B['v_idx']]
        
        # Check if A's last move was a storage move to the collision lane
        if not vehicle_A['moves']:
            return False
            
        last_move_A = vehicle_A['moves'][event_A['move_idx']]
        if last_move_A['move_type'] not in ('store', 'z'):
            return False
        
        # Check that A ended at the collision lane
        lane_A = self._get_lane_id_for_repair(last_move_A.get('to_location'))
        if lane_A != event_A['lane_id']:
            return False
            
        # Check if B is trying to do an empty move to this lane
        if event_B['move_idx'] >= len(vehicle_B['moves']):
            return False
            
        move_B = vehicle_B['moves'][event_B['move_idx']]
        if move_B['move_type'] not in ('empty', 'e'):
            return False
            
        # Check that B is going to the collision lane
        lane_B = self._get_lane_id_for_repair(move_B.get('to_location'))
        if lane_B != event_A['lane_id']:
            return False
            
        # Check if B's next move is a retrieval from this lane
        next_move_idx_B = event_B['move_idx'] + 1
        if next_move_idx_B >= len(vehicle_B['moves']):
            return False
            
        next_move_B = vehicle_B['moves'][next_move_idx_B]
        if next_move_B['move_type'] not in ('retrieve', 'y'):
            return False
            
        # Check that the retrieval is from the collision lane
        lane_next_B = self._get_lane_id_for_repair(next_move_B.get('from_location'))
        if lane_next_B != event_A['lane_id']:
            return False
            
        if self.verbose:
            print(f"    Retrieval Steal: V{event_A['vehicle_id']} stored at lane {event_A['lane_id']}, "
                  f"V{event_B['vehicle_id']} wants to retrieve from there.")
            print(f"    Stealing retrieval move from V{event_B['vehicle_id']} to V{event_A['vehicle_id']}.")
        
        # Create the retrieval move for A
        # A is already parked at the lane, so it can start the retrieval immediately
        # Start time: when A finished parking (last move end time)
        # We need to respect when B would have done it (time windows)
        
        # The retrieval should happen at the time B would have arrived at the lane
        retrieval_start = move_B['end_time']  # When B would have arrived
        
        # Create new retrieval move for A
        move_duration = next_move_B['end_time'] - next_move_B['start_time']
        new_move_A = {
            'move_type': next_move_B['move_type'],  # 'retrieve' or 'y'
            'ul_id': next_move_B.get('ul_id', next_move_B.get('unit_load_id', 0)),
            'from_location': next_move_B['from_location'],
            'to_location': next_move_B['to_location'],
            'from_tier': next_move_B.get('from_tier', 1),
            'to_tier': next_move_B.get('to_tier', 1),
            'unit_load_id': next_move_B.get('unit_load_id'),
            'start_time': retrieval_start,
            'end_time': retrieval_start + move_duration,
            'travel_distance': next_move_B.get('travel_distance', 0),
            'service_time': move_duration,
            'distance': next_move_B.get('distance', next_move_B.get('travel_distance', 0)),
        }
        
        # Add the new move to A
        vehicle_A['moves'].append(new_move_A)
        
        if self.verbose:
            print(f"    Added retrieval move to V{event_A['vehicle_id']}: "
                  f"{new_move_A['start_time']}-{new_move_A['end_time']} "
                  f"{new_move_A['from_location']} -> {new_move_A['to_location']}")
        
        # Remove the empty move and retrieval move from B
        # We need to remove TWO consecutive moves: the empty move and the retrieval
        del vehicle_B['moves'][event_B['move_idx']]  # Remove empty move
        # After deleting, the retrieval is now at event_B['move_idx']
        if event_B['move_idx'] < len(vehicle_B['moves']):
            del vehicle_B['moves'][event_B['move_idx']]  # Remove retrieval
        
        if self.verbose:
            print(f"    Removed empty and retrieval moves from V{event_B['vehicle_id']}.")
        
        return True

    def _recalculate_solution_metrics(self, solution: Dict) -> Dict:
        """Updates total_distance and total_time for the final solution."""
        total_distance = 0
        makespan = 0
        for vehicle in solution.get('vehicles', []):
            vehicle_dist = 0
            vehicle_time = 0
            for move in vehicle.get('moves', []):
                vehicle_dist += move.get('travel_distance', 0)
                if move.get('end_time', 0) > vehicle_time:
                    vehicle_time = move.get('end_time', 0)

            vehicle['total_distance'] = vehicle_dist
            vehicle['total_time'] = vehicle_time
            total_distance += vehicle_dist
            if vehicle_time > makespan:
                makespan = vehicle_time

        solution['total_distance'] = total_distance
        solution['total_time'] = makespan
        return solution

    def _calculate_safe_entry_tier(self, lane_ap_id: int, at_time: int, solution: Dict = None) -> int:
        """
        Determines the safe tier to enter for a given lane at a specific time.
        For Source/Sink, returns 1.
        For Buffer Lanes, checks if the lane is empty.
        If empty, returns 1.
        If occupied, returns 2 (assuming we can stack on top) or 1 if we risk it (but we shouldn't).
        
        Current logic: Conservative. If lane is not empty, we try to stack on top (Tier 2).
        If we can't verify, we default to 1 but this is risky.
        """
        if lane_ap_id == self.source_ap_id or lane_ap_id == self.sink_ap_id:
            return 1
            
        # Check if lane is empty of unit loads
        if solution and self._is_lane_empty_of_loads(solution, lane_ap_id, at_time):
            return 1
            
        # If not empty, we should ideally return the next available tier.
        # Since we don't track exact stack heights easily, we'll assume Tier 2 is safe(r) than Tier 1
        # if the lane is occupied. 
        # WARNING: This assumes max tier > 1.
        lane_obj = self._find_virtual_lane_by_ap_id(lane_ap_id)
        if lane_obj:
            max_tiers = 1
            if hasattr(lane_obj, 'get_tiers') and lane_obj.get_tiers():
                max_tiers = len(lane_obj.get_tiers())
            elif hasattr(lane_obj, 'stacks') and lane_obj.stacks is not None:
                max_tiers = len(lane_obj.stacks)
            
            if max_tiers > 1:
                return 2 # Try to stack on top
        
        return 1 # Fallback

    def _is_lane_empty_of_loads(self, solution: Dict, lane_ap_id: int, at_time: int) -> bool:
        """
        Checks if a lane contains any unit loads at a given time.
        Considers initial state and all store/retrieve moves up to `at_time`.
        """
        # 1. Check initial state
        # We need to access the initial buffer state. 
        # Assuming the instance has this info, but it might be complex to access.
        # For now, we'll assume empty initial state or rely on the solution moves.
        # If the instance has a way to check initial load, use it.
        # (Skipping complex initial state check for now as it requires parsing the instance deeply)
        
        load_count = 0
        
        # 2. Check solution moves
        for vehicle in solution.get('vehicles', []):
            for move in vehicle.get('moves', []):
                if move['end_time'] > at_time:
                    continue # Future move
                
                if move['move_type'] in ('store', 'z'):
                    to_lane = self._get_lane_id_for_repair(move.get('to_location'))
                    if to_lane == lane_ap_id:
                        load_count += 1
                elif move['move_type'] in ('retrieve', 'y'):
                    from_lane = self._get_lane_id_for_repair(move.get('from_location'))
                    if from_lane == lane_ap_id:
                        load_count -= 1
        
        return load_count <= 0

    def _enforce_dependencies(self, solution: Dict):
        """
        Post-processing check to ensure Store times < Retrieve times.
        If violated, pushes Retrieve times forward.
        """
        # 1. Map Unit Load IDs to their Store and Retrieve events
        ul_events = {} # {ul_id: {'store': end_time, 'retrieve_idx': (v_idx, m_idx)}}
        
        for v_idx, vehicle in enumerate(solution.get('vehicles', [])):
            for m_idx, move in enumerate(vehicle['moves']):
                ul_id = move.get('ul_id') or move.get('unit_load_id')
                if not ul_id: continue
                
                if move['move_type'] in ('store', 'z'):
                    if ul_id not in ul_events: ul_events[ul_id] = {}
                    ul_events[ul_id]['store_end'] = move['end_time']
                    
                elif move['move_type'] in ('retrieve', 'y'):
                    if ul_id not in ul_events: ul_events[ul_id] = {}
                    ul_events[ul_id]['retrieve_loc'] = (v_idx, m_idx)
                    ul_events[ul_id]['retrieve_start'] = move['start_time']

        # 2. Check for violations
        for ul_id, times in ul_events.items():
            if 'store_end' in times and 'retrieve_start' in times:
                if times['retrieve_start'] < times['store_end']:
                    # VIOLATION DETECTED
                    if 'retrieve_loc' in times:
                        v_idx, m_idx = times['retrieve_loc']
                        required_delay = times['store_end'] - times['retrieve_start']
                        
                        if self.verbose:
                            print(f"  Repair (Dependency): Correcting order for UL {ul_id}. Pushing retrieval by {required_delay}.")
                        
                        # Propagate delay to the retrieving vehicle
                        self._delay_intruder_and_propagate(solution, v_idx, m_idx, required_delay)