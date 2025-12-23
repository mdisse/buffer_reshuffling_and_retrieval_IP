import numpy as np
import copy
from src.examples_gen.unit_load import UnitLoad
from src.bay.buffer import Buffer
from src.examples_gen.rand_lane_gen import RandLaneGen

class MultiRobotConstructiveGenerator:
    def __init__(self, num_robots: int, seed: int = 42, fill_level: float = 0.8):
        self.num_robots = num_robots
        self.seed = seed
        self.fill_level = fill_level
        self.rng = np.random.default_rng(seed)
        self.lane_gen = RandLaneGen(self.rng)
        self.fill_order = []
        
        # Configuration
        self.reshuffle_penalty = 5
        self.base_op_time = 5
        self.slack = 15  # Widen slack for multi-robot to account for traffic
        self.min_storage_time = 0 # Will be set in generate_bays_priorities

    def _get_free_slots(self):
        """Finds all (bay_idx, lane, depth) tuples that are empty (0)."""
        free = []
        for b_idx, bay in enumerate(self.sim_bays):
            it = np.nditer(bay.state, flags=['multi_index'])
            for x in it:
                if x == 0:
                    free.append((b_idx, *it.multi_index))
        return free

    def _get_blockers(self, bay_idx, pos):
        """Identify blockers (simplified stack logic)."""
        bay = self.sim_bays[bay_idx]
        target_l, target_w, target_h = pos
        blockers = []
        # Check higher indices (assuming they block lower indices)
        for h in range(target_h + 1, bay.height):
             val = bay.state[target_l, target_w, h]
             if val != 0:
                 blockers.append(val)
        return blockers

    def _assign_robot(self, min_start_time=0):
        """
        Greedy Assignment: Find the robot that becomes free earliest, 
        but not before min_start_time.
        """
        # We want the robot who can start the task soonest.
        # Start Time = max(Robot_Free_Time, min_start_time)
        best_robot = -1
        best_start = float('inf')
        
        for r_id, free_t in enumerate(self.robot_free_times):
            possible_start = max(free_t, min_start_time)
            if possible_start < best_start:
                best_start = possible_start
                best_robot = r_id
                
        # To balance load, if multiple robots have same start time, pick random?
        # For feasibility, greedy is fine.
        return best_robot, best_start

    def generate(self, num_loads: int, fill_ratio_target: float = 0.8, initial_loads_count: int = 0):
        next_id = 1
        loads_to_create = num_loads - initial_loads_count
        finished_loads = []
        initial_allocation = []
        self.generated_trace = []
        
        total_slots = sum(b.state.size for b in self.sim_bays)
        
        # --- PRE-FILL BUFFER ---
        fill_idx = 0
        for _ in range(initial_loads_count):
            ul_id = next_id
            # Create load with negative arrival times
            # arrival_start = -100, arrival_end = -50 (arbitrary past)
            ul = UnitLoad(ul_id, 0, 0, -100, -50, is_mock=True)
            ul.stored = True # Explicitly mark as stored
            
            chosen_slot = None
            # Try to use fill_order if available
            if self.fill_order and fill_idx < len(self.fill_order):
                # Find next free slot in fill order
                while fill_idx < len(self.fill_order):
                    candidate = self.fill_order[fill_idx]
                    fill_idx += 1
                    b_idx, l, w, h = candidate
                    if self.sim_bays[b_idx].state[l, w, h] == 0:
                        chosen_slot = candidate
                        break
            
            if chosen_slot is None:
                free_slots_list = self._get_free_slots()
                if not free_slots_list:
                    break # Buffer full
                chosen_slot = self.rng.choice(free_slots_list)

            b_idx, l, w, h = chosen_slot
            
            # Update State directly (no robot time used)
            self.sim_bays[b_idx].state[l, w, h] = ul_id
            self.stored_positions[ul_id] = chosen_slot
            self.active_loads[ul_id] = ul
            self.load_ready_times[ul_id] = 0 # Ready immediately
            initial_allocation.append((b_idx, l, w, h, ul_id))
            
            next_id += 1
        
        # We simulate a continuous stream of requests
        while loads_to_create > 0 or len(self.active_loads) > 0:
            
            free_slots_list = self._get_free_slots()
            current_fill = len(self.active_loads) / total_slots if total_slots > 0 else 1.0
            
            # DECISION: Generate STORE or RETRIEVE task?
            can_store = (loads_to_create > 0) and (len(free_slots_list) > 0)
            can_retrieve = (len(self.active_loads) > 0)
            
            # Logic to keep buffer near target fill ratio
            if not can_retrieve: action = "STORE"
            elif not can_store: action = "RETRIEVE"
            elif current_fill < fill_ratio_target:
                action = "STORE" if self.rng.random() < 0.7 else "RETRIEVE"
            else:
                action = "RETRIEVE" if self.rng.random() < 0.7 else "STORE"

            if action == "STORE":
                # --- SIMULATE STORAGE ---
                ul_id = next_id
                ul = UnitLoad(ul_id, 0, 0, 0, 0, is_mock=True)
                
                # 1. Pick Slot
                # free_slots = self._get_free_slots() # Already got it
                chosen_slot = self.rng.choice(free_slots_list) # (bay, l, w, h)
                
                # 2. Assign Robot
                # Storage can start whenever a robot is free
                r_id, start_time = self._assign_robot(min_start_time=0)
                
                # 3. Execution
                duration = self.base_op_time
                end_time = start_time + duration
                
                self.generated_trace.append({
                    'type': 'store',
                    'ul_id': ul_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'robot_id': r_id,
                    'slot': chosen_slot
                })
                
                # 4. Update State
                self.robot_free_times[r_id] = end_time
                self.load_ready_times[ul_id] = end_time  # Item physically in slot now
                
                b_idx, l, w, h = chosen_slot
                self.sim_bays[b_idx].state[l, w, h] = ul_id
                self.stored_positions[ul_id] = chosen_slot
                self.active_loads[ul_id] = ul
                
                # 5. Set Time Window (Arrival)
                # It arrives roughly when we started processing it
                # Add slack to give solver flexibility
                ul.arrival_start = max(0, start_time - self.slack)
                ul.arrival_end = start_time + self.slack
                
                loads_to_create -= 1
                next_id += 1
                
            elif action == "RETRIEVE":
                # --- SIMULATE RETRIEVAL ---
                # 1. Pick Target
                # Filter candidates to ensure they have been stored for at least min_storage_time
                # We estimate current time as the minimum free time of any robot
                current_sim_time = min(self.robot_free_times)
                
                candidates = []
                for uid, ul in self.active_loads.items():
                    ready_time = self.load_ready_times[uid]
                    # Check if item has been stored long enough relative to when a robot *could* start
                    if ready_time + self.min_storage_time <= current_sim_time + 100: # Allow some lookahead
                         candidates.append(uid)
                
                # Fallback: if no candidates meet criteria, pick the oldest one (smallest ready_time)
                if not candidates:
                    candidates = sorted(self.active_loads.keys(), key=lambda k: self.load_ready_times[k])[:1]
                
                target_id = self.rng.choice(candidates)
                target_ul = self.active_loads[target_id]
                
                # 2. Constraint: Item must be physically stored first
                ready_time = self.load_ready_times[target_id]
                
                # 3. Assign Robot
                # Robot can only start AFTER item is ready AND after arrival window ends (to ensure consistency)
                # Also ensure start_time >= 1 because retrieval_start must be >= 1
                min_start = max(ready_time, target_ul.arrival_end, 1)
                r_id, start_time = self._assign_robot(min_start_time=min_start)
                
                # 4. Check & Move Blockers (Reshuffling)
                # Note: We assign these reshuffles to the SAME robot for simplicity
                # to guarantee atomic execution, though in reality others could help.
                # Doing it with one robot guarantees feasibility.
                
                b_idx, l, w, h = self.stored_positions[target_id]
                blockers = self._get_blockers(b_idx, (l, w, h))
                
                current_task_time = start_time
                
                for blk_id in blockers:
                    # Move blocker to new spot
                    free_slots = [s for s in self._get_free_slots() if s != (b_idx, l, w, h)]
                    if not free_slots: continue 
                    dest = self.rng.choice(free_slots)
                    
                    # Update State
                    old_blk_pos = self.stored_positions[blk_id]
                    self.sim_bays[old_blk_pos[0]].state[old_blk_pos[1:]] = 0
                    self.sim_bays[dest[0]].state[dest[1:]] = blk_id
                    self.stored_positions[blk_id] = dest
                    
                    # Advance Time (Simulate the move duration)
                    move_start = current_task_time
                    current_task_time += self.reshuffle_penalty
                    
                    self.generated_trace.append({
                        'type': 'reshuffle',
                        'ul_id': blk_id,
                        'start_time': move_start,
                        'end_time': current_task_time,
                        'robot_id': r_id,
                        'from': old_blk_pos,
                        'to': dest
                    })
                
                # 5. Retrieve Target
                move_start = current_task_time
                current_task_time += self.base_op_time
                
                self.generated_trace.append({
                    'type': 'retrieve',
                    'ul_id': target_id,
                    'start_time': move_start,
                    'end_time': current_task_time,
                    'robot_id': r_id
                })
                
                # Update Buffer State
                self.sim_bays[b_idx].state[l, w, h] = 0
                del self.stored_positions[target_id]
                del self.active_loads[target_id]
                del self.load_ready_times[target_id]
                
                # Update Robot State
                self.robot_free_times[r_id] = current_task_time
                
                # 6. Set Time Window (Retrieval)
                # Target leaves at current_task_time
                # Ensure retrieval_start >= arrival_end
                target_ul.retrieval_start = max(1, current_task_time - self.slack, target_ul.arrival_end)
                # Ensure retrieval_end > retrieval_start
                target_ul.retrieval_end = max(target_ul.retrieval_start + 1, current_task_time + self.slack)
                
                finished_loads.append(target_ul)
                
        return finished_loads, initial_allocation

    def generate_bays_priorities(self, bays: list, height: int = 1, source=True):
        """
        Interface method to be called by Instance class.
        """
        self.sim_bays = copy.deepcopy(bays)
        
        # Initialize state for bays if needed
        for bay in self.sim_bays:
            bay.height = height
            # Always reset state to empty 3D array for simulation
            bay.state = np.zeros((bay.length, bay.width, bay.height), dtype=int)
            
        # Calculate dimensions for time estimation
        max_x = max(b.x + b.width for b in self.sim_bays)
        max_y = max(b.y + b.length for b in self.sim_bays)

        # Manhattan distance from origin (0,0) to furthest point (max_x, max_y)
        # This is a good approximation for "Source -> Buffer -> Sink"
        manhattan_dist = max_x + max_y 

        # Add 50% buffer for turns, acceleration, and avoidance
        self.base_op_time = int(manhattan_dist * 2.0)
        
        # Set a "Hard Floor" to ensure it's not too low for small grids
        self.base_op_time = max(self.base_op_time, 25)

        self.min_storage_time = self.base_op_time * 2 # Force items to stay for a bit
        self.reshuffle_penalty = self.base_op_time # Reshuffling takes time
        
        # Initialize simulation state
        self.robot_free_times = [0] * self.num_robots
        self.load_ready_times = {}
        self.active_loads = {}
        self.stored_positions = {}
        
        # Generate fill order for pre-filling (Back-to-Front)
        self.fill_order = []
        all_slots_with_score = []
        
        for b_idx, bay in enumerate(self.sim_bays):
            lanes, _ = self.lane_gen.generate_lanes(bay)
            for lane in lanes:
                for depth, (l, w) in enumerate(lane):
                    for h in range(bay.height):
                        # Score: Higher depth is better (back of lane). 
                        # Lower height is better (bottom of stack).
                        # We sort descending, so we want higher score to mean "pick first".
                        # depth: 0 (front) ... N (back). We want N.
                        # height: 0 (bottom) ... M (top). We want 0.
                        # So score tuple: (depth, -height)
                        all_slots_with_score.append({
                            'slot': (b_idx, l, w, h),
                            'score': (depth, -h),
                            'rand': self.rng.random() # Tie breaker to mix lanes/bays
                        })
        
        # Sort descending based on score
        all_slots_with_score.sort(key=lambda x: (x['score'], x['rand']), reverse=True)
        self.fill_order = [x['slot'] for x in all_slots_with_score]
        
        # Determine number of loads to generate
        # We generate enough loads to match the requested fill level
        total_slots = sum(b.state.size for b in self.sim_bays)
        num_loads = int(total_slots * self.fill_level)
        if num_loads == 0: num_loads = 1 # Ensure at least one load
        
        initial_loads_count = num_loads // 2
        
        finished_loads, initial_allocation = self.generate(num_loads, self.fill_level, initial_loads_count=initial_loads_count)
        
        # Apply initial allocation to the provided bays
        for (b_idx, l, w, h, ul_id) in initial_allocation:
            bays[b_idx].state[l, w, h] = ul_id
        
        return finished_loads
