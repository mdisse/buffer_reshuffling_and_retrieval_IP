import heapq
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from src.examples_gen.unit_load import UnitLoad

# Constants for move types
MOVE_TYPE_STORE = 'store'
MOVE_TYPE_RETRIEVE = 'retrieve'
MOVE_TYPE_RESHUFFLE = 'reshuffle'

@dataclass
class AStarConfig:
    """Configuration for A* solver."""
    verbose: bool = False
    time_limit: float = 300.0  
    max_reshuffle_branching: int = 5  
    
@dataclass 
class MoveInfo:
    """Represents a move in the warehouse."""
    ul_id: int
    move_type: str
    from_pos: Optional[int] = None
    to_pos: Optional[int] = None
    from_tier: Optional[int] = None
    to_tier: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = {
            'ul_id': self.ul_id,
            'type': self.move_type
        }
        if self.from_pos is not None:
            result['from_pos'] = self.from_pos
        if self.to_pos is not None:
            result['to_pos'] = self.to_pos
        if self.from_tier is not None:
            result['from_tier'] = self.from_tier
        if self.to_tier is not None:
            result['to_tier'] = self.to_tier
        return result

class UnitLoadManager:
    """Manages unit load priorities and state transitions."""
    
    def __init__(self, unit_loads: List[UnitLoad], task_queue=None, verbose: bool = False):
        self.verbose = verbose
        self.unit_loads = self._initialize_unit_loads(unit_loads, task_queue)
        self.ul_priority_map = {ul.id: ul.priority for ul in self.unit_loads}
    
    def _initialize_unit_loads(self, unit_loads: List[UnitLoad], task_queue) -> List[UnitLoad]:
        """Initialize unit loads with proper priorities."""
        if task_queue:
            return self._apply_task_queue_priorities(unit_loads, task_queue)
        else: 
            raise ValueError("Task queue must be provided to initialize unit loads with priorities.")
    
    def _apply_task_queue_priorities(self, unit_loads: List[UnitLoad], task_queue) -> List[UnitLoad]:
        """Apply priorities from task queue to unit loads."""
        storage_priority_map = {}
        retrieval_priority_map = {}
        
        # Extract priorities from task queue
        for task in task_queue:
            # Get priority using get_priority() method and check it's not None
            task_priority = task.get_priority() if hasattr(task, 'get_priority') else getattr(task, 'priority', None)
            
            if task_priority is not None:
                if "_mock" in str(task.id):
                    real_ul_id = task.real_ul_id
                    storage_priority_map[real_ul_id] = task_priority
                else:
                    retrieval_priority_map[task.id] = task_priority
        
        # Verify we extracted priorities from the task queue
        if not storage_priority_map and not retrieval_priority_map:
            raise ValueError("No priorities were extracted from the provided task queue.")
        
        # Apply priorities to unit loads
        for ul in unit_loads:
            storage_priority = storage_priority_map.get(ul.id, 999)
            retrieval_priority = retrieval_priority_map.get(ul.id, 999)
            
            ul.set_storage_priority(storage_priority)
            ul.set_retrieval_priority(retrieval_priority)
            
            # Set initial priority based on current state
            if ul.is_stored:
                ul.priority = retrieval_priority
            else:
                ul.priority = self._select_initial_priority(storage_priority, retrieval_priority)
        
        if self.verbose:
            self._log_priority_assignments(unit_loads)
        
        return unit_loads
    
    def _select_initial_priority(self, storage_priority: int, retrieval_priority: int) -> int:
        """Select initial priority for unit load at source."""
        if storage_priority < 900 and retrieval_priority < 900:
            return min(storage_priority, retrieval_priority)
        elif storage_priority < 900:
            return storage_priority
        elif retrieval_priority < 900:
            return retrieval_priority
        else:
            return 999
    
    def _log_priority_assignments(self, unit_loads: List[UnitLoad]) -> None:
        """Log priority assignments for debugging and validate priority grouping."""
        print(f"✅ Applied task queue priorities to {len(unit_loads)} unit loads")
        print("  Storage and Retrieval Priorities:")
        
        sorted_uls = sorted(unit_loads[:15], key=lambda ul: ul.id)
        for ul in sorted_uls:
            storage_pri = getattr(ul, 'storage_priority', 'N/A')
            retrieval_pri = getattr(ul, 'retrieval_priority', 'N/A')
            current_pri = ul.priority
            stored_status = "stored" if ul.is_stored else "at_source"
            retrieval_end = getattr(ul, 'retrieval_end', 'N/A')
            print(f"  UL {ul.id}: storage_p={storage_pri}, retrieval_p={retrieval_pri}, "
                  f"current_p={current_pri}, retrieval_end={retrieval_end} ({stored_status})")
        
        if len(unit_loads) > 15:
            print(f"  ... and {len(unit_loads) - 15} more")
        
        # Validate priority grouping: ULs with same retrieval_end should have same retrieval_priority
        retrieval_end_to_priority = {}
        priority_conflicts = []
        
        for ul in unit_loads:
            if hasattr(ul, 'retrieval_end') and ul.retrieval_end is not None:
                if ul.retrieval_end in retrieval_end_to_priority:
                    expected_priority = retrieval_end_to_priority[ul.retrieval_end]
                    if hasattr(ul, 'retrieval_priority') and ul.retrieval_priority != expected_priority:
                        priority_conflicts.append({
                            'ul_id': ul.id,
                            'retrieval_end': ul.retrieval_end,
                            'expected_priority': expected_priority,
                            'actual_priority': ul.retrieval_priority
                        })
                else:
                    if hasattr(ul, 'retrieval_priority'):
                        retrieval_end_to_priority[ul.retrieval_end] = ul.retrieval_priority
        
        if priority_conflicts:
            print("\n  ⚠️  PRIORITY GROUPING CONFLICTS DETECTED:")
            print("  Unit loads with the SAME retrieval deadline have DIFFERENT priorities!")
            print("  This breaks same-priority blocking detection (>= condition won't work)")
            for conflict in priority_conflicts[:5]:  # Show first 5
                print(f"    UL {conflict['ul_id']}: retrieval_end={conflict['retrieval_end']}, "
                      f"priority={conflict['actual_priority']} (expected {conflict['expected_priority']})")
            if len(priority_conflicts) > 5:
                print(f"    ... and {len(priority_conflicts) - 5} more conflicts")
            print("  ❌ This indicates internal priority calculation was used instead of task_queue!")
        else:
            print("\n  ✅ Priority grouping validated: Same retrieval_end = Same priority")
    
    def get_ul_by_id(self, ul_id: int) -> Optional[UnitLoad]:
        """Get unit load by ID."""
        for ul in self.unit_loads:
            if ul.id == ul_id:
                return ul
        return None
    
    def get_ul_priority(self, ul_id: int) -> float:
        """Get priority for unit load."""
        return self.ul_priority_map.get(ul_id, float('inf'))
class AStarNode:
    """Represents a node in the A* search tree."""
    
    def __init__(self, buffer_state, unit_load_manager: UnitLoadManager, 
                 unit_loads_at_sources: List[UnitLoad], unit_loads_at_sinks: List[UnitLoad],
                 parent=None, move: Optional[MoveInfo] = None, g_cost: float = 0, 
                 current_time: float = 0, tabu_list: Optional[List[int]] = None,
                 retrieval_sequence: Optional[List[int]] = None):
        self.buffer_state = buffer_state
        self.unit_load_manager = unit_load_manager
        self.unit_loads_at_sources = unit_loads_at_sources
        self.unit_loads_at_sinks = unit_loads_at_sinks
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.current_time = current_time
        self.weight = 2
        self.tabu_list = tabu_list if tabu_list is not None else []
        self.retrieval_sequence = retrieval_sequence if retrieval_sequence is not None else []
        self._h_cost: Optional[float] = None
        self._h_cost_base: Optional[float] = None
        self.is_fully_evaluated = False
        
        # Performance caching - computed lazily
        self._stored_unit_loads: Optional[Set] = None
        self._accessible_unit_loads: Optional[Dict] = None
        self._blocking_moves: Optional[List] = None
        self._empty_slots: Optional[List] = None
        self._empty_lanes: Optional[List] = None
        self._state_key: Optional[Tuple] = None
        self._ul_position_map: Optional[Dict] = None

    @property
    def vehicle_pos(self) -> Optional[int]:
        """Get current vehicle position."""
        if self.move:
            return self.move.to_pos
        if self.parent:
            return self.parent.vehicle_pos
        return None

    @property
    def h_cost(self) -> float:
        """Get heuristic cost (lazy evaluation)."""
        if self._h_cost is None:
             raise ValueError("Heuristic cost not calculated")
        return self._h_cost
    
    @h_cost.setter
    def h_cost(self, value: float):
        """Set heuristic cost."""
        self._h_cost = value

    @property
    def f_cost(self) -> float:
        """Get total cost (g + h)."""
        if self._h_cost is not None:
             return self.g_cost + self.weight * self.h_cost
        elif self._h_cost_base is not None:
             # If only base heuristic is available, use it (will be updated later)
             return self.g_cost + self.weight * self._h_cost_base
        return float('inf')

    @property
    def stored_unit_loads(self) -> Set[int]:
        """Get all stored unit loads (cached)."""
        if self._stored_unit_loads is None:
            self._stored_unit_loads = self.buffer_state.get_all_stored_unit_loads()
        return self._stored_unit_loads

    @property
    def accessible_unit_loads(self) -> Dict[int, any]:
        """Get all accessible unit loads (cached)."""
        if self._accessible_unit_loads is None:
            self._accessible_unit_loads = self.buffer_state.get_accessible_unit_loads()
        return self._accessible_unit_loads

    @property
    def blocking_moves(self) -> List[Dict]:
        """Get all blocking moves (cached)."""
        if self._blocking_moves is None:
            self._blocking_moves = self.buffer_state.get_all_blocking_moves(self.unit_load_manager.unit_loads)
        return self._blocking_moves

    @property
    def empty_slots(self) -> List:
        """Get all empty slots (cached)."""
        if self._empty_slots is None:
            self._empty_slots = self.buffer_state.get_all_empty_slots()
        return self._empty_slots

    @property
    def empty_lanes(self) -> List:
        """Get all empty lanes (cached)."""
        if self._empty_lanes is None:
            self._empty_lanes = self.buffer_state.get_all_empty_lanes()
        return self._empty_lanes

    @property
    def ul_position_map(self) -> Dict[int, Tuple]:
        """Get map of ul_id -> (lane, stack_idx) (cached)."""
        if self._ul_position_map is None:
            self._ul_position_map = {}
            for lane in self.buffer_state.virtual_lanes:
                if not lane.is_sink_or_source():
                    for stack_idx, ul_id in enumerate(lane.stacks):
                        if ul_id != 0:
                            self._ul_position_map[ul_id] = (lane, stack_idx)
        return self._ul_position_map

    def get_state_key(self) -> Tuple:
        """Get hashable state key for duplicate detection (cached)."""
        if self._state_key is None:
            buffer_hash = self.buffer_state.get_hashable_state()
            sources_hash = tuple(ul.id for ul in self.unit_loads_at_sources)
            sinks_hash = tuple(ul.id for ul in self.unit_loads_at_sinks)
            self._state_key = (buffer_hash, sources_hash, sinks_hash)
        return self._state_key

    def to_dict(self) -> Dict:
        """Create serializable dictionary representation."""
        stored_uls = [ul for ul in self.unit_load_manager.unit_loads 
                     if ul.is_stored and not ul.is_at_sink]
        
        buffer_lanes_state = self._create_buffer_state_dict()
        serializable_move = self._create_serializable_move()

        return {
            'move': serializable_move,
            'g_cost': self.g_cost,
            'h_cost': self.h_cost,
            'f_cost': self.f_cost,
            'buffer_state': buffer_lanes_state,
            'unit_loads_at_sources': sorted([ul.id for ul in self.unit_loads_at_sources]),
            'unit_loads_at_sinks': sorted([ul.id for ul in self.unit_loads_at_sinks]),
            'stored_unit_loads': sorted([ul.id for ul in stored_uls])
        }
    
    def _create_buffer_state_dict(self) -> List[Dict]:
        """Create buffer state dictionary for serialization."""
        buffer_lanes_state = []
        ul_map = {ul.id: ul for ul in self.unit_load_manager.unit_loads}
        
        # Create sets for efficient lookup
        source_ids = {ul.id for ul in self.unit_loads_at_sources}
        sink_ids = {ul.id for ul in self.unit_loads_at_sinks}
        
        if self.buffer_state.virtual_lanes:
            for lane in self.buffer_state.virtual_lanes:
                stack_with_priority = []
                for ul_id in lane.stacks.tolist():
                    if ul_id == 0:
                        stack_with_priority.append(0)
                    else:
                        unit_load = ul_map.get(ul_id)
                        if unit_load:
                            # Determine priority based on current state
                            if ul_id in source_ids:
                                # At source, use storage priority
                                p = unit_load.storage_priority
                            elif ul_id in sink_ids:
                                # At sink, use retrieval priority
                                p = unit_load.retrieval_priority
                            else:
                                # In buffer, use retrieval priority
                                p = unit_load.retrieval_priority
                            stack_with_priority.append([ul_id, f"P{p}"])
                        else:
                            p = self.unit_load_manager.ul_priority_map.get(ul_id, '?')
                            stack_with_priority.append([ul_id, f"P{p}"])

                buffer_lanes_state.append({
                    'ap_id': lane.ap_id,
                    'stacks': stack_with_priority
                })
        
        return buffer_lanes_state
    
    def _create_serializable_move(self) -> Dict:
        """Create serializable move dictionary."""
        if not self.move:
            return {}
        
        return self.move.to_dict()

    def __lt__(self, other) -> bool:
        """Compare nodes for priority queue (lower f_cost has higher priority)."""
        if self.f_cost == other.f_cost:
            # Prefer deeper nodes (higher g_cost) to break ties
            # This encourages the search to move towards the goal rather than exploring breadth
            return self.g_cost > other.g_cost
        return self.f_cost < other.f_cost


class DistanceCalculator:
    """Handles distance and cost calculations."""
    
    def __init__(self, dist_matrix: np.ndarray, handling_time: float, 
                 initial_buffer_state, verbose: bool = False):
        self.dist_matrix = dist_matrix
        self.handling_time = handling_time
        self.verbose = verbose
        self.source_ap = self._find_source_ap(initial_buffer_state)
        self.sink_ap = self._find_sink_ap(initial_buffer_state)
        self._calculate_average_distances(initial_buffer_state)
        
        # Store n_slots mapping for tier depth calculations
        # Maps ap_id -> number of slots in that lane
        self.lane_n_slots = {}
        for lane in initial_buffer_state.virtual_lanes:
            n_slots = len(lane.stacks) if hasattr(lane, 'stacks') else len(lane.get_tiers())
            self.lane_n_slots[lane.ap_id] = n_slots
    
    def _find_source_ap(self, buffer_state) -> Optional[int]:
        """Find source access point."""
        for lane in buffer_state.virtual_lanes:
            if lane.is_source:
                return lane.ap_id
        if self.verbose:
            print("Warning: Source AP not found in buffer state.")
        return None

    def _find_sink_ap(self, buffer_state) -> Optional[int]:
        """Find sink access point."""
        for lane in buffer_state.virtual_lanes:
            if lane.is_sink:
                return lane.ap_id
        if self.verbose:
            print("Warning: Sink AP not found in buffer state.")
        return None
    
    def _calculate_average_distances(self, buffer_state) -> None:
        """Calculate average distances for heuristic calculations."""
        buffer_lanes = [lane for lane in buffer_state.virtual_lanes 
                       if not lane.is_source and not lane.is_sink]
        
        if not buffer_lanes:
            self.avg_source_to_buffer = 10
            self.avg_buffer_to_sink = 10
            self.avg_reshuffle = 10
            return

        # Source to buffer distances
        source_dists = [self.dist_matrix[self.source_ap][lane.ap_id] for lane in buffer_lanes]
        self.avg_source_to_buffer = np.mean(source_dists) if source_dists else 10

        # Buffer to sink distances
        sink_dists = [self.dist_matrix[lane.ap_id][self.sink_ap] for lane in buffer_lanes]
        self.avg_buffer_to_sink = np.mean(sink_dists) if sink_dists else 10
        
        # Reshuffle distances
        reshuffle_dists = []
        if len(buffer_lanes) > 1:
            for lane1 in buffer_lanes:
                for lane2 in buffer_lanes:
                    if lane1.ap_id != lane2.ap_id:
                        reshuffle_dists.append(self.dist_matrix[lane1.ap_id][lane2.ap_id])
            self.avg_reshuffle = np.mean(reshuffle_dists) if reshuffle_dists else 20
        else:
            self.avg_reshuffle = 20
    
    def calculate_move_cost(self, move_info: MoveInfo) -> float:
        """
        Calculate cost for a specific move.
        
        Tier numbering: Tier 1 = BACK/DEEPEST, Tier N = FRONT/CLOSEST to access point
        For a lane with n_slots:
        - Tier 1 requires (n_slots - 1) moves to reach access point
        - Tier n requires (n_slots - n) moves to reach access point
        
        Total distance = tier_depth_from + lane_distance + tier_depth_to
        where tier_depth = n_slots - tier_number
        """
        # Calculate tier depth cost (distance from tier to access point)
        # Tier 1 = deepest, requires most moves to reach access point
        from_tier_depth = 0
        to_tier_depth = 0
        
        if move_info.from_tier and move_info.from_pos in self.lane_n_slots:
            n_slots_from = self.lane_n_slots[move_info.from_pos]
            from_tier_depth = n_slots_from - move_info.from_tier
        
        if move_info.to_tier and move_info.to_pos in self.lane_n_slots:
            n_slots_to = self.lane_n_slots[move_info.to_pos]
            to_tier_depth = n_slots_to - move_info.to_tier
        
        tier_distance = from_tier_depth + to_tier_depth
        
        if move_info.move_type == MOVE_TYPE_STORE:
            lane_distance = (self.dist_matrix[self.source_ap][move_info.to_pos] + 
                            self.dist_matrix[move_info.to_pos][self.sink_ap])
            travel_time = lane_distance + tier_distance
            return travel_time + 4 * self.handling_time
        
        elif move_info.move_type == MOVE_TYPE_RETRIEVE:
            lane_distance = self.dist_matrix[move_info.from_pos][self.sink_ap]
            travel_time = lane_distance + tier_distance
            return travel_time + 2* self.handling_time
        
        elif move_info.move_type == MOVE_TYPE_RESHUFFLE:
            lane_distance = self.dist_matrix[move_info.from_pos][move_info.to_pos]
            travel_time = lane_distance + tier_distance
            return travel_time + 2 * self.handling_time
        
        return 0.0

class HeuristicCalculator:
    """Calculates heuristic costs for A* search."""
    
    def __init__(self, distance_calc: DistanceCalculator, fleet_size: int = 3, initial_stored_ids: Set[int] = None):
        self.distance_calc = distance_calc
        self.fleet_size = fleet_size
        self.initial_stored_ids = initial_stored_ids if initial_stored_ids is not None else set()
        self._ap_cost_cache = {}
        self._all_retrieval_uls = None
        
        # Precompute AP costs
        source_ap = self.distance_calc.source_ap
        sink_ap = self.distance_calc.sink_ap
        if source_ap is not None and sink_ap is not None:
            for ap_id in range(len(self.distance_calc.dist_matrix)):
                source_to_slot = self.distance_calc.dist_matrix[source_ap][ap_id]
                slot_to_sink = self.distance_calc.dist_matrix[ap_id][sink_ap]
                self._ap_cost_cache[ap_id] = source_to_slot + slot_to_sink
    
    def _calculate_storage_cost(self, source_ids: Set[int], empty_slots: List) -> float:
        """
        Calculate cost for storing unit loads from source to buffer.
        
        Uses a greedy best-match heuristic: estimates the cost by assigning each
        remaining UL to the best (cheapest) available slot. This is admissible because
        it assumes optimal greedy assignment, which is a lower bound on actual cost.
        
        This approach avoids the pitfall of simple averaging, which can incorrectly
        prefer using expensive slots (to remove them from the average) over cheaper ones.
        """
        if len(source_ids) == 0:
            return 0.0
        
        if not empty_slots:
            # Fallback: if no empty slots, use global average (shouldn't happen in valid states)
            avg_dist = self.distance_calc.avg_source_to_buffer + self.distance_calc.avg_buffer_to_sink
            return len(source_ids) * (avg_dist + 4 * self.distance_calc.handling_time)
        
        source_ap = self.distance_calc.source_ap
        sink_ap = self.distance_calc.sink_ap
        
        # Calculate round-trip cost for each available slot
        slot_costs = []
        for slot in empty_slots:
            if slot.ap_id in self._ap_cost_cache:
                slot_costs.append(self._ap_cost_cache[slot.ap_id])
            else:
                source_to_slot = self.distance_calc.dist_matrix[source_ap][slot.ap_id]
                slot_to_sink = self.distance_calc.dist_matrix[slot.ap_id][sink_ap]
                round_trip = source_to_slot + slot_to_sink
                slot_costs.append(round_trip)
        
        # Sort to get best (cheapest) slots first
        slot_costs.sort()
        
        # Greedy assignment: assign each UL to best available slot
        num_uls_to_store = len(source_ids)
        
        if num_uls_to_store <= len(slot_costs):
            # Use the N cheapest slots
            total_cost = sum(slot_costs[:num_uls_to_store])
        else:
            # More ULs than slots - use all slots + fallback for remainder
            total_cost = sum(slot_costs)
            remaining = num_uls_to_store - len(slot_costs)
            avg_dist = self.distance_calc.avg_source_to_buffer + self.distance_calc.avg_buffer_to_sink
            total_cost += remaining * avg_dist
        
        # Add handling time (4x for store operations)
        return total_cost + num_uls_to_store * 4 * self.distance_calc.handling_time
    
    def _find_ul_position(self, ul_id: int, buffer_state, node=None) -> Tuple[Optional[float], int]:
        """
        Find the lane distance and tier depth for a stored unit load.
        
        Returns:
            Tuple of (lane_distance to sink, tier_depth in lane)
            Returns (None, 0) if not found
        """
        # Try to use cached map from node
        if node and hasattr(node, 'ul_position_map'):
            if ul_id in node.ul_position_map:
                lane, stack_idx = node.ul_position_map[ul_id]
                sink_ap = self.distance_calc.sink_ap
                lane_distance = self.distance_calc.dist_matrix[lane.ap_id][sink_ap]
                return (lane_distance, stack_idx)
            return (None, 0)

        sink_ap = self.distance_calc.sink_ap
        
        for lane in buffer_state.virtual_lanes:
            if not lane.is_sink_or_source():
                for stack_idx, stack_item_id in enumerate(lane.stacks):
                    if stack_item_id == ul_id:
                        lane_distance = self.distance_calc.dist_matrix[lane.ap_id][sink_ap]
                        # tier_depth = stack_idx (distance from AP to this position)
                        return (lane_distance, stack_idx)
        
        return (None, 0)
    
    def _calculate_retrieval_cost(self, stored_ul_ids: Set[int], sink_ids: Set[int], 
                                   node_or_buffer_state) -> float:
        """Calculate cost for retrieving stored unit loads to sink."""
        stored_not_retrieved_ids = stored_ul_ids - sink_ids
        total_cost = 0.0
        
        # Handle both node and buffer_state for backward compatibility if needed
        if hasattr(node_or_buffer_state, 'buffer_state'):
            node = node_or_buffer_state
            buffer_state = node.buffer_state
        else:
            node = None
            buffer_state = node_or_buffer_state
        
        for ul_id in stored_not_retrieved_ids:
            lane_distance, tier_depth = self._find_ul_position(ul_id, buffer_state, node)
            
            if lane_distance is not None:
                # Exact cost: lane distance + tier depth + handling time
                total_cost += lane_distance + tier_depth + 2 * self.distance_calc.handling_time
            else:
                # Fallback to average if position not found
                total_cost += self.distance_calc.avg_buffer_to_sink + 2 * self.distance_calc.handling_time
        
        return total_cost
    
    def _calculate_reshuffle_cost_to_lane(self, from_tier_depth: int, from_ap: int, 
                                          to_lane) -> float:
        """
        Calculate the cost to reshuffle a UL from current position to a specific empty lane.
        
        Args:
            from_tier_depth: Tier depth of blocker in current lane (stack_idx)
            from_ap: Access point ID of the source lane
            to_lane: Destination lane (must be empty)
        
        Returns:
            Total reshuffle cost including tier depths and handling time
        """
        to_ap = to_lane.ap_id
        lane_distance = self.distance_calc.dist_matrix[from_ap][to_ap]
        
        # For empty lane, UL is placed at deepest tier (Tier 1)
        # tier_to_depth = n_slots - 1
        to_tier_depth = len(to_lane.stacks) - 1
        
        # Total: tier_from + lane_distance + tier_to + handling
        return from_tier_depth + lane_distance + to_tier_depth + 2 * self.distance_calc.handling_time
    
    def _calculate_blocking_cost(self, blocking_moves: List[Dict], empty_lanes: List) -> float:
        """
        Calculate cost for resolving blocking situations via reshuffles.
        
        Blocking occurs when a lower-priority UL is physically in front of a higher-priority UL,
        OR when ULs with the same priority are in the same lane.
        """
        if len(blocking_moves) == 0:
            return 0.0
        
        # Fleet-size-dependent blocking multiplier: 10 / fleet_size
        BLOCKING_MULTIPLIER = 5.0 / self.fleet_size
        
        total_cost = 0.0
        base_costs = []  # Track base costs (before multiplier) for analysis
        
        for blocking_move in blocking_moves:
            blocker_ul_id = blocking_move['ul_id']
            from_lane = blocking_move['from_lane']
            from_ap = from_lane.ap_id
            
            # Find tier depth of the blocking UL
            tier_depth = 0
            for stack_idx, ul_id in enumerate(from_lane.stacks):
                if ul_id == blocker_ul_id:
                    tier_depth = stack_idx
                    break
            
            if empty_lanes:
                # Find best empty lane (minimum cost)
                best_cost = float('inf')
                for to_lane in empty_lanes:
                    if to_lane.ap_id != from_ap:  # Don't reshuffle to same lane
                        cost = self._calculate_reshuffle_cost_to_lane(tier_depth, from_ap, to_lane)
                        best_cost = min(best_cost, cost)
                
                if best_cost < float('inf'):
                    base_costs.append(best_cost)
                    total_cost += best_cost * BLOCKING_MULTIPLIER
                else:
                    # No valid reshuffling lane found - use average
                    avg_cost = self.distance_calc.avg_reshuffle + 2 * self.distance_calc.handling_time
                    base_cost = avg_cost + tier_depth
                    base_costs.append(base_cost)
                    total_cost += base_cost * BLOCKING_MULTIPLIER
            else:
                # No empty lanes available - blocking is VERY expensive
                # Must wait for retrieval to free up space
                avg_cost = self.distance_calc.avg_reshuffle + 2 * self.distance_calc.handling_time
                base_cost = (avg_cost + tier_depth) * 2.0
                base_costs.append(base_cost)
                total_cost += base_cost * BLOCKING_MULTIPLIER
        
        # Debug output if enabled
        import os
        if os.environ.get('DEBUG_BLOCKING_COSTS') == '1' and base_costs:
            print(f"    [BLOCKING] {len(blocking_moves)} blocks, multiplier={BLOCKING_MULTIPLIER:.2f}x, "
                  f"base_costs: min={min(base_costs):.1f}, max={max(base_costs):.1f}, "
                  f"avg={sum(base_costs)/len(base_costs):.1f}, total={total_cost:.1f}")
        
        return total_cost
    
    def _calculate_priority_violation_penalty(self, node: AStarNode, source_ids: Set[int], 
                                               sink_ids: Set[int]) -> float:
        """
        Calculate penalty for priority violations:
        1. Out-of-order retrievals: comparing actual retrieval sequence against ideal
        2. Premature retrievals: retrieving before higher-priority storage tasks are complete
        
        All priorities are on the same scale (1, 2, 3, ...) where lower = higher priority.
        """
        total_penalty = 0.0
        
        # PART 1: Penalty for out-of-order retrievals
        # Build a complete priority map of ALL ULs that need retrieval (not just retrieved ones)
        # Use cached map if available
        if self._all_retrieval_uls is None:
            self._all_retrieval_uls = {}
            for ul in node.unit_load_manager.unit_loads:
                if ul.retrieval_priority is not None and ul.retrieval_priority < 900:
                    self._all_retrieval_uls[ul.id] = ul.retrieval_priority
        
        all_retrieval_uls = self._all_retrieval_uls
        
        if node.unit_loads_at_sinks and all_retrieval_uls:
            # Get the actual sequence of retrieved ULs (in order they were retrieved)
            # Use cached sequence from node
            actual_sequence = node.retrieval_sequence
            
            # Count inversions in two ways:
            # 1. Between already-retrieved ULs (past inversions)
            # 2. Between retrieved ULs and pending ULs (skipped higher-priority items)
            num_inversions = 0
            
            # Type 1: Inversions between already-retrieved ULs
            if len(actual_sequence) >= 2:
                for i in range(len(actual_sequence)):
                    for j in range(i + 1, len(actual_sequence)):
                        ul_i = actual_sequence[i]
                        ul_j = actual_sequence[j]
                        
                        # Skip if either UL doesn't have a retrieval priority
                        if ul_i not in all_retrieval_uls or ul_j not in all_retrieval_uls:
                            continue
                        
                        priority_i = all_retrieval_uls[ul_i]
                        priority_j = all_retrieval_uls[ul_j]
                        
                        # Inversion if different priorities and ul_i retrieved before ul_j but has lower priority
                        if priority_i != priority_j and priority_i > priority_j:
                            num_inversions += 1
            
            # Type 2: Inversions with pending retrievals (retrieved low-priority before high-priority pending)
            # Get all ULs that still need to be retrieved (not at sink yet)
            retrieved_ids = set(actual_sequence)
            pending_ul_ids = set(all_retrieval_uls.keys()) - retrieved_ids
            
            for retrieved_id in retrieved_ids:
                if retrieved_id not in all_retrieval_uls:
                    continue
                retrieved_priority = all_retrieval_uls[retrieved_id]
                
                # Check if we retrieved this UL while skipping higher-priority pending ones
                for pending_id in pending_ul_ids:
                    if pending_id not in all_retrieval_uls:
                        continue
                    pending_priority = all_retrieval_uls[pending_id]
                    
                    # If retrieved UL has lower priority than a pending UL, that's an inversion
                    if retrieved_priority != pending_priority and retrieved_priority > pending_priority:
                        num_inversions += 1
            
            # Apply penalty per inversion
            INVERSION_PENALTY_FACTOR = 100.0 * self.distance_calc.avg_buffer_to_sink
            total_penalty += num_inversions * INVERSION_PENALTY_FACTOR
        
        # PART 2: Penalty for premature retrievals (retrieving before higher-priority storage)
        # Check if any retrieved UL had lower priority than pending storage tasks at the time
        if node.unit_loads_at_sinks:
            # For each retrieved UL, check if there were higher-priority storage tasks waiting
            num_premature_retrievals = 0
            
            # Get all retrieved UL IDs
            retrieved_ul_ids = {ul.id for ul in node.unit_loads_at_sinks}
            
            # For each retrieved UL, check its retrieval priority
            for retrieved_ul_id in retrieved_ul_ids:
                retrieved_ul = node.unit_load_manager.get_ul_by_id(retrieved_ul_id)
                if not retrieved_ul or not retrieved_ul.retrieval_priority or retrieved_ul.retrieval_priority >= 900:
                    continue
                
                # Check if there are ULs at source with higher-priority storage tasks
                for ul_at_source in node.unit_loads_at_sources:
                    if (hasattr(ul_at_source, 'storage_priority') and
                        ul_at_source.storage_priority is not None and
                        ul_at_source.storage_priority < 900 and
                        ul_at_source.storage_priority < retrieved_ul.retrieval_priority):
                        # This retrieval happened before a higher-priority storage task
                        num_premature_retrievals += 1
                        break  # Count each retrieval only once
            
            # Apply penalty for premature retrievals
            PREMATURE_PENALTY_FACTOR = 20.0 * self.distance_calc.avg_buffer_to_sink
            total_penalty += num_premature_retrievals * PREMATURE_PENALTY_FACTOR
        
        return total_penalty
    
    def calculate_base_h_cost(self, node: AStarNode) -> float:
        """
        Calculate the 'cheap' base components of the heuristic.
        Includes only storage and retrieval distance estimations.
        """
        source_ids = {ul.id for ul in node.unit_loads_at_sources}
        sink_ids = {ul.id for ul in node.unit_loads_at_sinks}
        
        storage_cost = self._calculate_storage_cost(source_ids, node.empty_slots)
        retrieval_cost = self._calculate_retrieval_cost(node.stored_unit_loads, sink_ids, node)
        
        return storage_cost + retrieval_cost

    def calculate_penalty_h_cost(self, node: AStarNode) -> float:
        """
        Calculate the 'expensive' penalty components of the heuristic.
        Includes blocking cost and priority violations.
        """
        source_ids = {ul.id for ul in node.unit_loads_at_sources}
        sink_ids = {ul.id for ul in node.unit_loads_at_sinks}
        
        blocking_cost = self._calculate_blocking_cost(node.blocking_moves, node.empty_lanes)
        priority_penalty = self._calculate_priority_violation_penalty(node, source_ids, sink_ids)
        priority_blocking_penalty = self._calculate_priority_blocking_penalty(node, sink_ids)
        premature_storage_penalty = self._calculate_premature_storage_penalty(node, sink_ids)
        
        return blocking_cost + priority_penalty + priority_blocking_penalty + premature_storage_penalty

    def calculate_h_cost(self, node: AStarNode) -> float:
        """
        Calculate heuristic cost for the node. 
        DEPRECATED: Use calculate_base_h_cost and calculate_penalty_h_cost instead for lazy evaluation.
        """
        return self.calculate_base_h_cost(node) + self.calculate_penalty_h_cost(node)
    
    def _calculate_priority_blocking_penalty(self, node: AStarNode, sink_ids: Set[int]) -> float:
        """
        Calculate penalty for lower-priority ULs blocking higher-priority ones in the buffer.
        
        This penalizes states where a lower-priority UL is in front of (blocking) a higher-priority UL
        in the same lane.
        """
        penalty = 0.0
        BLOCKING_PENALTY_FACTOR = 50.0 * self.distance_calc.avg_buffer_to_sink
        
        # Check each lane in the buffer
        for lane in node.buffer_state.virtual_lanes:
            # Build list of (ul_id, tier, priority) for ULs in this lane
            uls_in_lane = []
            for tier_idx, ul_id in enumerate(lane.stacks):
                if ul_id == 0 or ul_id is None or ul_id in sink_ids:
                    continue
                
                ul = node.unit_load_manager.get_ul_by_id(ul_id)
                if not ul:
                    continue
                
                # Calculate tier number (tier 1 = deepest, higher tier = closer to access)
                tier = len(lane.stacks) - tier_idx
                
                # Get retrieval priority
                priority = getattr(ul, 'retrieval_priority', 999)
                uls_in_lane.append((ul_id, tier, priority))
            
            # Check for blocking: UL with higher tier (closer to front) blocks UL with lower tier (deeper)
            for i, (ul_id_i, tier_i, priority_i) in enumerate(uls_in_lane):
                for ul_id_j, tier_j, priority_j in uls_in_lane[i+1:]:
                    # If ul_i is in front of ul_j (higher tier) but has same or lower priority (higher/equal number)
                    # This includes same-priority blocking (>=) which still requires reshuffling
                    if tier_i > tier_j and priority_i >= priority_j and priority_j < 900:
                        # Same or lower-priority UL is blocking higher/equal-priority UL
                        penalty += BLOCKING_PENALTY_FACTOR
        
        return penalty

    def _calculate_premature_storage_penalty(self, node: AStarNode, sink_ids: Set[int]) -> float:
        """
        Calculate penalty for storing low-priority unit loads when high-priority retrievals are pending.
        
        This penalizes states where we have performed a storage task (moving a UL from source to buffer)
        for a lower-priority item, while a higher-priority item was available for retrieval in the buffer.
        """
        penalty = 0.0
        PREMATURE_STORAGE_PENALTY_FACTOR = 50.0 * self.distance_calc.avg_buffer_to_sink
        
        # Identify newly stored unit loads (present in buffer but not in initial state)
        current_stored_ids = node.stored_unit_loads
        newly_stored_ids = current_stored_ids - self.initial_stored_ids
        
        if not newly_stored_ids:
            return 0.0
            
        # Find the highest priority (lowest value) pending retrieval
        # Pending retrievals are ULs in the buffer (stored) that are not yet at sink
        pending_retrieval_ids = current_stored_ids - sink_ids
        
        if not pending_retrieval_ids:
            return 0.0
            
        min_retrieval_priority = 999
        
        for ul_id in pending_retrieval_ids:
            ul = node.unit_load_manager.get_ul_by_id(ul_id)
            if ul and hasattr(ul, 'retrieval_priority') and ul.retrieval_priority < 900:
                if ul.retrieval_priority < min_retrieval_priority:
                    min_retrieval_priority = ul.retrieval_priority
        
        if min_retrieval_priority >= 900:
            return 0.0
            
        # Check if any newly stored UL has lower priority (higher value) than the best pending retrieval
        for ul_id in newly_stored_ids:
            ul = node.unit_load_manager.get_ul_by_id(ul_id)
            if not ul:
                continue
                
            # Use storage priority because that was the priority when we decided to store it
            storage_priority = getattr(ul, 'storage_priority', 999)
            
            if storage_priority < 900 and storage_priority > min_retrieval_priority:
                # We stored a lower priority item (e.g. 5) while a higher priority item (e.g. 1) was waiting
                penalty += PREMATURE_STORAGE_PENALTY_FACTOR
                
        return penalty

class AStarSolver:
    """Main A* solver for warehouse optimization."""
    
    def __init__(self, initial_buffer_state, all_unit_loads, dist_matrix, handling_time, 
                 instance=None, config: Optional[AStarConfig] = None, task_queue=None,
                 verbose: Optional[bool] = None, time_limit: Optional[float] = None):
        # Handle backward compatibility with old parameter names
        if config is None:
            config = AStarConfig(
                verbose=verbose if verbose is not None else False,
                time_limit=time_limit if time_limit is not None else 300.0
            )
        elif verbose is not None or time_limit is not None:
            # If config is provided but old params are also given, update config
            if verbose is not None:
                config.verbose = verbose
            if time_limit is not None:
                config.time_limit = time_limit
        
        self.config = config
        self.initial_buffer_state = initial_buffer_state
        self.instance = instance
        self.task_queue = task_queue
        self.start_time = None
        self.heuristic_cache = {}  # Cache for heuristic calculations
        
        # Use aggressive configuration for all problem sizes
        self.config.max_reshuffle_branching = 5
        
        # Initialize components
        self.unit_load_manager = UnitLoadManager(
            self._initialize_unit_load_objects(all_unit_loads), 
            task_queue, 
            self.config.verbose
        )
        self.distance_calc = DistanceCalculator(
            dist_matrix, handling_time, initial_buffer_state, self.config.verbose
        )
        
        # Get fleet size from instance, default to 3 if not available
        fleet_size = 3
        if instance and hasattr(instance, 'get_fleet_size'):
            fleet_size = instance.get_fleet_size()
        
        self.fleet_size = fleet_size  # Store for tabu tenure
        
        # Get initial stored IDs for premature storage detection
        initial_stored_ids = initial_buffer_state.get_all_stored_unit_loads()
        self.heuristic_calc = HeuristicCalculator(self.distance_calc, fleet_size, initial_stored_ids)
        self.move_generator = MoveGenerator(self.distance_calc, self.config, self.unit_load_manager)
        
        if self.config.verbose:
            print(f"A* solver initialized with {len(self.unit_load_manager.unit_loads)} unit loads.")
            print(f"  Fleet size: {self.fleet_size} (tabu tenure will be {self.fleet_size - 1})")

    def _initialize_unit_load_objects(self, all_unit_loads) -> List[UnitLoad]:
        """Initialize unit load objects from various input formats."""
        if all_unit_loads and hasattr(next(iter(all_unit_loads)), 'priority'):
            return list(all_unit_loads)

        if self.instance and self.instance.get_unit_loads():
            return self._create_unit_loads_from_instance(all_unit_loads)

        return self._create_mock_unit_loads(all_unit_loads)

    def _create_unit_loads_from_instance(self, all_unit_loads) -> List[UnitLoad]:
        """Create unit loads from instance data."""
        instance_unit_loads = self.instance.get_unit_loads()

        if isinstance(all_unit_loads, set):
            return [ul for ul in instance_unit_loads if ul.id in all_unit_loads]
        else:
            return instance_unit_loads

    def _create_mock_unit_loads(self, all_unit_loads) -> List[UnitLoad]:
        """Create mock unit loads for testing."""
        unit_load_objects = []
        if isinstance(all_unit_loads, set):
            sorted_ids = sorted(list(all_unit_loads))
            for i, ul_id in enumerate(sorted_ids):
                mock_ul = UnitLoad(
                    id=ul_id,
                    retrieval_start=1,
                    retrieval_end=100,
                    priority=i + 1,
                    is_mock=True
                )
                unit_load_objects.append(mock_ul)
        return unit_load_objects

    def solve(self) -> Tuple[Optional[List], Optional[List], Dict]:
        """Solve the warehouse optimization problem using A*."""
        self.start_time = time.time()
        
        start_node = self._create_start_node()
        if not start_node:
            return None, None, self.unit_load_manager.ul_priority_map

        open_set = [start_node]
        closed_set = set()
        nodes_explored = 0

        while open_set:
            if self._should_timeout():
                if self.config.verbose:
                     print("A* search timed out.")
                return None, None, self.unit_load_manager.ul_priority_map
            
            current_node = heapq.heappop(open_set)

            # Lazy Evaluation: Calculate expensive penalties only when node is popped
            if not current_node.is_fully_evaluated:
                penalties = self.heuristic_calc.calculate_penalty_h_cost(current_node)
                current_node.h_cost = current_node._h_cost_base + penalties
                current_node.is_fully_evaluated = True
                
                # Push back to heap with updated cost
                heapq.heappush(open_set, current_node)
                continue

            state_key = current_node.get_state_key()

            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            nodes_explored += 1

            if self._is_goal_state(current_node):
                return self._handle_solution_found(current_node, nodes_explored)

            successors = self._generate_all_successors(current_node)
            for successor in successors:
                if successor.get_state_key() not in closed_set:
                    heapq.heappush(open_set, successor)

        return self._handle_no_solution_found(nodes_explored)
    
    def _should_timeout(self) -> bool:
        """Check if search should timeout."""
        return time.time() - self.start_time > self.config.time_limit
    
    def _handle_solution_found(self, current_node: AStarNode, nodes_explored: int) -> Tuple[List, List, Dict]:
        """Handle when solution is found."""
        solution_time = time.time() - self.start_time
        path, nodes = self._reconstruct_path(current_node)
        
        if self.config.verbose:
            print(f"Found solution with {len(path)} moves (cost: {current_node.g_cost:.2f}) "
                  f"in {solution_time:.2f}s after exploring {nodes_explored} nodes.")
        
        solution_states = [node.to_dict() for node in nodes]
        return path, solution_states, self.unit_load_manager.ul_priority_map
    
    def _handle_no_solution_found(self, nodes_explored: int) -> Tuple[None, None, Dict]:
        """Handle when no solution is found."""
        if self.config.verbose:
            solution_time = time.time() - self.start_time
            print(f"No solution found after {solution_time:.2f}s, exploring {nodes_explored} nodes.")
        return None, None, self.unit_load_manager.ul_priority_map

    def _reconstruct_path(self, node: AStarNode) -> Tuple[List[Dict], List[AStarNode]]:
        """Reconstruct the solution path from goal node."""
        path = []
        nodes = []
        current = node
        
        while current is not None:
            if current.move:
                path.append(current.move.to_dict())
            nodes.append(current)
            current = current.parent
            
        return path[::-1], nodes[::-1]

    def _create_start_node(self) -> Optional[AStarNode]:
        """Create the initial node for A* search."""
        already_stored_ids = self.initial_buffer_state.get_all_stored_unit_loads()
        
        unit_loads_at_sources = []
        
        for ul in self.unit_load_manager.unit_loads:
            ul_copy = ul.copy()
            if ul.id in already_stored_ids:
                ul_copy.is_stored = True
                ul_copy.is_at_sink = False
            else:
                ul_copy.is_stored = False
                ul_copy.is_at_sink = False
                unit_loads_at_sources.append(ul_copy)

        if self.config.verbose:
            stored_ids = [ul.id for ul in self.unit_load_manager.unit_loads if ul.is_stored]
            source_ids = [ul.id for ul in unit_loads_at_sources]
            print(f"Initial state: {len(stored_ids)} stored ULs ({stored_ids}), "
                  f"{len(source_ids)} ULs at source ({source_ids}).")

        start_node = AStarNode(
            buffer_state=self.initial_buffer_state,
            unit_load_manager=self.unit_load_manager,
            unit_loads_at_sources=sorted(unit_loads_at_sources, key=lambda ul: ul.storage_priority),
            unit_loads_at_sinks=[],
            g_cost=0,
            current_time=0,
            tabu_list=[],
            retrieval_sequence=[]
        )
        # Use full heuristic for start node
        base_h = self.heuristic_calc.calculate_base_h_cost(start_node)
        penalty_h = self.heuristic_calc.calculate_penalty_h_cost(start_node)
        start_node.h_cost = base_h + penalty_h
        start_node._h_cost_base = base_h
        start_node.is_fully_evaluated = True
        
        return start_node

    def _is_goal_state(self, node: AStarNode) -> bool:
        """Check if node represents goal state."""
        if len(node.unit_loads_at_sources) > 0:
            return False
        
        # Goal: all unit loads that were initially present should be at the sink
        return len(node.unit_loads_at_sinks) == len(self.unit_load_manager.unit_loads)

    def _generate_all_successors(self, current_node: AStarNode) -> List[AStarNode]:
        """Generate all possible successor nodes."""
        successors = []
        successors.extend(self.move_generator.generate_source_to_buffer_moves(
            current_node, self._create_successor))
        successors.extend(self.move_generator.generate_buffer_to_sink_moves(
            current_node, self._create_successor))
        successors.extend(self.move_generator.generate_reshuffling_moves(
            current_node, self._create_successor))
        
        # Apply aggressive pruning for all problem sizes  
        max_successors = 20
        if len(successors) > max_successors:
            successors.sort(key=lambda x: x.f_cost)
            successors = successors[:max_successors]
            
        return successors

    def _create_successor(self, current_node: AStarNode, move_info: MoveInfo, 
                          new_buffer_state, new_sources: List[UnitLoad], 
                          new_sinks: List[UnitLoad]) -> AStarNode:
        """Create a successor node."""
        # Calculate deadhead travel time (time to reach the start of the task)
        current_pos = current_node.vehicle_pos
        if current_pos is None:
            # Assume vehicle starts at source for the first move
            current_pos = self.distance_calc.source_ap
            
        task_start_pos = None
        if move_info.move_type == MOVE_TYPE_STORE:
            task_start_pos = self.distance_calc.source_ap
        elif move_info.move_type in [MOVE_TYPE_RETRIEVE, MOVE_TYPE_RESHUFFLE]:
            task_start_pos = move_info.from_pos
            
        deadhead_time = 0
        if current_pos != task_start_pos:
            deadhead_time = self.distance_calc.dist_matrix[current_pos][task_start_pos]
            
        arrival_at_task = current_node.current_time + deadhead_time
        
        # Calculate waiting time for arrival windows (only for STORE tasks)
        start_time = arrival_at_task
        if move_info.move_type == MOVE_TYPE_STORE:
            ul = self.unit_load_manager.get_ul_by_id(move_info.ul_id)
            if ul and hasattr(ul, 'arrival_start'):
                start_time = max(arrival_at_task, ul.arrival_start)
        
        # Calculate actual move duration
        move_duration = self.distance_calc.calculate_move_cost(move_info)
        
        # For RETRIEVE tasks, check if we need to wait for retrieval window start
        # The vehicle is occupied until the delivery can be made
        if move_info.move_type == MOVE_TYPE_RETRIEVE:
            ul = self.unit_load_manager.get_ul_by_id(move_info.ul_id)
            arrival_at_sink = start_time + move_duration
            if ul and hasattr(ul, 'retrieval_start'):
                # Vehicle is busy until at least retrieval_start
                new_time = max(arrival_at_sink, ul.retrieval_start)
            else:
                new_time = arrival_at_sink
        else:
            new_time = start_time + move_duration
        
        # Create new tabu list - carry over parent's tabu list and add new tabu positions
        new_tabu_list = list(current_node.tabu_list)  # Copy parent's tabu list
        # TABU_TENURE = self.fleet_size - 1  
        num_lanes = new_buffer_state.get_num_non_source_sink_lanes()
        max_tenure = max(1, int(num_lanes / 2) - 1)
        TABU_TENURE = min(self.fleet_size - 1, max_tenure)

        if move_info:
            tabu_position = None
            
            # For STORE moves: do NOT tabu the destination lane.
            # We want to allow filling the same lane sequentially if it's the best lane.
            if move_info.move_type == MOVE_TYPE_STORE:
                pass
            
            # For RETRIEVE/RESHUFFLE moves: tabu the source lane (from_pos) to avoid immediate reuse
            elif move_info.move_type in [MOVE_TYPE_RETRIEVE, MOVE_TYPE_RESHUFFLE]:
                from_pos = move_info.from_pos
                if from_pos not in [self.distance_calc.source_ap, self.distance_calc.sink_ap]:
                    tabu_position = from_pos
            
            # Add to tabu list if we have a position to tabu
            if tabu_position is not None:
                # Remove if already in list (to update position to front)
                if tabu_position in new_tabu_list:
                    new_tabu_list.remove(tabu_position)
                # Insert at front (most recent)
                new_tabu_list.insert(0, tabu_position)
        
        # Maintain tabu tenure - remove oldest if exceeded
        while len(new_tabu_list) > TABU_TENURE:
            new_tabu_list.pop()

        # Update retrieval sequence
        new_retrieval_sequence = list(current_node.retrieval_sequence)
        if move_info and move_info.move_type == MOVE_TYPE_RETRIEVE:
            new_retrieval_sequence.append(move_info.ul_id)

        successor = AStarNode(
            buffer_state=new_buffer_state,
            unit_load_manager=current_node.unit_load_manager,
            unit_loads_at_sources=new_sources,
            unit_loads_at_sinks=new_sinks,
            parent=current_node,
            move=move_info,
            g_cost=current_node.g_cost + (new_time - current_node.current_time), # Cost is total time elapsed
            current_time=new_time,
            tabu_list=new_tabu_list,
            retrieval_sequence=new_retrieval_sequence
        )
        
        # Use cached heuristic if available
        state_key = successor.get_state_key()
        if state_key in self.heuristic_cache:
            successor.h_cost = self.heuristic_cache[state_key]
            successor.is_fully_evaluated = True 
        else:
             # Lazy Evaluation: Only calculate base cost initially
            base_cost = self.heuristic_calc.calculate_base_h_cost(successor)
            successor._h_cost_base = base_cost
            successor.is_fully_evaluated = False
            # h_cost is not set yet, f_cost will use _h_cost_base temporarily
            
        return successor
        if state_key in self.heuristic_cache:
            successor.h_cost = self.heuristic_cache[state_key]
        else:
            successor.h_cost = self.heuristic_calc.calculate_h_cost(successor)
            self.heuristic_cache[state_key] = successor.h_cost
            
        return successor

class MoveGenerator:
    """Generates possible moves for the A* search."""
    
    def __init__(self, distance_calc: DistanceCalculator, config: AStarConfig, unit_load_manager):
        self.distance_calc = distance_calc
        self.config = config
        self.unit_load_manager = unit_load_manager
    
    def generate_source_to_buffer_moves(self, current_node: AStarNode, 
                                        create_successor_fn) -> List[AStarNode]:
        """Generate moves from source to buffer."""
        successors = []
        if not current_node.unit_loads_at_sources:
            return []

        # Find minimum priority unit loads
        min_priority = min(ul.storage_priority for ul in current_node.unit_loads_at_sources)
        uls_to_store = [ul for ul in current_node.unit_loads_at_sources 
                       if ul.storage_priority == min_priority]
        
        # OPTIMIZATION: Break symmetry by enforcing an order on storage tasks.
        # If we have multiple items to store with the same priority, pick the one with 
        # the highest retrieval priority (lowest value) to store next.
        # This avoids generating N! permutations of storage operations.
        # We use ul.id as a tie-breaker to ensure deterministic behavior.
        uls_to_store.sort(key=lambda ul: (ul.retrieval_priority, ul.id))
        uls_to_store = uls_to_store[:1]

        # if self.config.verbose:
        #     print(f"DEBUG: Generating storage moves. Sources: {len(current_node.unit_loads_at_sources)}, Min Prio: {min_priority}, Candidates: {len(uls_to_store)}")

        sim_buffer_state = current_node.buffer_state.copy()
        # Use cached empty slots
        empty_slots = current_node.empty_slots
        
        # if self.config.verbose:
        #     print(f"DEBUG: Empty slots found: {len(empty_slots)}")
            
        tabu_slots = current_node.tabu_list
        
        # Sort empty slots - this is ONLY for ordering which successors to try first
        # The actual choice is made by f_cost comparison in A* search
        sorted_slots = sorted(empty_slots, 
                            key=lambda slot: (
                                not slot.is_empty(),  # False (empty) sorts before True (has ULs)
                                self.distance_calc.dist_matrix[slot.ap_id][self.distance_calc.sink_ap]
                            ))

        # Pruning: Cluster slots by cost and take representatives
        unique_costs = set()
        filtered_slots = []
        
        for slot in sorted_slots:
            cost = self.distance_calc.dist_matrix[self.distance_calc.source_ap][slot.ap_id]
            # Round cost to group similar slots
            cost_key = round(cost, 1) 
            
            if cost_key not in unique_costs:
                unique_costs.add(cost_key)
                filtered_slots.append(slot)
            
            if len(filtered_slots) >= 3: # Limit to max 3 different storage options
                break
        
        for ul_to_store in uls_to_store:
            # Consider filtered slots
            slots_to_consider = filtered_slots

            for slot in slots_to_consider:
                # Check tabu, but allow if this is a move involving the SAME unit load
                # (aspiration criterion: override tabu if it's for the same UL)
                is_tabu = slot.ap_id in tabu_slots
                if is_tabu:
                    # Check if the last move to this position involved the same UL
                    override_tabu = False
                    temp_node = current_node
                    while temp_node and temp_node.move:
                        if temp_node.move.to_pos == slot.ap_id:
                            # Found the move that made this position tabu
                            if temp_node.move.ul_id == ul_to_store.id:
                                override_tabu = True
                            break
                        temp_node = temp_node.parent
                    
                    if not override_tabu:
                        continue

                move_info, new_buffer_state, new_sources = self._create_store_move(
                    ul_to_store, slot, sim_buffer_state, current_node.unit_loads_at_sources
                )
                
                if move_info:
                    successor = create_successor_fn(
                        current_node, move_info, new_buffer_state, 
                        new_sources, current_node.unit_loads_at_sinks
                    )
                    successors.append(successor)
                elif self.config.verbose:
                     print(f"DEBUG: Failed to create store move for UL {ul_to_store.id} to slot {slot.ap_id}")
            
        return successors
    
    def _create_store_move(self, ul_to_store: UnitLoad, slot, sim_buffer_state, 
                          current_sources: List[UnitLoad]) -> Tuple[Optional[MoveInfo], any, List[UnitLoad]]:
        """Create a store move."""
        temp_buffer_state = sim_buffer_state.copy()
        target_lane_in_copy = next(
            (lane for lane in temp_buffer_state.virtual_lanes if lane.ap_id == slot.ap_id), 
            None
        )
        if not target_lane_in_copy:
            return None, None, []

        target_tier = self._find_target_tier(target_lane_in_copy)
        if target_tier is None:
            return None, None, []

        temp_buffer_state.add_unit_load(ul_to_store, target_lane_in_copy)
        new_sources = [ul for ul in current_sources if ul.id != ul_to_store.id]
        
        # Update unit load state
        ul_to_store_copy = ul_to_store.copy()
        ul_to_store_copy.is_stored = True
        ul_to_store_copy.priority = ul_to_store_copy.retrieval_priority

        move_info = MoveInfo(
            ul_id=ul_to_store.id,
            move_type=MOVE_TYPE_STORE,
            from_pos=self.distance_calc.source_ap,
            to_pos=slot.ap_id,
            to_tier=target_tier
        )
        
        return move_info, temp_buffer_state, new_sources
    
    def _find_target_tier(self, lane) -> Optional[int]:
        """Find the target tier for placement in LIFO order.
        
        TIER 1 IS THE BACK/DEEPEST POSITION!
        
        Tier numbering (USER'S REQUIREMENT):
        - Tier 1 = BACK/deepest = highest index (e.g., index 1 in 2-slot lane)
        - Tier 2 = FRONT/shallowest = lowest index (e.g., index 0 in 2-slot lane)
        
        For a 2-slot lane [index_0, index_1]:
        - First UL goes to index 1 -> This is Tier 1 (BACK/DEEPEST)
        - Second UL goes to index 0 -> This is Tier 2 (FRONT/SHALLOWEST)
        
        Formula: tier = len(stacks) - index
        - index 1: tier = 2 - 1 = 1 (Tier 1, BACK)
        - index 0: tier = 2 - 0 = 2 (Tier 2, FRONT)
        """
        # Find the highest empty index (where add_load will place the UL)
        for i in range(len(lane.stacks) - 1, -1, -1):
            if lane.stacks[i] == 0:
                # Convert index to tier number: Tier 1 is at highest index
                tier = len(lane.stacks) - i
                return tier
        
        # Lane is full
        return None
    
    def generate_buffer_to_sink_moves(self, current_node: AStarNode, 
                                     create_successor_fn) -> List[AStarNode]:
        """Generate moves from buffer to sink, prioritizing correctly when the buffer is full."""
        successors = []
        accessible_uls = current_node.accessible_unit_loads
        if not accessible_uls:
            return []

        storage_possible = len(current_node.empty_slots) > 0
        uls_to_consider = []

        if not storage_possible:
            # Buffer is full. We MUST retrieve to make space.
            # Find the single accessible UL with the highest priority (lowest priority value).
            highest_priority_ul_data = None
            min_priority_value = float('inf')
            
            for ul_id, from_lane in accessible_uls.items():
                ul = current_node.unit_load_manager.get_ul_by_id(ul_id)
                if ul and hasattr(ul, 'retrieval_priority') and ul.retrieval_priority < min_priority_value:
                    min_priority_value = ul.retrieval_priority
                    highest_priority_ul_data = (ul, from_lane)
            
            # Only consider this single best option for retrieval.
            if highest_priority_ul_data:
                uls_to_consider.append(highest_priority_ul_data)
        else:
            # Buffer has space. Consider all accessible retrievals, but penalize out-of-order ones.
            for ul_id, from_lane in accessible_uls.items():
                ul_to_move = current_node.unit_load_manager.get_ul_by_id(ul_id)
                if not ul_to_move:
                    continue
                
                # Don't filter out-of-order retrievals - they will be penalized in move cost
                uls_to_consider.append((ul_to_move, from_lane))

        # Generate successor nodes for the unit loads that were selected for consideration.
        for ul_to_move, from_lane in uls_to_consider:
            move_info, new_buffer_state, new_sinks = self._create_retrieve_move(
                ul_to_move, from_lane, current_node
            )
            
            if move_info:
                successor = create_successor_fn(
                    current_node, move_info, new_buffer_state,
                    current_node.unit_loads_at_sources, new_sinks
                )
                successors.append(successor)
            
        return successors
    
    def _create_retrieve_move(self, ul_to_move: UnitLoad, from_lane, 
                             current_node: AStarNode) -> Tuple[Optional[MoveInfo], any, List[UnitLoad]]:
        """Create a retrieve move."""
        sim_buffer_state = current_node.buffer_state.copy()
        sim_buffer_state.retrieve_ul(ul_to_move.id)
        
        new_sinks = sorted(
            current_node.unit_loads_at_sinks + [ul_to_move.copy()], 
            key=lambda ul: ul.retrieval_priority
        )

        # Update unit load state
        ul_to_move_copy = ul_to_move.copy()
        ul_to_move_copy.is_at_sink = True
        ul_to_move_copy.is_stored = False

        from_tier = self._find_from_tier(from_lane, ul_to_move.id)
        
        move_info = MoveInfo(
            ul_id=ul_to_move.id,
            move_type=MOVE_TYPE_RETRIEVE,
            from_pos=from_lane.ap_id,
            from_tier=from_tier,
            to_pos=self.distance_calc.sink_ap
        )
        
        return move_info, sim_buffer_state, new_sinks
    
    def _find_from_tier(self, lane, ul_id: int) -> Optional[int]:
        """Find the tier from which unit load is retrieved."""
        for i, item in enumerate(lane.stacks):
            if item == ul_id:
                return len(lane.stacks) - i
        return None
    
    def _violates_priority_order(self, ul_to_retrieve: UnitLoad, 
                                current_node: AStarNode) -> bool:
        """
        Check if retrieving this unit load would violate priority order.
        
        A retrieval violates priority order if there's ANY higher-priority task waiting:
        1. Higher-priority storage task at source (lower storage_priority number), OR
        2. Higher-priority retrieval task accessible in buffer (lower retrieval_priority number)
        
        All priorities are on the same scale (1, 2, 3, ...) where lower = higher priority.
        """
        if not ul_to_retrieve.retrieval_priority or ul_to_retrieve.retrieval_priority >= 900:
            return False

        # Check unit loads at source - use STORAGE priority (their next pending task)
        for ul in current_node.unit_loads_at_sources:
            if ul.id == ul_to_retrieve.id:
                continue
            
            # UL at source with higher storage priority (lower number) should be stored first
            # Compare against the retrieval task priority we're trying to execute
            if (hasattr(ul, 'storage_priority') and 
                ul.storage_priority is not None and
                ul.storage_priority < 900 and
                ul.storage_priority < ul_to_retrieve.retrieval_priority):
                # There's a storage task with higher priority than this retrieval task
                return True
        
        # Check ACCESSIBLE stored unit loads - use RETRIEVAL priority (their next pending task)
        # Only check if there's a higher-priority retrieval waiting
        accessible_ul_ids = set(current_node.accessible_unit_loads.keys())
        
        for stored_ul_id in accessible_ul_ids:
            if stored_ul_id == ul_to_retrieve.id:
                continue
                
            stored_ul = current_node.unit_load_manager.get_ul_by_id(stored_ul_id)
            if not stored_ul:
                continue

            if (stored_ul.retrieval_priority is not None and
                stored_ul.retrieval_priority < 900 and
                stored_ul.retrieval_priority < ul_to_retrieve.retrieval_priority):
                # Found a retrieval task with higher priority than this retrieval task
                return True
                
        return False

    
    def generate_reshuffling_moves(self, current_node: AStarNode, 
                                  create_successor_fn) -> List[AStarNode]:
        """Generate reshuffling moves."""
        successors = []
        sim_buffer_state = current_node.buffer_state.copy()
        
        # Use cached blocking moves
        blocking_moves = current_node.blocking_moves
        
        if not blocking_moves:
            return []
        
        # Get both empty lanes AND empty slots
        empty_lanes = current_node.empty_lanes
        empty_slots = current_node.empty_slots
        tabu_slots = current_node.tabu_list
        
        # Combine empty lanes and empty slots for reshuffle targets
        # Empty lanes are preferred, but we also allow reshuffling to empty slots in partially filled lanes
        reshuffle_targets = []
        
        # Add empty lanes first (preferred)
        for lane in empty_lanes:
            reshuffle_targets.append(('empty_lane', lane))
        
        # Add empty slots in partially filled lanes
        for slot in empty_slots:
            # Skip slots in empty lanes (already added above)
            is_empty_lane = False
            for el in empty_lanes:
                if el.ap_id == slot.ap_id:
                    is_empty_lane = True
                    break
            
            if not is_empty_lane:
                # Only add if the lane has at least one empty slot
                reshuffle_targets.append(('empty_slot', slot))
        
        if not reshuffle_targets:
            return []

        # OPTIMIZATION: Prune reshuffle targets
        # Sort targets by distance from blocking lane to target (reshuffle distance)
        # We only really need one "good" reshuffle move per blocker to resolve it.
        # Trying 2-3 targets (e.g. closest empty lane + closest empty slot) is usually sufficient.
        
        limit_per_blocker = 2  # Max targets to try per blocking UL
        
        for move in blocking_moves[:self.config.max_reshuffle_branching]:
            ul_to_move_id = move['ul_id']
            from_lane_ap_id = move['from_lane'].ap_id
            
            ul_to_move = current_node.unit_load_manager.get_ul_by_id(ul_to_move_id)
            if not ul_to_move:
                continue

            # Sort targets by distance from this blocker
            sorted_targets = sorted(reshuffle_targets, 
                                  key=lambda t: self.distance_calc.dist_matrix[from_lane_ap_id][t[1].ap_id])
            
            targets_tried = 0
            for target_type, target in sorted_targets:
                if targets_tried >= limit_per_blocker:
                    break
                    
                target_ap_id = target.ap_id
                
                # Can't reshuffle within the same lane
                if from_lane_ap_id == target_ap_id:
                    continue
                
                # Check tabu, but allow if this is a move involving the SAME unit load
                # (aspiration criterion: override tabu if it's for the same UL)
                is_tabu = target_ap_id in tabu_slots
                if is_tabu:
                    # Check if the last move to this position involved the same UL
                    override_tabu = False
                    temp_node = current_node
                    while temp_node and temp_node.move:
                        if temp_node.move.to_pos == target_ap_id:
                            # Found the move that made this position tabu
                            if temp_node.move.ul_id == ul_to_move_id:
                                override_tabu = True
                            break
                        temp_node = temp_node.parent
                    
                    if not override_tabu:
                        continue

                move_info, new_buffer_state = self._create_reshuffle_move(
                    ul_to_move, move, target, sim_buffer_state
                )
                
                if move_info:
                    successor = create_successor_fn(
                        current_node, move_info, new_buffer_state,
                        current_node.unit_loads_at_sources, 
                        current_node.unit_loads_at_sinks
                    )
                    successors.append(successor)
                    targets_tried += 1
            
        return successors
    
    def _create_reshuffle_move(self, ul_to_move: UnitLoad, move_data: Dict, 
                              empty_lane_template, sim_buffer_state) -> Tuple[Optional[MoveInfo], any]:
        """Create a reshuffle move."""
        from_lane_ap_id = move_data['from_lane'].ap_id
        
        temp_buffer_state = sim_buffer_state.copy()
        from_lane_in_copy = next(
            (lane for lane in temp_buffer_state.virtual_lanes if lane.ap_id == from_lane_ap_id), 
            None
        )
        to_lane_in_copy = next(
            (lane for lane in temp_buffer_state.virtual_lanes if lane.ap_id == empty_lane_template.ap_id), 
            None
        )
        
        if not from_lane_in_copy or not to_lane_in_copy:
            return None, None

        from_tier = self._find_from_tier(from_lane_in_copy, ul_to_move.id)
        to_tier = self._find_target_tier(to_lane_in_copy)
        
        if from_tier is None or to_tier is None:
            return None, None

        temp_buffer_state.move_unit_load(ul_to_move.id, from_lane_in_copy, to_lane_in_copy)
        
        move_info = MoveInfo(
            ul_id=ul_to_move.id,
            move_type=MOVE_TYPE_RESHUFFLE,
            from_pos=from_lane_in_copy.ap_id,
            from_tier=from_tier,
            to_pos=to_lane_in_copy.ap_id,
            to_tier=to_tier
        )
        
        return move_info, temp_buffer_state