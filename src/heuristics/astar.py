import heapq
import time
import copy
import numpy as np
from src.examples_gen.unit_load import UnitLoad

# Constants for move types
MOVE_TYPE_STORE = 'store'
MOVE_TYPE_RETRIEVE = 'retrieve'
MOVE_TYPE_RESHUFFLE = 'reshuffle'

class AStarNode:
    def __init__(self, current_buffer_state, all_initial_unit_loads, unit_loads_at_sources, unit_loads_at_sinks, parent=None, move=None, g_cost=0, h_cost=0, current_time=0, tabu_list=None):
        self.current_buffer_state = current_buffer_state
        self.all_initial_unit_loads = all_initial_unit_loads
        self.unit_loads_at_sources = unit_loads_at_sources
        self.unit_loads_at_sinks = unit_loads_at_sinks
        self.parent = parent
        self.move = move
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.current_time = current_time
        # The tabu_list stores the ap_id of slots that were the destination of the previous move.
        self.tabu_list = tabu_list if tabu_list is not None else []

    @property
    def vehicle_pos(self):
        if self.move:
            return self.move.get('to_pos')
        if self.parent:
            return self.parent.vehicle_pos
        return None

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

    def get_state_key(self):
        buffer_hash = self.current_buffer_state.get_hashable_state()
        sources_hash = tuple(ul.id for ul in self.unit_loads_at_sources)
        sinks_hash = tuple(ul.id for ul in self.unit_loads_at_sinks)
        # The tabu list is part of the state to differentiate nodes that are otherwise identical
        tabu_hash = tuple(self.tabu_list)
        return (buffer_hash, sources_hash, sinks_hash, tabu_hash)

    def to_dict(self, ul_priority_map=None):
        """ Creates a serializable dictionary representation of the node state. """
        stored_uls = [ul for ul in self.all_initial_unit_loads if ul.is_stored and not ul.is_at_sink]
        
        ul_map = {ul.id: ul for ul in self.all_initial_unit_loads}

        buffer_lanes_state = []
        if self.current_buffer_state.virtual_lanes:
            for lane in self.current_buffer_state.virtual_lanes:
                stack_with_priority = []
                for ul_id in lane.stacks.tolist():
                    if ul_id == 0:
                        stack_with_priority.append(0)
                    else:
                        unit_load = ul_map.get(ul_id)
                        if unit_load:
                            if unit_load.is_stored:
                                p = unit_load.retrieval_priority
                            else:
                                p = unit_load.storage_priority
                            stack_with_priority.append([ul_id, f"P{p}"])
                        else:
                            p = ul_priority_map.get(ul_id, '?') if ul_priority_map else '?'
                            stack_with_priority.append([ul_id, f"P{p}"])

                buffer_lanes_state.append({
                    'ap_id': lane.ap_id,
                    'stacks': stack_with_priority
                })
        
        serializable_move = {}
        if self.move:
            serializable_move = self.move.copy()
            for key, value in self.move.items():
                if hasattr(value, 'ap_id'):
                    serializable_move[key] = value.ap_id
                else:
                    serializable_move[key] = value

        return {
            'move': serializable_move,
            'g_cost': self.g_cost,
            'h_cost': self.h_cost,
            'f_cost': self.f_cost,
            'current_time': self.current_time,
            'buffer_state': buffer_lanes_state,
            'unit_loads_at_sources': sorted([ul.id for ul in self.unit_loads_at_sources]),
            'unit_loads_at_sinks': sorted([ul.id for ul in self.unit_loads_at_sinks]),
            'stored_unit_loads': sorted([ul.id for ul in stored_uls])
        }

    def __lt__(self, other):
        # Tie-break on g_cost (travel time) if f_cost is equal
        if self.f_cost == other.f_cost:
            return self.g_cost < other.g_cost
        return self.f_cost < other.f_cost


class AStarSolver:
    def __init__(self, initial_buffer_state, all_unit_loads, dist_matrix, handling_time, instance=None, verbose=False, time_limit=300, task_queue=None):
        self.initial_buffer_state = initial_buffer_state
        self.all_unit_loads = all_unit_loads
        self.dist_matrix = dist_matrix
        self.handling_time = handling_time
        self.instance = instance
        self.verbose = verbose
        self.time_limit = time_limit
        self.task_queue = task_queue
        self.start_time = None
        self.unit_load_objects = self._initialize_unit_load_objects()
        self.ul_priority_map = {ul.id: ul.priority for ul in self.unit_load_objects}
        self.source_ap = self._find_source_ap()
        self.sink_ap = self._find_sink_ap()
        self.vehicle_start_pos = self.sink_ap
        self._average_source_to_buffer_dist = None
        self._average_buffer_to_sink_dist = None
        self._average_reshuffle_dist = None
        self._calculate_average_distances()

    def _calculate_average_distances(self):
        buffer_lanes = [lane for lane in self.initial_buffer_state.virtual_lanes if not lane.is_source and not lane.is_sink]
        if not buffer_lanes:
            self._average_source_to_buffer_dist = 10
            self._average_buffer_to_sink_dist = 10
            self._average_reshuffle_dist = 10
            return

        source_dists = [self.dist_matrix[self.source_ap][lane.ap_id] for lane in buffer_lanes]
        self._average_source_to_buffer_dist = np.mean(source_dists) if source_dists else 10

        sink_dists = [self.dist_matrix[lane.ap_id][self.sink_ap] for lane in buffer_lanes]
        self._average_buffer_to_sink_dist = np.mean(sink_dists) if sink_dists else 10
        
        reshuffle_dists = []
        if len(buffer_lanes) > 1:
            for lane1 in buffer_lanes:
                for lane2 in buffer_lanes:
                    if lane1.ap_id != lane2.ap_id:
                        reshuffle_dists.append(self.dist_matrix[lane1.ap_id][lane2.ap_id])
            self._average_reshuffle_dist = np.mean(reshuffle_dists) if reshuffle_dists else 20
        else:
            self._average_reshuffle_dist = 20

    def _find_source_ap(self):
        for lane in self.initial_buffer_state.virtual_lanes:
            if lane.is_source:
                return lane.ap_id
        if self.verbose:
            print("Warning: Source AP not found in buffer state.")
        return None

    def _find_sink_ap(self):
        for lane in self.initial_buffer_state.virtual_lanes:
            if lane.is_sink:
                return lane.ap_id
        if self.verbose:
            print("Warning: Sink AP not found in buffer state.")
        return None

    def _initialize_unit_load_objects(self):
        if self.all_unit_loads and hasattr(next(iter(self.all_unit_loads)), 'priority'):
            return list(self.all_unit_loads)

        if self.instance and self.instance.get_unit_loads():
            return self._create_unit_loads_from_instance()

        return self._create_mock_unit_loads()

    def _create_unit_loads_from_instance(self):
        instance_unit_loads = self.instance.get_unit_loads()

        if isinstance(self.all_unit_loads, set):
            filtered_uls = [ul for ul in instance_unit_loads if ul.id in self.all_unit_loads]
        else:
            filtered_uls = instance_unit_loads

        # If task_queue is provided, use its priority mapping instead of creating our own
        if self.task_queue:
            return self._apply_task_queue_priorities(filtered_uls)
        else:
            return self._apply_internal_priorities(filtered_uls)

    def _apply_task_queue_priorities(self, filtered_uls):
        """
        Apply priorities from the task queue to unit loads.
        This uses the priority logic from create_task_queue function.
        """
        # Create mappings from unit load IDs to their storage and retrieval priorities
        storage_priority_map = {}
        retrieval_priority_map = {}
        
        for task in self.task_queue:
            if hasattr(task, 'priority'):
                if "_mock" in str(task.id):
                    # Storage task - map to real unit load ID
                    real_ul_id = task.real_ul_id
                    storage_priority_map[real_ul_id] = task.priority
                else:
                    # Retrieval task - use task ID directly
                    retrieval_priority_map[task.id] = task.priority
        
        # Apply priorities to unit loads
        for ul in filtered_uls:
            storage_priority = storage_priority_map.get(ul.id, 999)  # Default to high number if not found
            retrieval_priority = retrieval_priority_map.get(ul.id, 999)
            
            # Set specific storage and retrieval priorities
            ul.set_storage_priority(storage_priority)
            ul.set_retrieval_priority(retrieval_priority)
            
            # Set the initial general .priority attribute based on current state and available priorities
            if ul.is_stored:
                # Already stored, use retrieval priority
                ul.priority = retrieval_priority
            else:
                # At source - use the minimum priority (highest importance) between storage and retrieval
                if storage_priority < 900 and retrieval_priority < 900:
                    ul.priority = min(storage_priority, retrieval_priority)
                elif storage_priority < 900:
                    ul.priority = storage_priority
                elif retrieval_priority < 900:
                    ul.priority = retrieval_priority
                else:
                    ul.priority = 999  # No valid priorities found
        
        if self.verbose:
            print(f"Applied task queue priorities to {len(filtered_uls)} unit loads")
            print("  Storage and Retrieval Priorities:")
            # Sort by unit load ID for consistent output
            sorted_uls = sorted(filtered_uls[:15], key=lambda ul: ul.id)
            for ul in sorted_uls:  
                storage_pri = getattr(ul, 'storage_priority', 'N/A')
                retrieval_pri = getattr(ul, 'retrieval_priority', 'N/A')
                current_pri = ul.priority
                stored_status = "stored" if ul.is_stored else "at_source"
                print(f"  UL {ul.id}: storage_p={storage_pri}, retrieval_p={retrieval_pri}, current_p={current_pri} ({stored_status})")
            if len(filtered_uls) > 15:
                print(f"  ... and {len(filtered_uls) - 15} more")
        
        return filtered_uls

    def _apply_internal_priorities(self, filtered_uls):
        """
        Apply priorities using the original A* internal logic.
        This is a fallback when no task queue is provided.
        """
        # --- NEW PRIORITY LOGIC ---
        # Create a single list of all tasks (storage and retrieval)
        all_tasks = []
        for ul in filtered_uls:
            # Each UL has a retrieval task
            all_tasks.append({'ul_id': ul.id, 'type': 'retrieval', 'due': ul.retrieval_end if ul.retrieval_end is not None else float('inf')})
            # If not stored initially, it also has a storage task
            if not ul.is_stored:
                all_tasks.append({'ul_id': ul.id, 'type': 'storage', 'due': ul.arrival_end if ul.arrival_end is not None else float('inf')})

        # Sort all tasks chronologically by their due date
        all_tasks.sort(key=lambda x: x['due'])

        # Create a mapping from ul_id to the object for easy access
        ul_map = {ul.id: ul for ul in filtered_uls}

        # Assign a single, unique, and monotonically increasing priority value across all tasks
        for i, task in enumerate(all_tasks):
            priority = i + 1
            ul = ul_map.get(task['ul_id'])
            if not ul: continue

            if task['type'] == 'storage':
                ul.set_storage_priority(priority)
            elif task['type'] == 'retrieval':
                ul.set_retrieval_priority(priority)

        # Set the initial general .priority attribute for the move generator
        for ul in filtered_uls:
            if ul.is_stored:
                ul.priority = ul.retrieval_priority
            else:
                # Storage priority is the first task for a new UL
                ul.priority = ul.storage_priority
        
        return filtered_uls

    def _create_mock_unit_loads(self):
        unit_load_objects = []
        if isinstance(self.all_unit_loads, set):
            sorted_ids = sorted(list(self.all_unit_loads))
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

    def solve(self):
        self.start_time = time.time()
        
        if self.verbose:
            print(f"A* solver initialized with {len(self.unit_load_objects)} unit loads.")
            # Priority details are already shown in _apply_task_queue_priorities, so just show initial state

        start_node = self._create_start_node()
        if not start_node:
            return None, None, self.ul_priority_map

        open_set = [start_node]
        closed_set = set()
        nodes_explored = 0

        while open_set:
            if time.time() - self.start_time > self.time_limit:
                if self.verbose:
                    print("A* search timed out.")
                return None, None, self.ul_priority_map

            current_node = heapq.heappop(open_set)
            state_key = current_node.get_state_key()

            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            nodes_explored += 1

            if self._is_goal_state(current_node):
                solution_time = time.time() - self.start_time
                path, nodes = self._reconstruct_path(current_node)
                if self.verbose:
                    print(f"Found solution with {len(path)} moves (cost: {current_node.g_cost:.2f}) "
                          f"in {solution_time:.2f}s after exploring {nodes_explored} nodes.")
                
                solution_states = [node.to_dict(self.ul_priority_map) for node in nodes]
                return path, solution_states, self.ul_priority_map

            successors = self._generate_all_successors(current_node)
            for successor in successors:
                if successor.get_state_key() not in closed_set:
                    heapq.heappush(open_set, successor)

        if self.verbose:
            solution_time = time.time() - self.start_time
            print(f"No solution found after {solution_time:.2f}s, exploring {nodes_explored} nodes.")
        return None, None, self.ul_priority_map

    def _create_start_node(self):
        already_stored_ids = self.initial_buffer_state.get_all_stored_unit_loads()
        
        unit_loads_at_sources = []
        initial_unit_loads = []

        for ul in self.unit_load_objects:
            ul_copy = copy.deepcopy(ul)
            if ul.id in already_stored_ids:
                ul_copy.is_stored = True
                ul_copy.is_at_sink = False
            else:
                ul_copy.is_stored = False
                ul_copy.is_at_sink = False
                unit_loads_at_sources.append(ul_copy)
            initial_unit_loads.append(ul_copy)

        if self.verbose:
            stored_ids = [ul.id for ul in initial_unit_loads if ul.is_stored]
            source_ids = [ul.id for ul in unit_loads_at_sources]
            print(f"Initial state: {len(stored_ids)} stored ULs ({stored_ids}), "
                  f"{len(source_ids)} ULs at source ({source_ids}).")

        start_node = AStarNode(
            current_buffer_state=self.initial_buffer_state,
            all_initial_unit_loads=initial_unit_loads,
            unit_loads_at_sources=sorted(unit_loads_at_sources, key=lambda ul: ul.priority),
            unit_loads_at_sinks=[],
            g_cost=0,
            h_cost=0,
            current_time=0,
            tabu_list=[]
        )
        start_node.h_cost = self._calculate_h_cost(start_node)
        return start_node

    def _is_goal_state(self, node: AStarNode):
        if len(node.unit_loads_at_sources) > 0:
            return False
        
        # The goal is to have all unit loads that were initially present or arrived at the source, at the sink.
        # This correctly handles cases where some ULs start at the source and some in the buffer.
        return len(node.unit_loads_at_sinks) == len(self.unit_load_objects)

    def _calculate_h_cost(self, astar_node: AStarNode) -> float:
        """
        Priority-aware heuristic function:
        - Base cost for remaining work
        - Heavy penalty for priority order violations
        """
        h_cost = 0.0
        
        # Create maps for quick lookups
        source_ids = {ul.id for ul in astar_node.unit_loads_at_sources}
        sink_ids = {ul.id for ul in astar_node.unit_loads_at_sinks}
        
        # Base costs for remaining work
        h_cost += len(source_ids) * 2  # Unit loads at source need storing + retrieving
        
        buffer_state = astar_node.current_buffer_state
        stored_ul_ids = buffer_state.get_all_stored_unit_loads()
        stored_not_retrieved_ids = stored_ul_ids - sink_ids
        h_cost += len(stored_not_retrieved_ids) * 1  # Unit loads in buffer need retrieving
        
        # HEAVY PENALTY for priority order violations
        # Check if any lower priority UL has been retrieved before a higher priority one
        retrieved_priorities = []
        remaining_priorities = []
        
        for ul in astar_node.unit_loads_at_sinks:
            if hasattr(ul, 'retrieval_priority') and ul.retrieval_priority < 900:
                retrieved_priorities.append(ul.retrieval_priority)
        
        for ul in astar_node.unit_loads_at_sources:
            if hasattr(ul, 'retrieval_priority') and ul.retrieval_priority < 900:
                remaining_priorities.append(ul.retrieval_priority)
                
        for ul_id in stored_not_retrieved_ids:
            ul = next((ul for ul in self.unit_load_objects if ul.id == ul_id), None)
            if ul and hasattr(ul, 'retrieval_priority') and ul.retrieval_priority < 900:
                remaining_priorities.append(ul.retrieval_priority)
        
        # Check for violations: if any retrieved priority is higher than any remaining priority
        if retrieved_priorities and remaining_priorities:
            max_retrieved = max(retrieved_priorities)
            min_remaining = min(remaining_priorities)
            if max_retrieved > min_remaining:
                # Severe penalty for processing lower priority before higher priority
                h_cost += 1000 * (max_retrieved - min_remaining)
        
        return h_cost

    def _generate_all_successors(self, current_node: AStarNode):
        successors = []
        successors.extend(self._generate_source_to_buffer_moves(current_node))
        successors.extend(self._generate_buffer_to_sink_moves(current_node))
        successors.extend(self._generate_reshuffling_moves(current_node))
        return successors

    def _create_successor(self, current_node, move, move_cost, new_buffer_state,
                          new_all_loads, new_sources, new_sinks):
        
        new_time = current_node.current_time + move_cost
        
        new_tabu_list = []
        if move:
            # Only make the DESTINATION of a store or reshuffle move tabu.
            if move.get('type') in [MOVE_TYPE_STORE, MOVE_TYPE_RESHUFFLE]:
                if 'to_pos' in move and move['to_pos'] not in [self.source_ap, self.sink_ap]:
                    new_tabu_list.append(move['to_pos'])
            elif move.get('type') == MOVE_TYPE_RETRIEVE:
                # Add the 'from_pos' of a retrieval move to the tabu list.
                if 'from_pos' in move and move['from_pos'] not in [self.source_ap, self.sink_ap]:
                    new_tabu_list.append(move['from_pos'])

        successor = AStarNode(
            current_buffer_state=new_buffer_state,
            all_initial_unit_loads=new_all_loads,
            unit_loads_at_sources=new_sources,
            unit_loads_at_sinks=new_sinks,
            parent=current_node,
            move=move,
            g_cost=current_node.g_cost + move_cost,
            h_cost=0,
            current_time=new_time,
            tabu_list=new_tabu_list
        )
        successor.h_cost = self._calculate_h_cost(successor)
        return successor

    def _generate_source_to_buffer_moves(self, current_node: AStarNode) -> list[AStarNode]:
        successors = []
        if not current_node.unit_loads_at_sources:
            return []

        # Find the minimum priority among all unit loads at the source.
        min_priority = min(ul.priority for ul in current_node.unit_loads_at_sources)
        
        # Get all unit loads that have this minimum priority.
        uls_to_store = [ul for ul in current_node.unit_loads_at_sources if ul.priority == min_priority]

        sim_buffer_state = copy.deepcopy(current_node.current_buffer_state)
        empty_slots = sim_buffer_state.get_all_empty_slots()
        
        tabu_slots = current_node.tabu_list

        # Generate moves for every unit load that shares the top priority.
        for ul_to_store in uls_to_store:
            # CHRONOLOGICAL ORDER ENFORCEMENT: Check task queue chronological order
            move_dict = {'type': MOVE_TYPE_STORE, 'ul_id': ul_to_store.id}
            if self._violates_chronological_order(move_dict, current_node):
                continue  # Skip this UL - it violates chronological order
                
            for slot in empty_slots:
                if slot.ap_id in tabu_slots:
                    continue

                temp_buffer_state = copy.deepcopy(sim_buffer_state)
                target_lane_in_copy = next((lane for lane in temp_buffer_state.virtual_lanes if lane.ap_id == slot.ap_id), None)
                if not target_lane_in_copy: continue

                target_tier = None
                for i in range(len(target_lane_in_copy.stacks) - 1, -1, -1):
                    if target_lane_in_copy.stacks[i] == 0:
                        all_deeper_occupied = all(target_lane_in_copy.stacks[j] != 0 for j in range(i + 1, len(target_lane_in_copy.stacks)))
                        if all_deeper_occupied:
                            target_tier = len(target_lane_in_copy.stacks) - i
                            break

                temp_buffer_state.add_unit_load(ul_to_store, target_lane_in_copy)
                new_sources = [ul for ul in current_node.unit_loads_at_sources if ul.id != ul_to_store.id]
                new_all_loads = [ul for ul in current_node.all_initial_unit_loads if ul.id != ul_to_store.id]
                ul_to_store_copy = copy.deepcopy(ul_to_store)
                ul_to_store_copy.is_stored = True
                ul_to_store_copy.priority = ul_to_store_copy.retrieval_priority
                new_all_loads.append(ul_to_store_copy)
                new_all_loads.sort(key=lambda ul: ul.id)

                last_pos = current_node.vehicle_pos or self.vehicle_start_pos
                travel_time = self.dist_matrix[last_pos][self.source_ap] + self.dist_matrix[self.source_ap][slot.ap_id]
                move_cost = travel_time + self.handling_time
                
                move_info = {'ul_id': ul_to_store.id, 'from_pos': self.source_ap, 'to_pos': slot.ap_id, 'to_tier': target_tier, 'type': MOVE_TYPE_STORE}
                
                successors.append(self._create_successor(current_node, move_info, move_cost, temp_buffer_state, new_all_loads, new_sources, current_node.unit_loads_at_sinks))
            
        return successors

    def _generate_buffer_to_sink_moves(self, current_node: AStarNode):
        successors = []
        accessible_uls = current_node.current_buffer_state.get_accessible_unit_loads()

        if not accessible_uls:
            return []
            
        for ul_id, from_lane in accessible_uls.items():
            ul_to_move = self._get_ul_by_id(ul_id, current_node.all_initial_unit_loads)
            if not ul_to_move: continue

            # PRIORITY ORDER ENFORCEMENT: Cut branches that violate priority order
            # Check if there are higher priority ULs that should be retrieved first
            if self._violates_priority_order(ul_to_move, current_node):
                continue  # Skip this move - it violates priority order
            
            # CHRONOLOGICAL ORDER ENFORCEMENT: Check task queue chronological order
            move_dict = {'type': MOVE_TYPE_RETRIEVE, 'ul_id': ul_id}
            if self._violates_chronological_order(move_dict, current_node):
                continue  # Skip this move - it violates chronological order
            
            sim_buffer_state = copy.deepcopy(current_node.current_buffer_state)
            sim_buffer_state.retrieve_ul(ul_to_move.id)
            new_sinks = sorted(current_node.unit_loads_at_sinks + [copy.deepcopy(ul_to_move)], key=lambda ul: ul.priority)
            new_all_loads = [ul for ul in current_node.all_initial_unit_loads if ul.id != ul_to_move.id]
            ul_to_move_copy = copy.deepcopy(ul_to_move)
            ul_to_move_copy.is_at_sink = True
            ul_to_move_copy.is_stored = False
            new_all_loads.append(ul_to_move_copy)

            last_pos = current_node.vehicle_pos or self.vehicle_start_pos
            travel_time = self.dist_matrix[last_pos][from_lane.ap_id] + self.dist_matrix[from_lane.ap_id][self.sink_ap]
            move_cost = travel_time + self.handling_time
            
            from_tier = None
            for i, item in enumerate(from_lane.stacks):
                if item == ul_to_move.id:
                    from_tier = len(from_lane.stacks) - i
                    break
            
            move_info = {'ul_id': ul_to_move.id, 'from_pos': from_lane.ap_id, 'from_tier': from_tier, 'to_pos': self.sink_ap, 'type': MOVE_TYPE_RETRIEVE}
            successors.append(self._create_successor(current_node, move_info, move_cost, sim_buffer_state, new_all_loads, current_node.unit_loads_at_sources, new_sinks))
            
        return successors

    def _violates_priority_order(self, ul_to_retrieve, current_node: AStarNode) -> bool:
        """
        Check if retrieving this unit load would violate the priority order.
        Returns True if there are higher-priority ULs that should be retrieved first.
        """
        if not ul_to_retrieve.retrieval_priority or ul_to_retrieve.retrieval_priority >= 900:
            return False  # No priority assigned, allow retrieval
        
        # Get currently accessible unit loads
        accessible_uls = current_node.current_buffer_state.get_accessible_unit_loads()
        
        # Check if there are accessible ULs with higher priority (lower priority number)
        for accessible_ul_id in accessible_uls.keys():
            accessible_ul = self._get_ul_by_id(accessible_ul_id, current_node.all_initial_unit_loads)
            if (accessible_ul and 
                accessible_ul.retrieval_priority is not None and 
                accessible_ul.retrieval_priority < 900 and
                accessible_ul.retrieval_priority < ul_to_retrieve.retrieval_priority):
                # There's a higher priority UL that's accessible - this would violate order
                if self.verbose:
                    pass  # Remove debug output
                return True
        
        return False

    def _violates_chronological_order(self, move, current_node: AStarNode):
        """
        Check if this move violates the chronological task order from task queue.
        Returns True if there's an earlier task that should be completed first.
        """
        if not self.task_queue:
            return False
            
        move_type = move.get('type')
        ul_id = move.get('ul_id')
        
        # Find the task for this move
        current_task = None
        for task in self.task_queue:
            if move_type == MOVE_TYPE_STORE and task.task_type == 'STORAGE':
                # For storage, match real_ul_id for mock tasks
                if hasattr(task, 'real_ul_id') and task.real_ul_id == ul_id:
                    current_task = task
                    break
            elif move_type == MOVE_TYPE_RETRIEVE and task.task_type == 'RETRIEVAL':
                # For retrieval, match task.id directly
                if task.id == ul_id:
                    current_task = task
                    break
        
        if not current_task:
            return False
            
        # Check if any earlier task (lower priority number) is still incomplete
        for earlier_task in self.task_queue:
            if earlier_task.priority >= current_task.priority:
                continue  # Not earlier
                
            # Check if this earlier task is still incomplete
            if earlier_task.task_type == 'STORAGE':
                # Storage task - check if UL is still at source
                real_ul_id = getattr(earlier_task, 'real_ul_id', earlier_task.id)
                ul_at_source = any(ul.id == real_ul_id for ul in current_node.unit_loads_at_sources)
                if ul_at_source:
                    return True
                    
            elif earlier_task.task_type == 'RETRIEVAL':
                # Retrieval task - check if UL is not yet at sink
                ul_at_sink = any(ul.id == earlier_task.id for ul in current_node.unit_loads_at_sinks)
                if not ul_at_sink:
                    return True
        
        return False

    def _generate_reshuffling_moves(self, current_node: AStarNode) -> list[AStarNode]:
        successors = []
        sim_buffer_state = copy.deepcopy(current_node.current_buffer_state)
        empty_lanes = sim_buffer_state.get_all_empty_lanes()
        if not empty_lanes:
            return []

        blocking_moves = sim_buffer_state.get_all_blocking_moves(self.unit_load_objects)
        tabu_slots = current_node.tabu_list

        # Limit the number of reshuffling moves to consider from a single state
        # to prevent state space explosion. We prioritize moves that unblock higher-priority ULs.
        # Note: get_all_blocking_moves should ideally return moves sorted by the priority of the UL they unblock.
        # Assuming it's not sorted, we'll just take the first few.
        MAX_RESHUFFLE_BRANCHING = 3 
        
        for move in blocking_moves[:MAX_RESHUFFLE_BRANCHING]:
            ul_to_move_id = move['ul_id']
            from_lane_ap_id = move['from_lane'].ap_id
            
            ul_to_move = self._get_ul_by_id(ul_to_move_id, current_node.all_initial_unit_loads)
            if not ul_to_move: continue

            for empty_lane_template in empty_lanes:
                if from_lane_ap_id == empty_lane_template.ap_id:
                    continue
                
                if empty_lane_template.ap_id in tabu_slots:
                    continue

                temp_buffer_state = copy.deepcopy(sim_buffer_state)
                from_lane_in_copy = next((lane for lane in temp_buffer_state.virtual_lanes if lane.ap_id == from_lane_ap_id), None)
                to_lane_in_copy = next((lane for lane in temp_buffer_state.virtual_lanes if lane.ap_id == empty_lane_template.ap_id), None)
                if not from_lane_in_copy or not to_lane_in_copy: continue

                from_tier = None
                for i, item in enumerate(from_lane_in_copy.stacks):
                    if item == ul_to_move.id:
                        from_tier = len(from_lane_in_copy.stacks) - i
                        break

                to_tier = None
                for i in range(len(to_lane_in_copy.stacks) - 1, -1, -1):
                    if to_lane_in_copy.stacks[i] == 0:
                        all_deeper_occupied = all(to_lane_in_copy.stacks[j] != 0 for j in range(i + 1, len(to_lane_in_copy.stacks)))
                        if all_deeper_occupied:
                            to_tier = len(to_lane_in_copy.stacks) - i
                            break

                temp_buffer_state.move_unit_load(ul_to_move.id, from_lane_in_copy, to_lane_in_copy)
                last_pos = current_node.vehicle_pos or self.vehicle_start_pos
                travel_time = self.dist_matrix[last_pos][from_lane_in_copy.ap_id] + self.dist_matrix[from_lane_in_copy.ap_id][to_lane_in_copy.ap_id]
                move_cost = travel_time + self.handling_time
                
                move_info = {'ul_id': ul_to_move.id, 'from_pos': from_lane_in_copy.ap_id, 'from_tier': from_tier, 'to_pos': to_lane_in_copy.ap_id, 'to_tier': to_tier, 'type': MOVE_TYPE_RESHUFFLE}
                successors.append(self._create_successor(current_node, move_info, move_cost, temp_buffer_state, current_node.all_initial_unit_loads, current_node.unit_loads_at_sources, current_node.unit_loads_at_sinks))

        return successors

    def _reconstruct_path(self, node: AStarNode):
        path = []
        nodes = []
        current = node
        while current is not None:
            if current.move:
                path.append(copy.deepcopy(current.move))
            nodes.append(current)
            current = current.parent
        return path[::-1], nodes[::-1]

    def _get_ul_priority(self, ul_id):
        for ul in self.unit_load_objects:
            if ul.id == ul_id:
                return ul.priority
        return float('inf')

    def _get_ul_by_id(self, ul_id, unit_load_list):
        for ul in unit_load_list:
            if ul.id == ul_id:
                return ul
        return None