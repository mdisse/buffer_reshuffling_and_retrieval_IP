import numpy as np

from src.bay.access_bay import AccessBay
from src.bay.access_point import AccessPoint
from src.preprocessing.layout_to_bays import layout_to_bays
from src.util.access_util import next_in_direction
from src.util.graph_distance_estimator import edges_to_neighbors
from src.util.graph_distance_estimator import estimate_distances_bfs
from src.convert_to_virtual_lanes.network_flow_model import NetworkFlowModel
from src.bay.virtual_lane import VirtualLane


class Buffer:
    def __init__(self, filename: str, access_directions : dict):
        """
        Generates a Buffer object based on a given layout. 
        Doesn't generate the stacks, leaves the bays empty.
        """
        dictionary = layout_to_bays(filename, access_directions)
        self.bays = dictionary["bays"]
        self.path_nodes = dictionary["path_nodes"]
        self.edges = dictionary["edges"]
        self.length = dictionary["length"]
        self.width = dictionary["width"]
        self.sinks = dictionary["sinks"]
        self.sources = dictionary["sources"]
        self.neighbors = edges_to_neighbors(self.edges)

        self.ap_distance = estimate_distances_bfs(self.unpack_access_points(), self.neighbors)

        self.virtual_lanes = None

        self.all_access_points = []
        ap_id_offset = 0

        if self.has_sources():
            for source in self.sources: 
                self.all_access_points.extend(source.access_points)
                for i in range(len(source.access_points)):
                    source.access_points[i].ap_id = i + ap_id_offset
                ap_id_offset += len(source.access_points)

        for bay in self.bays:
            self.all_access_points.extend(bay.access_points)
            for i in range(len(bay.access_points)):
                bay.access_points[i].ap_id = i + ap_id_offset
            ap_id_offset += len(bay.access_points)

        if self.has_sinks():
            for sink in self.sinks: 
                self.all_access_points.extend(sink.access_points)
                for i in range(len(sink.access_points)):
                    sink.access_points[i].ap_id = i + ap_id_offset
                ap_id_offset += len(sink.access_points)
        
        # Calculate average distance between all storage slots for heuristic optimization
        self.average_slot_distance = self._calculate_average_slot_distance()

    def __str__(self):
        """
        Returns a string representation of the buffer.
        This includes the number of bays, sources, and sinks as well as the unit load positions
        """
        base_info = f"Buffer with {len(self.bays)} bays, {len(self.sources)} sources, and {len(self.sinks)} sinks."
        
        # Add unit load positions if virtual lanes exist
        if self.virtual_lanes is not None:
            ul_info = "\nUnit Load Positions:"
            for i, lane in enumerate(self.virtual_lanes):
                if lane.has_loads():
                    non_zero_loads = [ul for ul in lane.stacks if ul > 0]
                    lane_type = ""
                    if lane.is_source:
                        lane_type = " (Source)"
                    elif lane.is_sink:
                        lane_type = " (Sink)"
                    ul_info += f"\n  Lane {i} (AP {lane.ap_id}){lane_type}: {non_zero_loads}"
            
            if ul_info == "\nUnit Load Positions:":
                ul_info += "\n  No unit loads currently in buffer"
                
            return base_info + ul_info
        
        return base_info
    
    def __do_move(self, move: tuple):
        """
        Moves a load from one virtual lane to another
        """
        try:
            self.virtual_lanes[move[0]], stacks = self.virtual_lanes[move[0]].remove_load()
            self.virtual_lanes[move[1]] = self.virtual_lanes[move[1]].add_load(stacks)
        except Exception as e:
            print(e)
            print("CANNOT MOVE IN WH")

    def update_buffer(self, move):
        self.__do_move(move)
        self.read_lanes(self.virtual_lanes)

    def read_lanes(self, lanes):
        for lane in lanes:
            ap: AccessPoint = self.all_access_points[lane.ap_id]
            bay: AccessBay = ap.bay
            stack = ap.get_stack_yx()
            for i in range(len(lane.stacks) // bay.height):
                start = bay.height * i
                end = start + bay.height
                bay.state[stack] = lane.stacks[start:end]
                stack = next_in_direction(bay, stack, ap.direction)

    def get_ap_from_vl(self, point: int):
        return self.virtual_lanes[point].ap_id

    def estimate_fill_level(self):
        n_loads = 0
        total = 0

        for bay in self.bays:
            n_loads += np.count_nonzero(bay.state)
            total += np.prod(bay.state.shape)

        return n_loads / total

    def unpack_access_points(self):
        """
        Unpacks all access points into a single list of (y,x) tuples preserving the order. 
        If adding things like sinks and sources to the buffer, put them into a list like below
        """
        baysAndSinksAndSources = self.sources + self.bays + self.sinks
        access_points_by_bay = [[ap.get_global_yx() for ap in bay.access_points] for bay in baysAndSinksAndSources]
        all_access_points = []
        for ap_list in access_points_by_bay:
            # Todo: Object needs to be added?
            # ap = AccessPoint() # global_x, global_y, stack_x: int, stack_y: int, direction: str
            all_access_points.extend(ap_list)

        return all_access_points

    def has_sinks(self):
        if len(self.sinks) > 0: 
            return True
        return False
    
    def has_sources(self):
        if len(self.sources) > 0: 
            return True
        return False

    def get_sink(self): 
        return self.sinks[0]

    def get_source(self):
        return self.sources[0]

    def get_bays_state_int(self): 
        return np.asarray([bay.get_state_int() for bay in self.bays])
        
    def get_bays_state_ul(self, unit_loads : list):
        return [bay.get_state_ul(unit_loads) for bay in self.bays]
        
    def get_virtual_lanes(self):
        if self.virtual_lanes is None:
            self._create_virtual_lanes()            
            for lane in self.virtual_lanes:
                lane.create_tiers()
        return self.virtual_lanes
    
    def get_virtual_lane(self, ap_id: int) -> VirtualLane | None:
        """
        Returns the virtual lane corresponding to the given access point ID.
        If no such lane exists, returns None.
        """
        if self.virtual_lanes is None:
            self.get_virtual_lanes()
        
        for lane in self.virtual_lanes:
            if lane.ap_id == ap_id:
                return lane
        return None

    def _create_virtual_lanes(self): 
        """
        Creates a virtual lane for each access point in the buffer
        """
        self.virtual_lanes = []

        if self.has_sources():
            self.virtual_lanes.extend([VirtualLane(np.asarray([0]), self.sources[0].access_points[0].ap_id, is_source=True)])

        for bay in self.bays: 
            # Handle both 2D and 3D bay states
            if len(bay.state.shape) == 2:
                # Convert 2D to 3D by adding height dimension
                bay.state = bay.state.reshape(bay.state.shape[0], bay.state.shape[1], 1)
            
            bay.state = bay.state.transpose((1, 0, 2))      # Transpose the state to get the correct order of the stacks
            nfm = NetworkFlowModel(bay)
            virtual_lanes = nfm.get_virtual_lanes()
            for virtual_lane in virtual_lanes:
                virtual_lane.stacks = np.asarray(virtual_lane.stacks).flatten()
            self.virtual_lanes.extend(virtual_lanes)
            bay.state = bay.state.transpose((1, 0, 2))      # Transpose the state back to the original order of the stacks

        if self.has_sinks():
            self.virtual_lanes.extend([VirtualLane(np.asarray([0]), self.sinks[0].access_points[0].ap_id, is_sink=True)])

    def has_sinks(self):
        return len(self.sinks) > 0
    
    def has_sources(self):
        return len(self.sources) > 0

    def unpack_access_points(self):
        """
        Unpacks all access points into a single list of (y,x) tuples preserving the order. 
        """
        baysAndSinksAndSources = self.sources + self.bays + self.sinks
        access_points_by_bay = [[ap.get_global_yx() for ap in bay.access_points] for bay in baysAndSinksAndSources]
        all_access_points = []
        for ap_list in access_points_by_bay:
            all_access_points.extend(ap_list)
        return all_access_points


    def get_distance_lanes(self, lane1, lane2):
        return self.ap_distance[lane1.ap_id][lane2.ap_id]

    def get_distance_sink(self, lane): 
        return self.ap_distance[lane.ap_id][self.sinks[0].access_points[0].ap_id]

    def get_distance_source(self, lane):
        return self.ap_distance[lane.ap_id][self.sources[0].access_points[0].ap_id]

    def get_distance_source_to_sink(self):
        """Calculate the direct distance from source to sink."""
        if self.has_sources() and self.has_sinks():
            source_ap_id = self.sources[0].access_points[0].ap_id
            sink_ap_id = self.sinks[0].access_points[0].ap_id
            return self.ap_distance[source_ap_id][sink_ap_id]
        return float('inf')  # No direct path if source or sink doesn't exist

    def _find_lane_for_ul(self, ul_id: int):
        """Helper to find the virtual lane containing a specific unit load."""
        if self.virtual_lanes is None: self.get_virtual_lanes()
        for lane in self.virtual_lanes:
            if ul_id in lane.stacks:
                return lane
        return None

    def is_accessible(self, ul_id: int) -> bool:
        """Checks if a unit load is at the front of its virtual lane."""
        lane = self._find_lane_for_ul(ul_id)
        if lane:
            non_zero_indices = np.where(lane.stacks > 0)[0]
            if non_zero_indices.size > 0:
                return lane.stacks[non_zero_indices[-1]] == ul_id
        return False

    def get_number_of_blockers(self, ul_id: int) -> int:
        """Counts how many unit loads are in front of the target UL in its virtual lane."""
        lane = self._find_lane_for_ul(ul_id)
        if lane:
            try:
                item_index = np.where(lane.stacks == ul_id)[0][0]
                return np.count_nonzero(lane.stacks[item_index + 1:])
            except IndexError:
                return 0
        return float('inf')

    def get_priority_blockers(self, ul_id: int, all_unit_loads: list) -> int:
        """
        Counts the total number of unit loads in the same lane that are causing priority blockages.
        A unit load causes a blockage if it has lower priority (higher priority number) and is 
        positioned in front of (higher index than) a higher priority unit load (lower priority number).
        
        Args:
            ul_id: The unit load ID to check (used to find the lane)
            all_unit_loads: List of all UnitLoad objects with priority information
            
        Returns:
            int: Total number of unit loads causing priority blockages in this lane
        """
        lane = self._find_lane_for_ul(ul_id)
        if not lane:
            return 0
            
        # Create priority lookup
        priority_lookup = {ul.id: ul.priority for ul in all_unit_loads}
        
        blocking_items = set()  # Use set to avoid counting same item multiple times
        
        # Go through all positions in the lane and identify blocking items
        for i in range(len(lane.stacks)):
            current_ul_id = lane.stacks[i]
            if current_ul_id <= 0 or current_ul_id not in priority_lookup:
                continue
                
            current_priority = priority_lookup[current_ul_id]
            
            # Check all items behind this position (lower indices)
            for j in range(i):
                behind_ul_id = lane.stacks[j]
                if behind_ul_id <= 0 or behind_ul_id not in priority_lookup:
                    continue
                    
                behind_priority = priority_lookup[behind_ul_id]
                
                # If current item has higher priority number (lower priority) than item behind it,
                # then current item is blocking the higher priority item behind
                if current_priority > behind_priority:
                    blocking_items.add(current_ul_id)
                    
        return len(blocking_items)

    def get_accessible_unit_loads(self) -> dict:
        """
        Returns a dictionary of {ul_id: from_lane_object} for all accessible ULs.
        An accessible UL is at the front of its lane (lowest index with a load).
        """
        if self.virtual_lanes is None: self.get_virtual_lanes()
        accessible_uls = {}
        for lane in self.virtual_lanes:
             if not lane.is_sink_or_source():
                non_zero_indices = np.where(lane.stacks > 0)[0]
                if non_zero_indices.size > 0:
                    # The accessible UL is at the front of the lane (lowest index)
                    accessible_ul_id = int(lane.stacks[non_zero_indices[0]])
                    accessible_uls[accessible_ul_id] = lane
        return accessible_uls

    def get_all_stored_unit_loads(self) -> set:
        """
        Returns a set of all unit load IDs currently stored in the buffer.
        """
        if self.virtual_lanes is None: self.get_virtual_lanes()
        stored_uls = set()
        # Iterate through storage lanes only (not source/sink)
        for lane in self.virtual_lanes:
            if not lane.is_sink_or_source():
                non_zero_indices = np.where(lane.stacks > 0)[0]
                for idx in non_zero_indices:
                    ul_id = int(lane.stacks[idx])
                    stored_uls.add(ul_id)
        return stored_uls

    def get_all_empty_slots(self) -> list:
        """
        Returns a list of VirtualLane objects that have physically valid empty slots.
        A slot is only valid if all deeper tiers (higher indices) are occupied.
        """
        if self.virtual_lanes is None: self.get_virtual_lanes()
        empty_lanes = []
        for lane in self.virtual_lanes:
            if not lane.is_sink_or_source(): # Don't try to store items in the sink/source
                # Check if the lane has any physically valid empty slots
                has_valid_slot = False
                for i in range(len(lane.stacks) - 1, -1, -1):
                    if lane.stacks[i] == 0:
                        # Check if all deeper tiers (higher indices) are occupied
                        all_deeper_occupied = True
                        for j in range(i + 1, len(lane.stacks)):
                            if lane.stacks[j] == 0:
                                all_deeper_occupied = False
                                break
                        
                        if all_deeper_occupied:
                            has_valid_slot = True
                            break
                
                if has_valid_slot:
                    empty_lanes.append(lane)
        return empty_lanes

    def move_unit_load(self, ul_id, from_pos, to_pos):
        """
        Applies a move to the buffer state. Uses ap_id for lane identification.
        from_pos and to_pos can be either ap_id integers or VirtualLane objects.
        """
        # Convert VirtualLane objects to ap_ids for consistent handling
        from_ap_id = from_pos.ap_id if hasattr(from_pos, 'ap_id') else from_pos
        to_ap_id = to_pos.ap_id if hasattr(to_pos, 'ap_id') else to_pos
        
        # Handle direct source-to-sink moves (direct retrieval)
        if from_ap_id == 'source' and to_ap_id == 'sink':
            # Direct retrieval - no buffer state changes needed
            return

        # Handle moves from the source (storage)
        if from_ap_id == 'source':
            # Find the destination lane by ap_id and add the unit load
            for i, lane in enumerate(self.virtual_lanes):
                if lane.ap_id == to_ap_id:
                    self.virtual_lanes[i] = lane.add_load(ul_id)
                    break
            return

        # Handle moves to the sink (retrieval)
        if to_ap_id == 'sink':
            # Find the source lane by ap_id and remove the unit load
            for i, lane in enumerate(self.virtual_lanes):
                if lane.ap_id == from_ap_id:
                    self.virtual_lanes[i], _ = lane.remove_load()
                    break
            return
            
        # Handle internal moves (reshuffling)
        # First, remove the specific unit load from the source lane
        removed_ul = None
        for i, lane in enumerate(self.virtual_lanes):
            if lane.ap_id == from_ap_id:
                self.virtual_lanes[i], removed_ul = lane.remove_specific_load(ul_id)
                break
        
        # Then, add the unit load to the destination lane
        for i, lane in enumerate(self.virtual_lanes):
            if lane.ap_id == to_ap_id:
                self.virtual_lanes[i] = lane.add_load(ul_id)
                break

    def add_unit_load(self, unit_load, position):
        """
        Adds a unit load to a specific position in the buffer.
        'position' can be a VirtualLane object or ap_id.
        """
        if self.virtual_lanes is None:
            self.get_virtual_lanes()
        
        # Convert to ap_id for consistent handling
        target_ap_id = position.ap_id if hasattr(position, 'ap_id') else position
        
        for i, lane in enumerate(self.virtual_lanes):
            if lane.ap_id == target_ap_id:
                self.virtual_lanes[i] = lane.add_load(unit_load.id)
                break

    def retrieve_ul(self, ul_id: int):
        """
        Removes a unit load from its lane, simulating retrieval to the sink.
        """
        lane_to_modify = self._find_lane_for_ul(ul_id)
        if lane_to_modify:
            # Find the lane in the list by ap_id and update it
            for i, lane in enumerate(self.virtual_lanes):
                if lane.ap_id == lane_to_modify.ap_id:
                    self.virtual_lanes[i], _ = lane.remove_specific_load(ul_id)
                    break

    def get_hashable_state(self) -> tuple:
        """Creates a hashable representation from the virtual lanes."""
        if self.virtual_lanes is None: self.get_virtual_lanes()
        return tuple(tuple(lane.stacks) for lane in self.virtual_lanes)
        
    def find_ul_position(self, ul_id: int) -> tuple:
        """Finds the (y, x) coordinates of a given unit load ID."""
        for bay in self.bays:
            coords = np.where(bay.state == ul_id)
            if coords[0].size > 0:
                return (coords[0][0], coords[1][0])
        return None

    def get_ul_pos(self, ul_id: int) -> int | None:
        """
        Finds the position (access point ID) of a given unit load within the buffer.
        """
        lane = self._find_lane_for_ul(ul_id)
        if lane:
            return lane.ap_id
        return None

    def get_all_empty_lanes(self) -> list[VirtualLane]:
        """
        Returns a list of all lanes that are completely empty.
        """
        if self.virtual_lanes is None: self.get_virtual_lanes()
        empty_lanes = []
        for lane in self.virtual_lanes:
            if not lane.is_sink_or_source(): # Don't consider sink/source lanes
                if np.all(lane.stacks == 0):  # Check if all positions in the lane are empty
                    empty_lanes.append(lane)
        return empty_lanes

    def _calculate_average_slot_distance(self) -> float:
        """
        Calculate the average distance between all storage slots in the buffer.
        This loops through all bay access points and calculates pairwise distances.
        
        Returns:
            float: Average distance between storage slots
        """
        if not self.bays:
            return 0.0
        
        # Get all bay access points (excluding sources and sinks)
        bay_access_points = []
        for bay in self.bays:
            bay_access_points.extend(bay.access_points)
        
        if len(bay_access_points) < 2:
            return 0.0  # Need at least 2 points to calculate distance
        
        total_distance = 0.0
        pair_count = 0
        
        # Calculate distance between all pairs of bay access points
        for i, ap1 in enumerate(bay_access_points):
            for j, ap2 in enumerate(bay_access_points):
                if i != j:  # Don't calculate distance from a point to itself
                    distance = self.ap_distance[ap1.ap_id][ap2.ap_id]
                    total_distance += distance
                    pair_count += 1
        
        if pair_count > 0:
            return total_distance / pair_count
        else:
            return 0.0
    
    def get_all_blocking_moves(self, all_unit_loads: list) -> list:
        """
        Identifies all unit loads that are blocking higher-priority unit loads within the same lane.
        A block occurs if a lower-priority UL is physically in front of a higher-priority UL,
        OR if unit loads with the same priority are in the same lane (tight time windows make
        sequential retrievals difficult).

        Args:
            all_unit_loads: A list of all UnitLoad objects to get priority info.

        Returns:
            A list of dictionaries, where each dict represents a blocking move
            with {'ul_id': blocking_ul_id, 'from_lane': lane_object}.
        """
        if self.virtual_lanes is None:
            self.get_virtual_lanes()

        blocking_moves = []
        # Use retrieval priority for blocking detection - this determines order of retrieval
        priority_lookup = {ul.id: ul.retrieval_priority for ul in all_unit_loads}

        for lane in self.virtual_lanes:
            if lane.is_sink_or_source() or lane.is_empty():
                continue

            # Get ULs in the lane, filtering out empty slots (0s)
            # The stacks are ordered from front of lane (index 0) to back.
            # An item at index i is in front of an item at index j if i < j.
            lane_uls = [ul_id for ul_id in lane.stacks if ul_id > 0]
            
            if len(lane_uls) < 2:
                continue # No possible blocking in a lane with 0 or 1 UL

            blocking_ul_ids_in_lane = set()

            for i in range(len(lane_uls)):
                for j in range(i + 1, len(lane_uls)):
                    ul_front_id = lane_uls[i]
                    ul_back_id = lane_uls[j]

                    priority_front = priority_lookup.get(ul_front_id, float('inf'))
                    priority_back = priority_lookup.get(ul_back_id, float('inf'))

                    # Block if front UL has lower priority OR same priority
                    # Same priority is problematic due to tight time windows for sequential retrieval
                    if priority_front >= priority_back:
                        # The unit load at the front (i) has lower or equal priority than the one
                        # at the back (j), so it's a blocking item.
                        blocking_ul_ids_in_lane.add(ul_front_id)
            
            for ul_id in blocking_ul_ids_in_lane:
                blocking_moves.append({'ul_id': ul_id, 'from_lane': lane})

        return blocking_moves

    def copy(self):
        """
        Creates an efficient copy of the buffer with all its current state.
        This is much faster than deep copy as it only copies the essential mutable state.
        """
        # Create new buffer instance
        new_buffer = Buffer.__new__(Buffer)
        
        # Copy immutable attributes (these can be shared)
        new_buffer.bays = self.bays  # AccessBay objects - immutable structure
        new_buffer.path_nodes = self.path_nodes
        new_buffer.edges = self.edges
        new_buffer.length = self.length
        new_buffer.width = self.width
        new_buffer.sinks = self.sinks
        new_buffer.sources = self.sources
        new_buffer.neighbors = self.neighbors
        new_buffer.ap_distance = self.ap_distance
        new_buffer.all_access_points = self.all_access_points
        new_buffer.average_slot_distance = self.average_slot_distance
        
        # Copy the virtual lanes (this is the main mutable state)
        if self.virtual_lanes is not None:
            new_buffer.virtual_lanes = [lane.copy() for lane in self.virtual_lanes]
        else:
            new_buffer.virtual_lanes = None
            
        return new_buffer

