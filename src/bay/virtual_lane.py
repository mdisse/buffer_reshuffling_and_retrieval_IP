import numpy as np
from src.bay.tier import Tier

class VirtualLane:
    def __init__(self, stacks: np.ndarray=None, ap_id: int=None, is_source: bool=False, is_sink: bool=False):
        # 1D array of stacks
        # ordered from the edge to the centre of a bay
        self.stacks = stacks
        # Access point index in the list
        self.ap_id = ap_id
        self.tiers = []
        # Flags to identify if this lane is a source or sink
        self.is_source = is_source
        self.is_sink = is_sink

    def __str__(self): 
        return str({
            "stacks": self.stacks, 
            "ap_id": self.ap_id
            })

    def has_slots(self) -> bool:
        return (0 in self.stacks)

    def has_loads(self) -> bool:
        return np.any(self.stacks > 0)

    def add_load(self, priority: int):
        """
        Adds a load to the lane and returns a new lane.
        Items must be stacked physically - can only place in a tier if all deeper tiers are occupied.
        """
        if not self.has_slots():
            raise Exception('The lane has no slots for new loads')
        
        # Find the first empty slot starting from the deepest (highest index) tier
        # But ensure all deeper tiers are occupied for physical feasibility
        for i in range(len(self.stacks) - 1, -1, -1):
            if self.stacks[i] == 0:
                # Check if all deeper tiers (higher indices) are occupied
                all_deeper_occupied = True
                for j in range(i + 1, len(self.stacks)):
                    if self.stacks[j] == 0:
                        all_deeper_occupied = False
                        break
                
                if all_deeper_occupied:
                    new_lane = VirtualLane()
                    new_lane.ap_id = self.ap_id
                    new_lane.is_source = self.is_source
                    new_lane.is_sink = self.is_sink
                    new_lane.stacks = self.stacks.copy()
                    new_lane.stacks[i] = priority
                    return new_lane
        
        # If we get here, no valid placement was found
        raise Exception('No valid placement found - physical stacking constraints violated')

    def remove_load(self):
        """
        Removes the topmost accessible load (LIFO - Last In, First Out).
        Can only remove items that are not blocked by other items in front of them.
        Returns the updated lane and removed load priority.
        """
        # Find the first non-empty tier from the top (lowest index)
        # This represents the item closest to the access point and thus accessible
        for i in range(len(self.stacks)):
            if self.stacks[i] != 0:
                new_lane = VirtualLane()
                new_lane.ap_id = self.ap_id
                new_lane.is_source = self.is_source
                new_lane.is_sink = self.is_sink
                new_lane.stacks = self.stacks.copy()
                removed_load = self.stacks[i]
                new_lane.stacks[i] = 0
                return new_lane, removed_load
        raise Exception('The lane has no loads')

    def remove_specific_load(self, ul_id):
        """
        Removes a specific unit load by ID, regardless of position.
        This is used for reshuffling operations where we need to move a specific unit load.
        Returns the updated lane and the removed load ID.
        """
        for i in range(len(self.stacks)):
            if self.stacks[i] == ul_id:
                new_lane = VirtualLane()
                new_lane.ap_id = self.ap_id
                new_lane.is_source = self.is_source
                new_lane.is_sink = self.is_sink
                new_lane.stacks = self.stacks.copy()
                new_lane.stacks[i] = 0
                return new_lane, ul_id
        raise Exception(f'Unit load {ul_id} not found in lane {self.ap_id}')

    def __eq__(self, other):
        if not isinstance(other, VirtualLane):
            return False
        return self.ap_id == other.ap_id and np.array_equal(self.stacks, other.stacks)

    def to_data_dict(self):
        data = dict()
        data['ap_id'] = self.ap_id
        data['n_slots'] = len(self.stacks)
        return data

    def get_highest_load(self): 
        """
        Returns the number of loads currently in a virtual lane
        """
        return np.max(self.stacks)

    def get_number_of_loads(self): 
        """
        Returns the number of loads currently in a virtual lane
        """
        return np.count_nonzero(self.stacks)

    def get_ap_id(self):
        return self.ap_id

    def create_tiers(self): 
        """
        Creates the tiers for the virtual lane by iterating 
        reverse through the stacks list and creating a tier for each slot and 
        placing the unitloads in the tiers
        """
        for slot in range(len(self.stacks), 0, -1): 
            self.tiers.append(Tier(slot, self.stacks[slot - 1])) 

    def get_tiers(self):
        return self.tiers

    def reverse_tiers(self):
        """Iterates over the list of tier objects and reverses them""" 
        self.stacks = self.stacks[::-1]
        self.tiers = []
        for i in range(len(self.stacks)): 
            id = i+1
            if self.stacks[i] != 0: 
                self.tiers.append(Tier(id, self.stacks[i]))
            else : 
                self.tiers.append(Tier(id, 0))

    def is_sink_or_source(self) -> bool:
        """
        Returns True if this virtual lane is either a source or sink lane.
        """
        return self.is_source or self.is_sink

    def is_empty(self):
        """Checks if the virtual lane has no unit loads."""
        return not self.has_loads()

    def get_tier(self, tier_id: int) -> Tier:
        """
        Returns the tier object for the given tier ID.
        If the tier does not exist, returns None.
        """
        if 0 < tier_id <= len(self.tiers):
            return self.tiers[tier_id - 1]
        return None

    def copy(self):
        """
        Creates an efficient copy of the virtual lane.
        Only copies the mutable state (stacks array).
        """
        new_lane = VirtualLane()
        new_lane.ap_id = self.ap_id
        new_lane.is_source = self.is_source
        new_lane.is_sink = self.is_sink
        new_lane.stacks = self.stacks.copy() if self.stacks is not None else None
        # Don't copy tiers as they can be recreated if needed
        new_lane.tiers = []
        return new_lane