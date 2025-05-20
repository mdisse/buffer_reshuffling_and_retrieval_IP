import numpy as np
from src.bay.tier import Tier

class VirtualLane:
    def __init__(self, stacks: np.ndarray=None, ap_id: int=None):
        # 1D array of stacks
        # ordered from the edge to the centre of a bay
        self.stacks = stacks
        # Access point index in the list
        self.ap_id = ap_id
        self.tiers = []

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
        """
        if not self.has_slots():
            raise Exception('The lane has no slots for new loads')
        for i in range(len(self.stacks) - 1, -1, -1):
            if self.stacks[i] == 0:
                new_lane = VirtualLane()
                new_lane.ap_id = self.ap_id
                new_lane.stacks = self.stacks.copy()
                new_lane.stacks[i] = priority
                return new_lane

    def remove_load(self):
        """
        Removes the highest load.
        Returns the updated lane and removed load priority.
        """
        for i in range(len(self.stacks)):
            if self.stacks[i] != 0:
                new_lane = VirtualLane()
                new_lane.ap_id = self.ap_id
                new_lane.stacks = self.stacks.copy()
                new_lane.stacks[i] = 0
                return new_lane, self.stacks[i]
        raise Exception('The lane has no loads')

    def __eq__(self, other):
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