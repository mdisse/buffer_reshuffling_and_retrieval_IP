import numpy as np
import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

from src.bay.access_bay import AccessBay

from src.examples_gen.rand_lane_gen import RandLaneGen

# Generates stacks for bays using randomly generated lanes
# No holes guaranteed
class LanedStackGenUnique:
    def __init__(self, max_priority : int, seed : int):
        """
        Creates a generator for filling a warehouse
        
        Arguments:
        max_priority (int): number of priorities
        seed (int): seed for random generation
        """
        self.max_p = max_priority
        self.rng = np.random.default_rng(seed)
        self.lanes_gen = RandLaneGen(self.rng)

    def __populate_lane(self, bay : AccessBay, lane : list, k : int, priorities : list):
        """Fills one lane stack by stack by k unit loads"""
        for i in range(len(lane) - 1, -1, -1):
            if k <= 0:
                break
            stack_k = min(bay.height, k)
            point = lane[i]
            stack = np.zeros(bay.height, dtype = int)
            stack[bay.height - stack_k:] = priorities[0:stack_k]
            priorities = priorities[stack_k:]
            bay.state[point] = stack
            k -= stack_k
    
    def __generate_stacks(self, bay : AccessBay, lanes : list):
        """Generates stacks for each lane."""
        for lane in lanes:
            n = len(lane) * bay.height
            priorities = [next(self.priorities_generator) for _ in range(n)]
            priorities = [p for p in priorities if p]
            print(priorities)
            k = len(priorities)
            self.__populate_lane(bay, lane, k, priorities)
                
    def __generate_priorities(self):
        priorities = np.arange(1, min(self.slots, self.max_p) + 1, 1)
        priorities = np.append(priorities, np.zeros(self.slots - len(priorities), dtype=int))
        self.rng.shuffle(priorities)
        for priority in priorities:
            yield priority

    def generate_bays_priorities(self, bays : list, height : int):
        """Generates random lanes and populates them with stacks"""

        self.slots = sum([bay.state.size*height for bay in bays])
        if self.max_p > self.slots:
            print("Error: Not enough slots for max_p")
            sys.exit(1)
        self.priorities_generator = self.__generate_priorities()

        lanes_list = []

        for bay in bays:
            bay.height = height
            bay.state = np.zeros((bay.length, bay.width, bay.height), dtype=int)
            lanes, _ = self.lanes_gen.generate_lanes(bay)
            lanes_list.append(lanes)
            self.__generate_stacks(bay, lanes)

        # return unit load objects if available
        return []

if __name__ == '__main__': 
    from src.bay.warehouse import Warehouse
    layout_file = "examples/Size_3x3_Layout_1x1.csv"
    access_directions = {
        "north": True,
        "east": True,
        "south": True,
        "west": True
    }
    wh = Warehouse(layout_file, access_directions)
    max_p = 5
    seed = 1
    height = 1

    inst_gen = LanedStackGenUnique(max_priority=max_p, seed=seed)
    inst_gen.generate_bays_priorities(wh.bays, height=height)

    for bay in wh.bays:
        print(bay)
        print(bay.state)
        print(type(bay.state))