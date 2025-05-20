import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

import numpy as np
from src.bay.access_bay import AccessBay

from src.examples_gen.rand_lane_gen import RandLaneGen

# Generates stacks for bays using randomly generated lanes
# No holes guaranteed
class LanedStackGen:
    def __init__(self, max_priority : int, fill_level : float, seed : int, enforce_fill_lvl = True):
        """
        Creates a generator for filling a warehouse
        
        Arguments:
        max_priority (int): number of priorities
        fill_level (floar): 0.0 to 1.0 warehouse fill level
        seed (int): seed for random generation
        enforce_fill_lvl (bool): enforce that fill level is as close to proposed as possible
        """
        self.max_p = max_priority
        self.fill_level = fill_level
        self.rng = np.random.default_rng(seed)
        self.lanes_gen = RandLaneGen(self.rng)
        self.enforce_fill_lvl = enforce_fill_lvl
    
    def __populate_lane(self, bay : AccessBay, lane : list, k : int):
        """Fills one lane stack by stack by k unit loads"""
        for i in range(len(lane) - 1, -1, -1):
            if k <= 0:
                break
            stack_k = min(bay.height, k)
            point = lane[i]
            stack = np.zeros(bay.height, dtype = int)
            stack[bay.height - stack_k:] = self.rng.integers(1, self.max_p+1, stack_k)
            bay.state[point] = stack
            k -= stack_k
    
    def __generate_priorities(self, bay : AccessBay, lanes : list):
        """Generates stacks for each lane."""
        for lane in lanes:
            n = len(lane) * bay.height
            k = self.rng.binomial(n, self.fill_level)
            self.__populate_lane(bay, lane, k)
    
    def __add_load(self, bay: AccessBay, lanes : list):
        """Adds a random load to a random lane"""
        lane_index = self.rng.integers(0, len(lanes))
        # lane = self.rng.choice(lanes)
        lane = lanes[lane_index]

        for i in range(len(lane) - 1, -1, -1):
            point = lane[i]
            if np.count_nonzero(bay.state[point]) < len(bay.state[point]):
                for j in range(len(bay.state[point]) - 1, -1, -1):
                    if bay.state[point][j] == 0:
                        bay.state[point][j] = self.rng.integers(0, self.max_p+1)
                        return
    
    def __rm_load(self, bay: AccessBay, lanes : list):
        """Removes a random load from a random lane"""
        lane_index = self.rng.integers(0, len(lanes))
        # lane = self.rng.choice(lanes)
        lane = lanes[lane_index]

        for point in lane:
            if np.count_nonzero(bay.state[point]) > 0:
                for j in range(len(bay.state[point])):
                    if bay.state[point][j] != 0:
                        bay.state[point][j] = 0
                        return
                    
    def generate_bays_priorities(self, bays : list, height : int):
        """Generates random lanes and populates them with stacks"""
        n_loads = 0
        total = 0

        lanes_list = []

        for bay in bays:
            bay.height = height
            bay.state = np.zeros((bay.length, bay.width, bay.height), dtype=int)
            lanes, _ = self.lanes_gen.generate_lanes(bay)
            lanes_list.append(lanes)
            self.__generate_priorities(bay, lanes)
            n_loads += np.count_nonzero(bay.state)
            total += np.prod(bay.state.shape)
        
        if self.enforce_fill_lvl:
            
            required_n_loads = int(total * self.fill_level + 0.5)

            while n_loads < required_n_loads:
                bay_index = self.rng.integers(0, len(bays))
                bay = bays[bay_index]
                lanes = lanes_list[bay_index]
                n_loads -= np.count_nonzero(bay.state)
                self.__add_load(bay, lanes)
                n_loads += np.count_nonzero(bay.state)
            while n_loads > required_n_loads:
                bay_index = self.rng.integers(0, len(bays))
                bay = bays[bay_index]
                lanes = lanes_list[bay_index]
                n_loads -= np.count_nonzero(bay.state)
                self.__rm_load(bay, lanes)
                n_loads += np.count_nonzero(bay.state)

        # return unit load objects if available
        return []

if __name__ == '__main__': 
    from src.bay.warehouse import Warehouse
    layout_file = "examples/Size_4x4_Layout_3x3.csv"
    access_directions = {
        "north": True,
        "east": True,
        "south": True,
        "west": True
    }
    wh = Warehouse(layout_file, access_directions)
    max_p = 20
    seed = 1
    height = 1
    fill_level = 0.4

    inst_gen = LanedStackGen(max_priority=max_p, fill_level=fill_level, seed=seed)
    inst_gen.generate_bays_priorities(wh.bays, height=height)