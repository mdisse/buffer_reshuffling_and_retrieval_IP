import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

import numpy as np
from src.examples_gen.unit_load import UnitLoad

from src.bay.access_bay import AccessBay

from src.examples_gen.rand_lane_gen import RandLaneGen


class UnitLoadGenerator: 
    def __init__(self, tw_length: int, fill_level: float, seed: int, rs_max: int = 50, as_max: int = 25, source=True): 
        """
        Creates a generator for unit loads with time windows

        Arguments:
        seed (int): seed for random generation
        fill_level (float): approximate fill level of the warehouse
        tw_length (int): approximate average length of the time windows
        rs_max (int): maximum retrieval start
        as_max (int): maximum arrival start
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.lanes_gen = RandLaneGen(self.rng)
        self.fill_level = fill_level
        self.tw_length = tw_length
        self.rs_max = rs_max    # maximum retrieval start
        self.as_max = as_max    # maximum arrival start
        self.source = source    # you can use this parameter to enforce the static variant

    def __populate_lane(self, bay: AccessBay, lane: list, k: int, unit_loads: list):
        """Fills one lane stack by stack by k unit loads"""
        unit_loads = [ul.id for ul in unit_loads]
        for i in range(len(lane) - 1, -1, -1):
            if k <= 0:
                break
            stack_k = min(bay.height, k)
            point = lane[i]
            stack = np.zeros(bay.height, dtype=int)
            stack[bay.height - stack_k:] = unit_loads[0:stack_k]
            unit_loads = unit_loads[stack_k:]
            bay.state[point] = stack
            k -= stack_k


    def __generate_times(self, source):
        while True:
            retrieval_start = int(self.rng.integers(1, self.rs_max))
            retrieval_end = retrieval_start + int(self.rng.normal(self.tw_length / 2, self.tw_length / 6))
            if source and self.source:
                arrival_start = max(0, int(self.rng.integers(-(self.as_max-self.tw_length), self.as_max)))
                arrival_end = arrival_start + int(self.rng.normal(self.tw_length / 2, self.tw_length / 6))
            else: 
                arrival_start = None
                arrival_end = None
            yield retrieval_start, retrieval_end, arrival_start, arrival_end


    def __generate_unit_loads(self, source):
        n = int(np.round(self.fill_level * self.slots))
        unit_load_ids = list(range(1, n+1))
        unit_loads = []
        while len(unit_loads) < len(unit_load_ids):
            try: 
                unit_loads.append(UnitLoad(unit_load_ids[len(unit_loads)], *next(self.__generate_times(source))))
            except ValueError: 
                pass 
        unit_loads += [None] * (self.slots - n)
        self.rng.shuffle(unit_loads)
        self.unit_loads = unit_loads
        for ul in unit_loads:
            yield ul

    
    def __generate_stacks(self, bay : AccessBay, lanes : list):
        """Generates stacks for each lane using __populate_lane."""

        for lane in lanes:
            n = len(lane) * bay.height
            unit_loads = [next(self.ul_generator) for _ in range(n)]
            unit_loads = [ul for ul in unit_loads if ul]    # nice hack 
            ul_stored = [ul for ul in unit_loads if ul.stored]
            # k = len(unit_loads)
            # self.__populate_lane(bay, lane, k, unit_loads)
            k = len(ul_stored)
            self.__populate_lane(bay, lane, k, ul_stored)


    def generate_bays_ids(self, bays: list, height: int):
        """Generates random lanes and populates them with stacks"""
        
        for bay in bays:
            bay.height = height
            bay.state = np.zeros((bay.length, bay.width, bay.height), dtype=int)
            lanes, _ = self.lanes_gen.generate_lanes(bay)
            self.__generate_stacks(bay, lanes)


    def generate_bays_priorities(self, bays: list, height: int = 1, source=True):
        # return self.prio_mock(bays, height)
        """Generates random lanes and populates them with stacks"""
        self.slots = sum([bay.state.size*height for bay in bays])
        self.ul_generator = self.__generate_unit_loads(source)
        self.unit_loads = []

        for bay in bays:
            bay.height = height
            bay.state = np.zeros((bay.length, bay.width, bay.height), dtype=int)
            lanes, _ = self.lanes_gen.generate_lanes(bay)
            self.__generate_stacks(bay, lanes)
        
        self.unit_loads = [ul for ul in self.unit_loads if ul is not None]

        return self.unit_loads
    
    # def prio_mock(self, bays, height): 
    #     # just for testing
    #     for bay in bays: 
    #         bay.height = height
    #         bay.state = np.array([[[0], [0], [6]], [[0], [0], [5]], [[0], [1], [2]]])
    #         # bay.state = np.array([[[0]], [[1]], [[0]]])
    #     unit_loads = []
    #     unit_loads_dic = [
    #         {"id": 1, "retrieval_start": 30, "retrieval_end": 40, "arrival_start": None, "arrival_end": None},
    #         {"id": 2, "retrieval_start": 2, "retrieval_end": 25, "arrival_start": None, "arrival_end": None},
    #         {"id": 3, "retrieval_start": 30, "retrieval_end": 32, "arrival_start": 10, "arrival_end": 15},
    #         {"id": 4, "retrieval_start": 30, "retrieval_end": 32, "arrival_start": 20, "arrival_end": 25},
    #         # {"id": 4, "retrieval_start": 100, "retrieval_end": 120, "arrival_start": None, "arrival_end": None},
    #         # {"id": 5, "retrieval_start": 20, "retrieval_end": 40, "arrival_start": None, "arrival_end": None},
    #         # {"id": 6, "retrieval_start": 50, "retrieval_end": 100, "arrival_start": None, "arrival_end": None},
    #         # {"id": 7, "retrieval_start": 10, "retrieval_end": 40, "arrival_start": None, "arrival_end": None},
    #         # {"id": 8, "retrieval_start": 120, "retrieval_end": 160, "arrival_start": 70, "arrival_end": 80},
    #         # {"id": 9, "retrieval_start": 140, "retrieval_end": 160, "arrival_start": 25, "arrival_end": 35},
    #         # {"id": 10, "retrieval_start": 110, "retrieval_end": 160, "arrival_start": 100, "arrival_end": 105},
    #     ]
    #     for ul in unit_loads_dic: 
    #         unit_loads.append(UnitLoad(ul["id"], ul["retrieval_start"], ul["retrieval_end"], ul["arrival_start"], ul["arrival_end"]))

    #     return unit_loads


        

if __name__ == '__main__': 
    from src.bay.warehouse import Warehouse
    layout_file = "examples/Size_3x3_Layout_1x1_sink.csv"
    access_directions = {
        "north": True,
        "east": True,
        "south": True,
        "west": True
    }
    wh = Warehouse(layout_file, access_directions)
    height = 1
    tw_length = 10
    fill_level = 0.5
    seed = 42

    inst_gen = UnitLoadGenerator(tw_length=tw_length, fill_level=fill_level, seed=seed)
    unit_loads = inst_gen.generate_bays_priorities(wh.bays, height=height)

    for ul in unit_loads:
        print(ul)

    for bay in wh.bays:
        print(bay)
        print(bay.state)
        print(type(bay.state))