import numpy as np
from src.bay.access_bay import AccessBay
from src.bay.access_point import AccessPoint

from src.util.access_util import next_in_direction

class RandLaneGen:
    """
    Generates random lanes within a bay
    """
    def __init__(self, rng : np.random.Generator):
        self.rng = rng
    
    def generate_lanes(self, bay : AccessBay):
        """
        Randomly generates lanes within a single bay.
        """
        length, width, height = bay.state.shape
        # matrix for storing lane number
        lane_numbers = -np.ones((length, width), dtype = int)
        # list for storing stacks in each lane
        lanes = [list() for ap in bay.access_points]
        # list of directions of each lane (e.g. 'north' means access FROM the North)
        directions = [ap.direction for ap in bay.access_points]
        # list for storing lanes that can be used
        open_lanes = set(range(len(bay.access_points)))

        while np.sum(lane_numbers == -1) > 0:
            perm_lanes = self.rng.permutation(list(open_lanes))
            for lane_num in perm_lanes:
                if len(lanes[lane_num]) == 0:
                    next = bay.access_points[lane_num].get_stack_yx()
                else:
                    next = next_in_direction(bay, lanes[lane_num][-1], directions[lane_num])
                if (next == None) or (lane_numbers[next] != -1):
                    open_lanes.remove(lane_num)
                    continue
                lane_numbers[next] = lane_num
                lanes[lane_num].append(next)
        
        return lanes, lane_numbers

if __name__ == '__main__':
    state = np.zeros((2,2,2))
    aps = [AccessPoint(0,0,0,0,'north'), AccessPoint(0,0,1,0,'north'),
        AccessPoint(0,0,0,0,'west'), AccessPoint(0,0,0,1,'west')]
    bay = AccessBay(0, 0, state, aps)
    gen = RandLaneGen(np.random.default_rng(0))
    lanes, lane_numbers = gen.generate_lanes(bay)
    print(lanes)
    print(lane_numbers)