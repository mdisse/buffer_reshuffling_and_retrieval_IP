import numpy as np

from src.bay.access_bay import AccessBay
from src.bay.access_point import AccessPoint
from src.preprocessing.layout_to_bays import layout_to_bays
from src.util.access_util import next_in_direction
from src.util.graph_distance_estimator import edges_to_neighbors
from src.util.graph_distance_estimator import estimate_distances_bfs
from src.convert_to_virtual_lanes.network_flow_model import NetworkFlowModel
from src.bay.virtual_lane import VirtualLane


class Warehouse:
    def __init__(self, filename: str, access_directions : dict):
        """
        Generates a Warehouse object based on a given layout. 
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

    def update_warehouse(self, move):
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
        If adding things like sinks and sources to the warehouse, put them into a list like below
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

    def _create_virtual_lanes(self): 
        """
        Creates a virtual lane for each access point in the warehouse
        """
        self.virtual_lanes = []

        if self.has_sources():
            self.virtual_lanes.extend([VirtualLane(np.asarray([0]), self.sources[0].access_points[0].ap_id)])

        for bay in self.bays: 
            bay.state = bay.state.transpose((1, 0, 2))      # Transpose the state to get the correct order of the stacks
            nfm = NetworkFlowModel(bay)
            virtual_lanes = nfm.get_virtual_lanes()
            for virtual_lane in virtual_lanes:
                virtual_lane.stacks = np.asarray(virtual_lane.stacks).flatten()
            self.virtual_lanes.extend(virtual_lanes)
            bay.state = bay.state.transpose((1, 0, 2))      # Transpose the state back to the original order of the stacks

        if self.has_sinks():
            self.virtual_lanes.extend([VirtualLane(np.asarray([0]), self.sinks[0].access_points[0].ap_id)])


    def get_distance_lanes(self, lane1, lane2):
        return self.ap_distance[lane1.ap_id][lane2.ap_id]

    def get_distance_sink(self, lane): 
        return self.ap_distance[lane.ap_id][self.sinks[0].access_points[0].ap_id]

    def get_distance_source(self, lane):
        return self.ap_distance[lane.ap_id][self.sources[0].access_points[0].ap_id]