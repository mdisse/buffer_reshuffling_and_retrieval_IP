import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

from src.bay.buffer import Buffer
from src.instance.instance_loader import InstanceLoader
from src.examples_gen.unit_load import UnitLoad
from src.examples_gen.unit_load_gen import UnitLoadGenerator
from src.autonomous_mobile_robot.amr_definition import AutonomousMobileRobot
from src.instance.instance_hasher import hash_instance, collect_hashes
import numpy as np
import json

class Instance(): 
    """
    Instance class object that handles all objects related to an instance and can 
    be created by either passing an instanceLoader object, which requires an instance 
    json file as input or by manually setting all the variables but instanceLoader.
    
    This class is able to describe instances that work with a max priority class and 
    with unit loads that have to be retrieved within a time window.
    To differentiate between the two, the max_p variable is used. If max_p is set to 0, 
    the instance is assumed to be a BRR instance with time windows,
    otherwise it is assumed to be a CP instance with a max priority class.
    """
    def __init__(self, 
                 instanceLoader: InstanceLoader=None, 
                 layout_file: str=None, 
                 fill_level: float=None, 
                 max_p: int=None,
                 height: int=1,
                 seed: int=None,
                 access_directions: dict=None,
                 fleet_size: int=1,
                 vehicle_speed: float=1,
                 handling_time: int=1,
                 exampleGenerator = None,
                 rs_max=None, 
                 as_max=None,
                 time_window_length=None,
                 ): 
        # try: 
        if instanceLoader is not None: 
            self.layout_file = instanceLoader.get_layout_filename()
            self.fill_level = instanceLoader.get_fill_level()
            self.max_p = instanceLoader.get_max_p()
            self.height = instanceLoader.get_height()
            self.seed = instanceLoader.get_seed()
            self.access_directions = instanceLoader.get_access_directions()
            self.sink = instanceLoader.get_sink()
            self.source = instanceLoader.get_source()
            self.fleet_size = instanceLoader.get_fleet_size()
            self.vehicle_speed = instanceLoader.get_vehicle_speed()
            self.handling_time = instanceLoader.get_handling_time()
            self._build_buffer()
            self.unit_loads = []
            self._populate_slots(instanceLoader=instanceLoader)
            self.rs_max = instanceLoader.get_rs_max()
            self.as_max = instanceLoader.get_as_max()
            self.time_window_length = instanceLoader.get_time_window_length()
        else: 
            self.layout_file = layout_file 
            self.fill_level = fill_level
            self.max_p = max_p
            self.height = height
            self.seed = seed
            self.access_directions = access_directions
            self._build_buffer()
            self.sink = self.wh_initial.has_sinks()
            self.source = self.wh_initial.has_sources()
            self.fleet_size = fleet_size
            self.vehicle_speed = vehicle_speed
            self.handling_time = handling_time
            self.rs_max = rs_max
            self.as_max = as_max
            self.time_window_length = time_window_length
            self.unit_loads = [] # stays empty when working with priorities
            self._populate_slots(exampleGenerator=exampleGenerator)
        if self.wh_initial.has_sinks(): 
            self._check_feasibility_for_sinks()
        if self.wh_initial.has_sources():
            self._check_feasibility_for_sources()
        if len(self.unit_loads) > 0:
            self._check_feasibility_for_unit_loads()


        # except TypeError as e: 
        #     print("Error: Make sure that you fully describe the instance by either using the instanceLoader \n or setting all of the other parameters when creating an object of the instance class.")
        #     print(e)
        #     sys.exit(1)

    def __str__(self): 
        return str({
            "layout_file": self.layout_file,
            "access_directions": self.access_directions, 
            "sink": self.sink,
            "source": self.source,
            "seed": self.seed, 
            "height": self.height, 
            "max_p": self.max_p, 
            "rs_max": self.rs_max,
            "fill_level": self.fill_level, 
            "unit_loads": len(self.unit_loads),
            "fleet_size": self.fleet_size,
            "vehicle_speed": self.vehicle_speed,
            "handling_time": self.handling_time,
        })

    
    def _build_buffer(self):
        self.wh_initial = Buffer(self.layout_file, self.access_directions) 
        self.wh_reshuffled = Buffer(self.layout_file, self.access_directions) 

    def _populate_slots(self, instanceLoader: InstanceLoader=None, exampleGenerator=None): 
        if instanceLoader is None and exampleGenerator is None: 
            print("Error: Buffer could not be populated. Either pass an instanceLoader or \n exampleGenerator to the Instance Constructor.")
            sys.exit(1)
        if instanceLoader is not None: 
            if instanceLoader.get_unit_loads(): 
                self.unit_loads = self._create_unit_loads(instanceLoader.get_unit_loads())
            else: 
                self.unit_loads = []
            instanceLoader.generate_bays_priorities(self.wh_initial.bays, height=self.height, source=self.source)
        elif exampleGenerator is not None: 
            self.unit_loads = exampleGenerator.generate_bays_priorities(self.wh_initial.bays, height=self.height, source=self.source)
        self.fill_level = self.wh_initial.estimate_fill_level()

    def _create_unit_loads(self, unit_loads):
        """
        Creates unit loads from the unit load dictionary in the instance file
        """
        unit_loads_list = []
        for ul in unit_loads: 
            unit_loads_list.append(UnitLoad(id=ul["id"], retrieval_start=ul["retrieval_start"], retrieval_end=ul["retrieval_end"], arrival_start=ul["arrival_start"], arrival_end=ul["arrival_end"]))
        return unit_loads_list

    def _check_feasibility_for_sinks(self): 
        """
        Searches all bays if atleast 1 item per priority class is present - this is required 
        for the Block Relocation Problem aka the problem with a sink
        """
        items = []
        for bay in self.wh_initial.bays: 
            items.append(bay.state.ravel())
        for priority in range(1, self.max_p +1): 
            exists = False
            for array in items:
                if np.any(array == priority):
                    exists = True
                    break
            if not exists:
                print(f"This instance is not feasible for the block relocation problem, as there is no item of priority {priority}:")
                print(self)
                sys.exit(1)

    def _check_feasibility_for_sources(self):
        if not self.wh_initial.has_sinks():
            print(f"A layout cannot contain a source without a corresponding sink:")
            print(self)
            sys.exit(1)

    def _check_feasibility_for_unit_loads(self):
        """
        Checks if the unit load ids are unique
        """
        ids = []
        for ul in self.unit_loads: 
            ids.append(ul.id)
        if len(ids) != len(set(ids)):
            print(f"Error: The unit load ids are not unique:")
            print(self)
            sys.exit(1)

    def get_access_directions(self):
        return self.access_directions
    
    def get_layout_file(self): 
        return self.layout_file

    def get_filename(self):
        return self.layout_file.split("/")[-1].split(".")[0]

    def get_fill_level(self): 
        return round(self.fill_level, 2)

    def get_max_p(self):
        return self.max_p
    
    def get_height(self):
        return self.height
    
    def get_seed(self):
        return self.seed
    
    def has_sink(self):
        return self.sink
    
    def has_source(self):
        return self.source

    def get_buffer(self): 
        return self.wh_initial
    
    def get_vehicles(self): 
        vehicles = [AutonomousMobileRobot(id+1) for id in range(self.fleet_size)]
        return vehicles

    def get_fleet_size(self):
        return self.fleet_size

    def get_vehicle_speed(self):
        return self.vehicle_speed

    def get_handling_time(self):
        return self.handling_time

    def get_rs_max(self):
        return self.rs_max
    
    def get_as_max(self):
        return self.as_max
    
    def get_time_window_length(self):
        return self.time_window_length

    def get_unit_loads(self):
        return self.unit_loads

    def set_unit_loads(self, unit_loads):
        """
        Updates the list of unit loads for this instance.
        This is used to inject prioritized unit loads before solving.
        """
        self.unit_loads = unit_loads

    def get_hash(self, path):
        return hash_instance(path)

    def check_if_solved(self, hash_path, instance_path): 
        """
        Checks if the instance has already been solved by checking if the instance hash is in the feasible or infeasible file.
        """
        hash = self.get_hash(instance_path)
        hashes = collect_hashes(hash_path)
        if hash in hashes: 
            return True
        return False
    
    def save_hash(self, result, hash_path, instance_path):
        if result == 'feasible':
            with open(f"{hash_path}/feasible.txt", 'a') as f:
                f.write(self.get_hash(instance_path) + '\n')
        elif result == 'infeasible':
            with open(f"{hash_path}/infeasible.txt", 'a') as f:
                f.write(self.get_hash(instance_path) + '\n')

    def save_instance(self, filename): 
        """
        Saves the instance as a json file.
        """
        data = dict()
        data['layout_file'] = self.get_layout_file()
        data['fill_level'] = self.get_fill_level()
        data['max_priority'] = self.get_max_p()
        data['height'] = self.get_height()
        data['seed'] = self.get_seed()
        data['fleet_size'] = self.fleet_size
        data['vehicle_speed'] = self.vehicle_speed
        data['handling_time'] = self.handling_time
        data['rs_max'] = self.rs_max
        data['as_max'] = self.as_max
        data['time_window_length'] = self.time_window_length
        data['sink'] = self.has_sink()
        data['source'] = self.has_source()

        data['bay_info'] = dict()
        data['sink_info'] = dict()
        data['source_info'] = dict()
        data['initial_state'] = dict()
        for bay in self.wh_initial.bays:
            data['bay_info'][bay.get_id()] = bay.to_data_dict()
            data['initial_state'][bay.get_id()] = bay.state.tolist()
    
        for sink in self.wh_initial.sinks: 
            data['sink_info'][sink.get_id()] = sink.to_data_dict()

        for source in self.wh_initial.sources: 
            data['source_info'][source.get_id()] = source.to_data_dict()

        data['access_points'] = []
        for point in self.wh_initial.all_access_points:
            data['access_points'].append(point.to_data_dict())

        data['unit_loads'] = []
        if self.max_p == 0: 
            for ul in self.unit_loads:
                data['unit_loads'].append(ul.to_data_dict()) 

        f = open(filename, 'w')
        json.dump(data, f, indent=4)
        f.close()

    def calculate_distance(self, lane1, tier1, lane2, tier2): 
        """
        Calculates the distance between two lanes
        Each tier moved adds 1 to the distance
        """
        if isinstance(lane1, str):
            if lane1 == "sink":
                if isinstance(lane2, str) and lane2 == "source":
                    lane_distance = self.get_buffer().get_distance_source_to_sink()
                else:
                    # Distance from sink to lane2
                    lane_distance = self.get_buffer().get_distance_sink(lane2)
            elif lane1 == "source":
                if isinstance(lane2, str) and lane2 == "sink":
                    lane_distance = self.get_buffer().get_distance_source_to_sink()
                else:
                    # Distance from source to lane2
                    lane_distance = self.get_buffer().get_distance_source(lane2)
            else:
                raise ValueError(f"Unknown string location for lane1: {lane1}")
        elif isinstance(lane2, str):
            if lane2 == "sink": 
                lane_distance = self.get_buffer().get_distance_sink(lane1)
            elif lane2 == "source": 
                lane_distance = self.get_buffer().get_distance_source(lane1)
            else:
                raise ValueError(f"Unknown string location for lane2: {lane2}")
        else: 
            lane_distance = self.get_buffer().get_distance_lanes(lane1, lane2)

        if tier1 is None or tier1 == 1: 
            t1 = 1
        else: 
            t1 = tier1.get_id()
        if tier2 is None or tier2 == 1:
            t2 = 1
        else: 
            t2 = tier2.get_id()
        tier_distance = t1-1 + t2-1
        return lane_distance + tier_distance


if __name__ == '__main__': 
    from src.examples_gen.lane_stack_gen import LanedStackGen

    # Test instance generator with priorities
    instance2 = Instance(
        layout_file="examples/Size_3x3_Layout_3x3_sink_source.csv",
        fill_level=0.8,
        max_p=4,
        height=1,
        seed=1,
        access_directions={"north": True, "east": True, "south": True, "west": True}, 
        exampleGenerator=LanedStackGen(max_priority=4, fill_level=0.7, seed=1), 
    )
    print(f"Generated Instance: {instance2}")
    instance2.save_instance("experiments/test2.json")

    # Test instance loader with priorities
    instance = Instance(InstanceLoader("experiments/test2.json"))
    print(f"InstanceLoader: {instance}")
    instance.save_instance("experiments/test3.json")

    # Test instance generator with unit loads
    instance3 = Instance(
        layout_file="examples/Size_3x3_Layout_2x2_sink_source.csv",
        fill_level=0.8,
        max_p=0,
        height=1,
        seed=1,
        access_directions={"north": True, "east": True, "south": True, "west": True}, 
        exampleGenerator=UnitLoadGenerator(tw_length=15, fill_level=0.8, seed=1),
    )
    print(f"Generated Instance: {instance3}")
    instance3.save_instance("experiments/test.json")

    # Test instance loader with unit loads
    instance4 = Instance(InstanceLoader("experiments/test.json"))
    print(f"InstanceLoader: {instance4}")