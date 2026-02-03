import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))
import json

import numpy as np
from src.bay.access_bay import AccessBay
from src.examples_gen.rand_lane_gen import RandLaneGen

class InstanceLoader(): 
    def __init__(self, instance_file): 
        self.instance_path = instance_file if isinstance(instance_file, str) else None
        instance_file = self._preprocessing_instance_file(instance_file)
        self.layout_file_name = instance_file["layout_file"].split("/")[-1].split(".")[0]
        self.fill_level = instance_file["fill_level"]
        self.max_priority = instance_file.get("max_priority", None)  
        self.height = instance_file["height"]
        self.seed = instance_file["seed"]
        self.fleet_size = instance_file["fleet_size"]
        self.vehicle_speed = instance_file["vehicle_speed"]
        self.handling_time = instance_file["handling_time"]
        self.rs_max = instance_file["rs_max"]
        self.as_max = instance_file["as_max"]
        self.time_window_length = instance_file["time_window_length"]
        example_bay = list(instance_file["bay_info"].keys())[0]
        # Iterate over all values in the bay_info dictionary to find all access directions
        all_directions = set()
        for bay_info in instance_file["bay_info"].values():
            all_directions.update(bay_info["access_directions"])
        self.access_directions = self._create_access_directions_dict(list(all_directions))
        self.inital_state = instance_file["initial_state"]
        self.bay_info = instance_file["bay_info"]
        # Try except here to catch old instance files where the sink was not implemented yet
        try:    
            self.sink = instance_file["sink"]
        except: 
            self.sink = False
        try:
            self.source = instance_file["source"]
        except: 
            self.source = False

        # Time window related variables
        if self.max_priority == 0: 
            self.unit_loads = instance_file["unit_loads"]
            self._assign_priorities()
            
    def __str__(self): 
        return str({
            "layout_file_name": self.layout_file_name,
            "fill_level": self.fill_level,
            "max_p": self.max_priority,
            "height": self.height,
            "seed": self.seed,
            "access_directions": self.access_directions,
            "initial_state": self.inital_state,
        })

    def _preprocessing_instance_file(self, instance_file): 
        """
        This function allows to either pass a path to the instance file or directly a instance json
        """
        try: 
            if isinstance(instance_file, str): 
                with open(instance_file, "r") as f: 
                    instance_file = json.load(f)
            _ = instance_file["layout_file"].split("/") # Test for TypeError
            return instance_file
        except TypeError: 
            print(f"Error: Make sure that you pass a path to an instance file or a dictionary to the InstanceLoader.\n You passed {instance_file}")
            sys.exit(1)

    def _create_access_directions_dict(self, access_directions): 
        north, east, south, west = False, False, False, False
        if "north" in access_directions: 
            north = True
        if "east" in access_directions: 
            east = True
        if "south" in access_directions: 
            south = True
        if "west" in access_directions: 
            west = True
        return {
            "north": north, 
            "east" : east,
            "south": south,
            "west" : west
        }

    def generate_bays_priorities(self, bays, height: int=0, source: bool=False):
        """Generates random lanes and populates them with stacks """
        for bay in bays: 
            if height is not None: 
                bay.height = height
            else: 
                bay.height = 1
            bay.state = np.zeros((bay.length, bay.width, bay.height), dtype=int)
            self._fill_bay(bay)

    def _fill_bay(self, bay: AccessBay): 
        bay_key = self._bay_finder(bay.x, bay.y)
        state = self.inital_state[bay_key]
        bay.state = np.asarray(state)

    def _bay_finder(self, x, y): 
        for bay_key in self.bay_info: 
            bay_info_value = self.bay_info[bay_key]
            if bay_info_value["x"] == x and bay_info_value["y"] == y:
                return bay_key
        print("Bay could not be found in the given JSON with the x and y coordinates. This Instance file seems to be manipulated.")
        sys.exit(1)

    def get_initial_state(self): 
        return self.inital_state

    def get_instance_path(self):
        return self.instance_path

    def get_layout_filename(self): 
        return f"examples/{self.layout_file_name}.csv"
    
    def get_access_directions(self): 
        return self.access_directions

    def get_max_p(self):
        return self.max_priority
    
    def get_fill_level(self):
        return self.fill_level

    def get_height(self):
        return self.height
    
    def get_seed(self):
        return self.seed
    
    def get_access_directions(self):
        return self.access_directions

    def get_sink(self): 
        return self.sink
    
    def get_source(self): 
        return self.source

    def get_unit_loads(self): 
        if self.max_priority > 0:
            return False
        return self.unit_loads

    def get_fleet_size(self): 
        return self.fleet_size

    def get_rs_max(self): 
        return self.rs_max
    
    def get_as_max(self): 
        return self.as_max
    
    def get_time_window_length(self): 
        return self.time_window_length
    
    def get_vehicle_speed(self): 
        return self.vehicle_speed

    def get_handling_time(self): 
        return self.handling_time
    
    def _assign_priorities(self):
        """Assigns priorities to unit loads based on their due dates (retrieval_end)."""
        # Sort unit loads by their due date (earliest first)
        sorted_uls = sorted(self.unit_loads, key=lambda ul: ul['retrieval_end'])
        
        # Assign priority based on sorted order
        for i, ul_data in enumerate(sorted_uls):
            # Find the corresponding unit load object and set its priority
            for ul_obj in self.unit_loads:
                if ul_obj['id'] == ul_data['id']:
                    ul_obj['priority'] = i + 1  # Priority 1 is the highest
                    break