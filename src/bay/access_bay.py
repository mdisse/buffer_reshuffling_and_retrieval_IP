import numpy as np

class AccessBay():
    def __init__(self, x: int, y: int, state: np.ndarray, access_points: list):
        # position of the north-west stack
        self.x = x
        self.y = y

        self.state = state
        self.width = state.shape[1]
        self.length = state.shape[0]
        if len(state.shape) == 2:
            self.height = 1
        else:
            self.height = state.shape[2]
        self.access_points = access_points
        self.virtual_lanes = None

    def __str__(self):
        return '{0}x{1} Bay at row {2}, column {3}, access_points {4}'.format(
            self.width, self.length, self.y, self.x, [point.ap_id for point in self.access_points]
        )

    def to_data_dict(self):
        data = dict()
        data['x'] = self.x
        data['y'] = self.y
        data['width'] = self.width
        data['length'] = self.length
        data['access_directions'] = list({point.direction for point in self.access_points})
        data['access_point_ids'] = [point.ap_id for point in self.access_points]
        return data

    def get_id(self):
        return str(self)

    def get_state_int(self): 
        """
        Returns the state of the bay as a numpy array of integers
        """
        return self.state 

    def get_state_ul(self, unit_loads : list):
        """
        Returns the state of the bay as a list of unit loads
        """
        bay_list = []
        for lane in self.state: 
            lane_list = []
            for slot in lane: 
                # search the unit load with the id in the slot and append it to the list
                unit_load = next((ul for ul in unit_loads if ul.id == slot), None)
                lane_list.append(unit_load)
            bay_list.append(lane_list)
        return bay_list
