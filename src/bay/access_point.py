class AccessPoint:
    def __init__(self, bay, global_x, global_y, stack_x: int, stack_y: int, direction: str):
        self.bay = bay

        # location in the layout
        self.global_x = global_x
        self.global_y = global_y

        # location of the pointed stack
        self.stack_x = stack_x
        self.stack_y = stack_y

        if not direction in ['north', 'south', 'west', 'east']:
            raise ValueError(f'Wrong diretion in Access Point: {direction}. \
                        Please take from: north, south, west, east')

        # direction of pointing (e.g. 'north')
        self.direction = direction

        self.ap_id = None

    def __str__(self):
        return f"x: {self.stack_x}, y: {self.stack_y}, direction: {self.direction}, ap_id: {self.ap_id}"

    def get_global_yx(self, index=None):
        if index == None:
            return (self.global_y, self.global_x)
        index = (index // self.bay.height) + 1
        dy, dx = 0, 0
        if self.direction == 'north':
            dy = index
        elif self.direction == 'south':
            dy = -index
        elif self.direction == 'west':
            dx = index
        elif self.direction == 'east':
            dx = -index
        return (self.global_y + dy, self.global_x + dx)

    def get_stack_yx(self, index=None):
        if index == None:
            return (self.stack_y, self.stack_x)

        index = index // self.bay.height
        dy, dx = 0, 0
        if self.direction == 'north':
            dy = index
        elif self.direction == 'south':
            dy = -index
        elif self.direction == 'west':
            dx = index
        elif self.direction == 'east':
            dx = -index
        return (self.stack_y + dy, self.stack_x + dx)

    def to_data_dict(self):
        data = dict()
        data['ap_id'] = self.ap_id
        data['direction'] = self.direction
        data['stack_x'] = self.stack_x
        data['stack_y'] = self.stack_y
        data["global_x"] = self.global_x
        data["global_y"] = self.global_y
        return data
