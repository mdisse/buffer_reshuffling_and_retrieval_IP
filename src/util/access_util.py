from src.bay.access_bay import AccessBay

def next_in_direction(bay : AccessBay, point : tuple, direction : str):
        """
        Gets next stack from access direction
        
        Parameters:
        bay (AccessBay): the bay to search in
        point (tuple): Y,X coordinates of the current stack
        direction (str): access direction

        Returns:
        tuple: the next point from this access direction 
            or None if reaches the end
        """
        length, width, height = bay.state.shape

        if point[0] < 0 or point[0] >= length:
            raise ValueError('Invalid Y coordinate: ' + str(point))

        if point[1] < 0 or point[1] >= width:
            raise ValueError('Invalid X coordinate: ' + str(point))

        if direction == 'north':
            if point[0] == length - 1:
                return None
            else:
                return (point[0] + 1, point[1])
        elif direction == 'south':
            if point[0] == 0:
                return None
            else:
                return (point[0] - 1, point[1])
        elif direction == 'west':
            if point[1] == width - 1:
                return None
            else:
                return (point[0], point[1] + 1)
        elif direction == 'east':
            if point[1] == 0:
                return None
            else:
                return (point[0], point[1] - 1)

        raise ValueError('Invalid direction')
