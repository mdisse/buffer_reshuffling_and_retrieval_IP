class AutonomousMobileRobot(object):
    '''
    Abstract class for mobile robots.

    Parameters:
#       last moves
    '''

    def __init__(self, id=None):
        self._last_moves = []
        self.id = id

    def get_executet_moves(self):
        return len(self._last_moves)

    def get_robot_move_history(self):
        return self._last_moves

    def move_robot(self, move, distance):
        raise NotImplementedError('Please Implement this method')

    def idle_step(self):
        raise NotImplementedError('Please Implement this method')

    def get_id(self):
        return self.id