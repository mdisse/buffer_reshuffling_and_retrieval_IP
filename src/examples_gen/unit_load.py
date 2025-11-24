class UnitLoad(): 
    def __init__(self, 
                 id, 
                 retrieval_start: int=None, 
                 retrieval_end: int=None, 
                 arrival_start: int=None, 
                 arrival_end: int=None,
                 storage_priority: int=None,
                 retrieval_priority: int=None,
                 is_mock: bool=False): 
        """
        In the variant without a source, the arrival_start and arrival_end are always None. 
        Time is modeled in discrete time steps as integers.
        """
        self.id = id
        self.retrieval_start = retrieval_start
        self.retrieval_end = retrieval_end
        self.arrival_start = arrival_start
        self.arrival_end = arrival_end
        self.storage_priority = storage_priority
        self.retrieval_priority = retrieval_priority
        self.priority = None  # Active priority
        self.due_date = retrieval_end
        self.is_stored = False
        self.is_at_sink = False

                # check if the unit load is already stored in the buffer
        if not is_mock:
            # For real unit loads, we need to check the feasibility of the time windows
            if arrival_start is None or (arrival_end is not None and arrival_end <= 0):
                # If arrival_start is None or arrival_end <= 0, the unit load is not stored
                self.is_stored = True
            else: 
                self.is_stored = False

            self._feasibility_checks()
            self._store_if_None()
        else:
            self.is_stored = False
        
        # Set initial priority after state is determined. This is important for when
        # the unit load objects are created, as the priorities are not yet assigned.
        # The priorities will be correctly assigned later in the A* solver.
        if self.is_stored:
            self.priority = self.retrieval_priority
        else:
            self.priority = self.storage_priority


    def __str__(self) -> str:
        return f"UnitLoad {self.id}, retrieval: {self.retrieval_start} - {self.retrieval_end}, arrival: {self.arrival_start} - {self.arrival_end}, storage_prio: {self.storage_priority}, retrieval_prio: {self.retrieval_priority}, active_prio: {self.priority}, stored: {self.is_stored}, at_sink: {self.is_at_sink}"
    
    def _feasibility_checks(self):
        if self.retrieval_start < 1: 
            raise ValueError("retrieval_start must be greater than 0")
        if self.retrieval_start is None: 
            raise ValueError("retrieval_start must be set")
        if self.retrieval_end is None: 
            raise ValueError("retrieval_end must be set")
        if self.retrieval_end < self.retrieval_start: 
            raise ValueError("retrieval_end must be greater than retrieval_start")
        if self.arrival_start is not None:
            if self.arrival_end is None:
                raise ValueError("arrival_end must be set if arrival_start is set")
            if self.arrival_end < self.arrival_start:
                raise ValueError("arrival_end must be greater than arrival_start")
            if self.arrival_end > self.retrieval_start:
                raise ValueError("arrival_end must be smaller than retrieval_start")
    
    def _store_if_None(self): 
        if self.arrival_start is None: 
            self.arrival_start = 0
        if self.arrival_end is None: 
            self.arrival_end = 0
        
    def retrieve(self): 
        if self.is_stored is False:
            raise ValueError("unit load must be stored to retrieve it")
        self.is_at_sink = True
        self.is_stored = False

    def store(self):
        if self.arrival_start is None:
            raise ValueError("arrival_start must be set to store the unit load")
        self.is_at_sink = False
        self.is_stored = True
        self.priority = self.retrieval_priority

    def to_data_dict(self): 
        data = dict()
        data['id'] = self.id
        data['retrieval_start'] = self.retrieval_start
        data['retrieval_end'] = self.retrieval_end
        data['arrival_start'] = self.arrival_start
        data['arrival_end'] = self.arrival_end
        return data

    def get_id(self): 
        return self.id
    
    def get_retrieval_start(self): 
        return self.retrieval_start
    
    def get_retrieval_end(self): 
        return self.retrieval_end
    
    def get_arrival_start(self): 
        return self.arrival_start
    
    def get_arrival_end(self): 
        return self.arrival_end

    def set_priority(self, priority: int): 
        self.priority = priority

    def set_retrieval_priority(self, priority: int):
        self.retrieval_priority = priority

    def set_storage_priority(self, priority: int):
        self.storage_priority = priority 
        """
        Set the priority of the unit load. 
        Priority is an integer where lower values indicate higher priority.
        """
        if priority < 1: 
            raise ValueError("Priority must be greater than 0")
        self.priority = priority
    
    def get_priority(self):
        """
        Get the priority of the unit load. 
        If no priority is set, return None.
        """
        return self.priority 
    
    @property
    def stored(self):
        """
        Alias for is_stored to maintain compatibility with existing code.
        """
        return self.is_stored
    
    @stored.setter
    def stored(self, value):
        """
        Setter for stored property.
        """
        self.is_stored = value

    def copy(self):
        """
        Creates an efficient copy of the unit load with all its current state.
        """
        new_ul = UnitLoad(
            id=self.id,
            retrieval_start=self.retrieval_start,
            retrieval_end=self.retrieval_end,
            arrival_start=self.arrival_start,
            arrival_end=self.arrival_end,
            storage_priority=self.storage_priority,
            retrieval_priority=self.retrieval_priority,
            is_mock=True  # Skip feasibility checks for copies
        )
        # Copy current state
        new_ul.priority = self.priority
        new_ul.due_date = self.due_date
        new_ul.is_stored = self.is_stored
        new_ul.is_at_sink = self.is_at_sink
        return new_ul