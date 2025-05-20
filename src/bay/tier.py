class Tier: 
    def __init__(self, id: int, inital_ul_id: int): 
        self.id = id
        self.inital_ul_id = inital_ul_id

    def __str__(self):
        return f"Tier: {self.id}, Unit Load: {self.inital_ul_id}"

    def get_id(self): 
        return self.id

    def get_initial_ul_id(self): 
        return self.inital_ul_id