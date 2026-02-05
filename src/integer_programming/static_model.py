import gurobipy as gp
from src.instance.instance import Instance
from src.integer_programming.constraints import * 
from src.integer_programming.initial_config_constraints import *
from math import ceil

class StaticModel:
    def __init__(self, instance: Instance, verbose=False) -> None:
        self.instance = instance
        self.verbose = verbose
        self.model = gp.Model("BRR_Static")
        self.Unit_loads = self.instance.unit_loads
        self.T = self._calculate_max_T() + 1
        self.Lanes = self.instance.get_buffer().get_virtual_lanes()[1:]      # Exclude the source lane
        for lane in self.Lanes[:-1]:    # reverse tiers as model uses them differently than the unit load gen creates them
            lane.reverse_tiers()

        # Define variables
        self.create_variables()

        self.add_objectives()
        self.add_constraints()

    def create_variables(self) -> None:
        """ 
        Creates the variables for the model defined by the instance
        This is neither pythonic nor efficient, but it is great to understand how it works ;)
        """
        # b_{i,j,n,t}   if unit load n is in tier j of lane i at time t
        self.b_vars = {}
        for t in range(1, self.T): 
            for n in self.Unit_loads: 
                for i in self.Lanes[:-1]: 
                    for j in i.get_tiers(): 
                        self.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"b_i{i.get_ap_id()}_j{j.get_id()}_n{n.get_id()}_t{t}")

        # x_{i,j,k,l,n,t}   if unit load n is relocated from tier j of lane i to tier l of lane k at time t
        self.x_vars = {}
        for t in range(1, self.T):
            for n in self.Unit_loads: 
                for i in self.Lanes[:-1]: 
                    for j in i.get_tiers():
                        for k in self.Lanes[:-1]:
                            for l in k.get_tiers():
                                self.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"x_i{i.get_ap_id()}_j{j.get_id()}_k{k.get_ap_id()}_l{l.get_id()}_n{n.get_id()}_t{t}")

        # y_{i,j,n,t}   if unit load n is retrieved from tier j of lane i at time t
        self.y_vars = {}
        for t in range(1, self.T):
            for n in self.Unit_loads: 
                for i in self.Lanes[:-1]: 
                    for j in i.get_tiers():
                        self.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"y_i{i.get_ap_id()}_j{j.get_id()}_n{n.get_id()}_t{t}")

        # g_{n,t}   if unit load n has been retrieved from the buffer at time t' < t
        self.g_vars = {}
        for t in range(1, self.T):
            for n in self.Unit_loads:
                self.g_vars[(n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"g_n{n.get_id()}_t{t}")

        # e_{i,j,k,l,t}  if the vehicle repositions from tier j of lane i to tier l of lane k at time t
        self.e_vars = {}
        for t in range(1, self.T):
            for i in self.Lanes: 
                for j in i.get_tiers():
                    for k in self.Lanes:
                        for l in k.get_tiers():
                            self.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"e_i{i.get_ap_id()}_j{j.get_id()}_k{k.get_ap_id()}_l{l.get_id()}_t{t}")

        # c_{i,j,t}  if the vehicle is in tier j of lane i at time t
        self.c_vars = {}
        for t in range(1, self.T):
            for i in self.Lanes: 
                for j in i.get_tiers():
                    self.c_vars[(i.get_ap_id(), j.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"c_i{i.get_ap_id()}_j{j.get_id()}_t{t}")

        self.model.update()
        if self.verbose:
            print(f"b_vars: {len(self.b_vars)}")
            print(f"x_vars: {len(self.x_vars)}")
            print(f"y_vars: {len(self.y_vars)}")
            print(f"g_vars: {len(self.g_vars)}")
            print(f"e_vars: {len(self.e_vars)}")
            print(f"c_vars: {len(self.c_vars)}")
            print(f"Total vars: {len(self.b_vars) + len(self.x_vars) + len(self.y_vars) + len(self.g_vars) + len(self.e_vars) + len(self.c_vars)}")


    def add_objectives(self) -> None:
        """
        Adds the objective function to the model
        """
        obj = gp.LinExpr()
        for i in self.Lanes: 
            for j in i.get_tiers(): 
                for t in range(1, self.T): 
                    for n in self.Unit_loads:
                        # Check if the variable exists before adding it to the objective
                        if (i.get_ap_id(), j.get_id(), n.get_id(), t) in self.y_vars:
                            obj += self.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] * self._calculate_distance(i, j, "sink", None)
                    for k in self.Lanes: 
                        for l in k.get_tiers():
                            # Check if the variable exists before adding it to the objective
                            if (i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t) in self.e_vars:
                                obj += self.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t)] * self._calculate_distance(i, j, k, l)
                            for n in self.Unit_loads:
                                # Check if the variable exists before adding it to the objective
                                if (i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t) in self.x_vars:
                                    obj += self.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t)] * self._calculate_distance(i, j, k, l)

        # Set the objective function to the model
        self.model.setObjective(obj, gp.GRB.MINIMIZE)
        # if self.verbose: 
        #     print(f"Objective: {obj}")

        

    def add_constraints(self) -> None:
        """
        Adds the constraints to the model
        """
        # Constraints
        ul_in_yard(self)
        at_most_ul_in_slot(self)
        no_hollow_spaces(self)
        one_move_per_vehicle(self)
        one_move_per_vehicle_sink(self)
        one_move_per_ul(self)
        config_update(self)
        vehicle_update(self)
        vehicle_update_sink(self)
        relations_retrieval_config_vars(self)
        retrieval_after_arrival(self)
        retrieval_in_window(self)
        retrieval_before_deadline(self)
        # Initial configuration constraints
        vehicle_start(self)
        unit_load_start(self)
        self.model.update()
        if self.verbose: 
            print(f"Constraints: {len(self.model.getConstrs())}")


    def get_state(self, t=1): 
        """
        Returns the state of the buffer at time t
        """
        for lane in self.Lanes:
            print(lane)
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("b") and v.VarName.endswith(f"t{t}"):
                    print(f"{v.VarName} = {v.X}")

    def print_g(self): 
        """
        Returns the end time of the retrieval of the unit loads
        """
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("g"):
                    print(f"{v.VarName} = {v.X}")

    def print_c(self): 
        """
        Returns the end time of the retrieval of the unit loads
        """
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("c"):
                    print(f"{v.VarName} = {v.X}")

    def print_b(self):
        """
        Returns the end time of the retrieval of the unit loads
        """
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("b"):
                    print(f"{v.VarName} = {v.X}")


    def solve(self) -> None:
        self.model.optimize()


    def get_solution(self):
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            solution = {}  # Initialize an empty dictionary to store variable names and their optimized values
            # Filter variables that start with 'e', 'x', or 'y' and have a value of 1
            for var in self.model.getVars():
                if var.varName.startswith(('e', 'x', 'y')) and var.x == 1:
                    solution[var.varName] = var.x
            sorted_solution = {k: v for k, v in sorted(solution.items(), key=self._sort_by_time)} 
            return sorted_solution
        elif self.model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible.")
            # Compute the Irreducible Inconsistent Subsystem (IIS)
            self.model.computeIIS()
            print("\nThe following constraint(s) cannot be satisfied:")
            for constr in self.model.getConstrs():
                if constr.IISConstr:
                    print(f"- {constr.constrName}")
            for var in self.model.getVars():
                if var.IISLB > 0 or var.IISUB > 0:
                    print(f"- Bounds on variable {var.varName}")
            # Optionally, write the IIS to a file for further inspection
            self.model.write("model_infeasible.ilp")
            return None
        else:
            print(f"Optimization was not successful. Status code: {self.model.status}")
            return None
        
    
    def _sort_by_time(self, item): 
        key = item[0]
        # Assuming the time component is always at the end of the key following the format '_t#'
        time_part = key.split('_t')[-1]
        return int(time_part)


    def _calculate_max_T(self): 
        """
        Calculates the maximum time of the instance
        """
        T = 0
        for unit_load in self.Unit_loads: 
            if unit_load.arrival_end is not None:
                T = max(T, unit_load.retrieval_end, unit_load.arrival_end)
            else:
                T = max(T, unit_load.retrieval_end)
        return T
    
    def _calculate_distance(self, lane1, tier1, lane2, tier2): 
        """
        Calculates the distance between two lanes including tier depth.
        Uses the instance's calculate_distance method for consistency.
        """
        return self.instance.calculate_distance(lane1, tier1, lane2, tier2)

    def calculate_travel_time(self, lane1, tier1, lane2, tier2): 
        """
        Calculates the travel time between two lanes
        Can be modified to also consider the tiers of the slots, 
        to consider the speed of the vehicle or to 

        returns the travel time rounded up to the next integer, as 
        the timesteps are discrete and the travel time is continuous
        """
        distance = self._calculate_distance(lane1, tier1, lane2, tier2)
        travel_time = distance
        return max(1, ceil(travel_time))        # return at least 1 to avoid multiple repositionings at a time step 

    def get_solution_distances(self, decisions): 
        """
        Takes the solution's decisions and returns the distances of the vehicles
        """
        solution_dict = {}
        for k, v in decisions.items():
            l1, t1, l2, t2 = self._get_lane_and_tiers(k)
            solution_dict.update({k: self._calculate_distance(l1, t1, l2, t2)})
        return solution_dict
            
    def _get_lane_and_tiers(self, solution_key): 
        """
        Returns the lane(s) and the tier(s) of the solution key
        """
        decision, lane1, tier1, lane2, tier2 = solution_key.split("_")[:5] 
        if decision == "e" or decision == "x": 
            for lane in self.Lanes: 
                if lane.get_ap_id() == int(lane1[1:]): 
                    l1 = lane
                    for tier in lane.get_tiers():
                        if tier.get_id() == int(tier1[1:]):
                            t1 = tier
                if lane.get_ap_id() == int(lane2[1:]):
                    l2 = lane
                    for tier in lane.get_tiers():
                        if tier.get_id() == int(tier2[1:]):
                            t2 = tier
        elif decision == "y":
            l2 = "sink"
            t2 = None
            for lane in self.Lanes: 
                if lane.get_ap_id() == int(lane1[1:]): 
                    l1 = lane
                    for tier in lane.get_tiers():
                        if tier.get_id() == int(tier1[1:]):
                            t1 = tier
        return l1, t1, l2, t2