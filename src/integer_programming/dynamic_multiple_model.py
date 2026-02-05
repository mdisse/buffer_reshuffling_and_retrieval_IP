import gurobipy as gp
from src.instance.instance import Instance
from src.integer_programming.constraints import * 
from src.integer_programming.initial_config_constraints import *
from math import ceil

class DynamicMultipleModel:
    def __init__(self, instance: Instance, decisions=False, verbose=False, check_mode=False) -> None:
        self.instance = instance
        self.verbose = verbose
        self.check_mode = check_mode
        self.model = gp.Model("BRR_Dynamic_Multiple_AMRs")
        
        # ONLY disable optimization when check_mode=True (for heuristic validation)
        # Regular solve mode still uses all optimization parameters
        if self.check_mode:
            # For check mode: disable presolve and limit iterations for fast feasibility check only
            self.model.setParam('Presolve', 0)  # Disable presolve
            self.model.setParam('TimeLimit', 10)  # Very short time limit
            self.model.setParam('IterationLimit', 0)  # Don't optimize, just check constraints
            self.model.setParam('FeasibilityTol', 1e-8)
            self.model.setParam('OptimalityTol', 1e-8)
        else:
            # Normal optimization mode (for run_BRR_experiment.py)
            self.model.setParam('TimeLimit', 3600)
            self.model.setParam('MIPGap', 0.05)
            self.model.setParam('Heuristics', 0.5)
            self.model.setParam('BarHomogeneous', 1)
            self.model.setParam('MIPFocus', 1)
            self.model.setParam("NumericFocus", 3)  # High focus on numerical accuracy
            self.model.setParam("ScaleFlag", 2)  # Aggressive scaling
            self.model.setParam("ObjScale", 0.5)
            self.model.setParam("FeasibilityTol", 1e-8)
            self.model.setParam("OptimalityTol", 1e-8)
            
        self.Unit_loads = self.instance.unit_loads
        self.T = self._calculate_max_T() + 51 # Add safety buffer for heuristic tardiness
        self.Lanes = self.instance.get_buffer().get_virtual_lanes()
        for lane in self.Lanes[1:-1]:   # reverse tiers as model uses them differently than the unit load gen creates them 
            lane.reverse_tiers()
        self.Vehicles = self.instance.get_vehicles()
        self.vehicle_speed = self.instance.get_vehicle_speed()

        # Define variables
        self.create_variables()

        self.add_objectives()
        self.add_constraints()
        if decisions:
            self.add_decisions(decisions)

    def create_variables(self) -> None:
        """ 
        Creates the variables for the model defined by the instance
        This is neither pythonic nor efficient, but it is great to understand how it works ;)
        """
        # b_{i,j,n,t}   if unit load n is in tier j of lane i at time t
        self.b_vars = {}
        for t in range(1, self.T): 
            for n in self.Unit_loads: 
                for i in self.Lanes[1:-1]: 
                    for j in i.get_tiers(): 
                        self.b_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"b_i{i.get_ap_id()}_j{j.get_id()}_n{n.get_id()}_t{t}")

        # x_{i,j,k,l,n,t,v}   if unit load n is relocated from tier j of lane i to tier l of lane k at time t by vehicle v
        self.x_vars = {}
        for v in self.Vehicles:
            for t in range(1, self.T):
                for n in self.Unit_loads: 
                    for i in self.Lanes[1:-1]: 
                        for j in i.get_tiers():
                            for k in self.Lanes[1:-1]:
                                for l in k.get_tiers():
                                    self.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id())] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"x_i{i.get_ap_id()}_j{j.get_id()}_k{k.get_ap_id()}_l{l.get_id()}_n{n.get_id()}_t{t}_v{v.get_id()}")

        # y_{i,j,n,t,v}   if unit load n is retrieved from tier j of lane i at time t by vehicle v
        self.y_vars = {}
        for v in self.Vehicles:
            for t in range(1, self.T):
                for n in self.Unit_loads: 
                    for i in self.Lanes[:-1]: 
                        for j in i.get_tiers():
                            self.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id())] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"y_i{i.get_ap_id()}_j{j.get_id()}_n{n.get_id()}_t{t}_v{v.get_id()}")

        # g_{n,t}   if unit load n has been retrieved from the buffer at time t' < t
        self.g_vars = {}
        for t in range(1, self.T):
            for n in self.Unit_loads:
                self.g_vars[(n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"g_n{n.get_id()}_t{t}")

        # z_{i,j,n,t,v}  if a unit load n is stored in i,j at time t from vehicle v
        self.z_vars = {}
        for v in self.Vehicles: 
            for t in range(1, self.T):
                for n in self.Unit_loads:
                    for i in self.Lanes[1:-1]:
                        for j in i.get_tiers():
                            self.z_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id())] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"z_i{i.get_ap_id()}_j{j.get_id()}_n{n.get_id()}_t{t}_v{v.get_id()}")
        
        # s_{n,t}  if a unit load n has been stored in the buffer at time t' < t
        self.s_vars = {}
        for t in range(1, self.T):
            for n in self.Unit_loads:
                self.s_vars[(n.get_id(), t)] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"s_n{n.get_id()}_t{t}")


        # e_{i,j,k,l,t, v}  if the vehicle v repositions from tier j of lane i to tier l of lane k at time t
        self.e_vars = {}
        for v in self.Vehicles: 
            for t in range(1, self.T):
                for i in self.Lanes: 
                    for j in i.get_tiers():
                        for k in self.Lanes:
                            for l in k.get_tiers():
                                self.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id())] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"e_i{i.get_ap_id()}_j{j.get_id()}_k{k.get_ap_id()}_l{l.get_id()}_t{t}_v{v.get_id()}")

        # c_{i,j,t, v}  if the vehicle v is in tier j of lane i at time t
        self.c_vars = {}
        for v in self.Vehicles: 
            for t in range(1, self.T):
                for i in self.Lanes: 
                    for j in i.get_tiers():
                        self.c_vars[(i.get_ap_id(), j.get_id(), t, v.get_id())] = self.model.addVar(vtype=gp.GRB.BINARY, name=f"c_i{i.get_ap_id()}_j{j.get_id()}_t{t}_v{v.get_id()}")

        self.model.update()
        if self.verbose:
            print(f"b_vars: {len(self.b_vars)}")
            print(f"x_vars: {len(self.x_vars)}")
            print(f"y_vars: {len(self.y_vars)}")
            print(f"g_vars: {len(self.g_vars)}")
            print(f"z_vars: {len(self.z_vars)}")
            print(f"s_vars: {len(self.s_vars)}")
            print(f"e_vars: {len(self.e_vars)}")
            print(f"c_vars: {len(self.c_vars)}")
            print(f"Total vars: {len(self.b_vars) + len(self.x_vars) + len(self.y_vars) + len(self.g_vars) + len(self.z_vars) + len(self.s_vars) + len(self.e_vars) + len(self.c_vars)}")


    def add_objectives(self) -> None:
        """
        Adds the objective function to the model
        """
        obj = gp.LinExpr()
        for v in self.Vehicles:
            for i in self.Lanes: 
                for j in i.get_tiers(): 
                    for t in range(1, self.T): 
                        for n in self.Unit_loads:
                            # Check if the variable exists before adding it to the objective
                            if (i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()) in self.y_vars:
                                obj += self.y_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id())] * self._calculate_distance(i, j, "sink", None)
                            if (i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id()) in self.z_vars: 
                                obj += self.z_vars[(i.get_ap_id(), j.get_id(), n.get_id(), t, v.get_id())] * self._calculate_distance(i, j, "source", None)
                        for k in self.Lanes: 
                            for l in k.get_tiers():
                                # Check if the variable exists before adding it to the objective
                                if (i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id()) in self.e_vars:
                                    obj += self.e_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), t, v.get_id())] * self._calculate_distance(i, j, k, l)
                                for n in self.Unit_loads:
                                    # Check if the variable exists before adding it to the objective
                                    if (i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id()) in self.x_vars:
                                        obj += self.x_vars[(i.get_ap_id(), j.get_id(), k.get_ap_id(), l.get_id(), n.get_id(), t, v.get_id())] * self._calculate_distance(i, j, k, l)

        # Set the objective function to the model
        self.model.setObjective(obj, gp.GRB.MINIMIZE)

        

    def add_constraints(self) -> None:
        """
        Adds the constraints to the model
        """
        # Constraints
        ul_in_yard_dm(self)
        ul_in_yard2_dm(self)
        at_most_ul_in_slot_dm(self)
        no_hollow_spaces_dm(self) # Disabled to allow initial states with hollow spaces
        one_move_per_vehicle_dm(self)
        one_move_per_vehicle_sink_dm(self)
        one_move_per_vehicle_source_dm(self)
        direct_retrieval_if_not_stored_dm(self)
        one_move_per_ul_dm(self)
        config_update_dm(self)
        relations_retrieval_config_vars_dm(self)
        relations_storage_config_vars_dm(self)
        vehicle_update_dm(self)
        vehicle_update_sink_dm(self)
        vehicle_update_source_dm(self)
        retrieval_after_arrival_dm(self)
        retrieval_in_window_dm(self)
        retrieval_before_deadline_dm(self)
        stack_after_arrival_dm(self)
        stack_in_window_dm(self)
        stack_before_deadline_dm(self)
        one_vehicle_per_lane_dm(self)
        lane_monopolization(self)
        lifo(self)
        # Initial configuration constraints
        vehicle_start_dm(self)
        unit_load_start_dm(self)
        self.model.update()
        if self.verbose: 
            print(f"Constraints: {len(self.model.getConstrs())}")

    def add_decisions(self, decisions: list[str]) -> None:
        """
        Adds constraints to the model to fix variables based on a list of decision strings.
        Each decision string identifies a variable (e, x, y, or z) and its indices.
        The corresponding variable in the model is constrained to be equal to 1.
        Raises ValueError if a decision string is malformed or the variable is not found.
        """
        if self.verbose:
            print(f"Received decisions to fix: {decisions}")

        for decision_str in decisions:
            parts = decision_str.split('_')
            var_type = parts[0]
            
            var_to_set = None
            
            try:
                if var_type == 'e':
                    # Expected format: e_i<val>_j<val>_k<val>_l<val>_t<val>_v<val>
                    if len(parts) == 7:
                        i_val = int(parts[1][1:])
                        j_val = int(parts[2][1:])
                        k_val = int(parts[3][1:])
                        l_val = int(parts[4][1:])
                        time_val = int(parts[5][1:])
                        vehicle_val = int(parts[6][1:])
                        var_to_set = self.e_vars.get((i_val, j_val, k_val, l_val, time_val, vehicle_val))
                    else:
                        raise ValueError(f"Malformed decision string for type 'e': {decision_str}. Expected 7 parts, got {len(parts)}.")
                elif var_type == 'x':
                    # Expected format: x_i<v1>_j<v2>_k<v3>_l<v4>_n<v5>_t<v6>_v<v7>
                    if len(parts) == 8:
                        i_val = int(parts[1][1:])
                        j_val = int(parts[2][1:])
                        k_val = int(parts[3][1:])
                        l_val = int(parts[4][1:])
                        n_val = int(parts[5][1:])
                        time_val = int(parts[6][1:])
                        vehicle_val = int(parts[7][1:])
                        var_to_set = self.x_vars.get((i_val, j_val, k_val, l_val, n_val, time_val, vehicle_val))
                    else:
                        raise ValueError(f"Malformed decision string for type 'x': {decision_str}. Expected 8 parts, got {len(parts)}.")
                elif var_type == 'y':
                    # Expected format: y_i<v1>_j<v2>_n<v3>_t<v4>_v<v5>
                    if len(parts) == 6:
                        i_val = int(parts[1][1:])
                        j_val = int(parts[2][1:])
                        n_val = int(parts[3][1:])
                        time_val = int(parts[4][1:])
                        vehicle_val = int(parts[5][1:])
                        var_to_set = self.y_vars.get((i_val, j_val, n_val, time_val, vehicle_val))
                    else:
                        raise ValueError(f"Malformed decision string for type 'y': {decision_str}. Expected 6 parts, got {len(parts)}.")
                elif var_type == 'z':
                    # Expected format: z_i<v1>_j<v2>_n<v3>_t<v4>_v<v5>
                    if len(parts) == 6:
                        i_val = int(parts[1][1:])
                        j_val = int(parts[2][1:])
                        n_val = int(parts[3][1:])
                        time_val = int(parts[4][1:])
                        vehicle_val = int(parts[5][1:])
                        var_to_set = self.z_vars.get((i_val, j_val, n_val, time_val, vehicle_val))
                    else:
                        raise ValueError(f"Malformed decision string for type 'z': {decision_str}. Expected 6 parts, got {len(parts)}.")
                else:
                    raise ValueError(f"Unknown variable type '{var_type}' in decision string: {decision_str}")
            except ValueError as e: # Catches errors from int() conversion or malformed strings
                raise ValueError(f"Error parsing decision string '{decision_str}': {e}")
            
            if var_to_set:
                self.model.addConstr(var_to_set == 1, name=f"fix_{decision_str}")
                if self.verbose:
                    print(f"Constraint added: {var_to_set.VarName} == 1")
            else:
                # This case handles when var_to_set is None after a successful parse (variable not in dictionary)
                raise ValueError(f"Variable for decision '{decision_str}' not found in the model. Ensure it was defined in create_variables() and the indices are correct.")
        
        self.model.update()

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
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("g"):
                    print(f"{v.VarName} = {v.X}")

    def print_c(self): 
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("c"):
                    print(f"{v.VarName} = {v.X}")

    def print_b(self):
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("b"):
                    print(f"{v.VarName} = {v.X}")

    def print_s(self):
        if self.model.status == gp.GRB.OPTIMAL:  # Check if the model was solved optimally
            for v in self.model.getVars(): 
                if v.X == 1 and v.VarName.startswith("s"):
                    print(f"{v.VarName} = {v.X}")


    def solve(self) -> None:
        self.model.optimize()


    def get_solution(self):
        if self.model.status == gp.GRB.OPTIMAL or self.model.SolCount > 0:  # Check if the model was solved optimally
            solution = {}  # Initialize an empty dictionary to store variable names and their optimized values
            # Filter variables that start with 'e', 'x', 'y' or 'z' and have a value of 1
            for var in self.model.getVars():
                if var.varName.startswith(('e', 'x', 'y', 'z')) and var.x == 1:
                    solution[var.varName] = var.x
            sorted_solution = {k: v for k, v in sorted(solution.items(), key=self._sort_by_time)} 
            return sorted_solution
        elif self.model.status == gp.GRB.INFEASIBLE and self.verbose:
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
        if '_v' in time_part: 
            time_part = time_part.split('_v')[0]
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
        Calculates the distance between two lanes
        Each tier moved adds 1 to the distance
        """
        return self.instance.calculate_distance(lane1, tier1, lane2, tier2)

    def calculate_travel_time(self, lane1, tier1, lane2, tier2, handling_time=False): 
        """
        returns the travel time rounded up to the next integer, as 
        the timesteps are discrete and the travel time is continuous
        """
        distance = self._calculate_distance(lane1, tier1, lane2, tier2)
        travel_time = distance / self.vehicle_speed
        if handling_time: 
            return max(1, ceil(travel_time) + 2*self.instance.get_handling_time())
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
        elif decision == "z": 
            l2 = "source"
            t2 = None
            for lane in self.Lanes: 
                if lane.get_ap_id() == int(lane1[1:]): 
                    l1 = lane
                    for tier in lane.get_tiers(): 
                        if tier.get_id() == int(tier1[1:]):
                            t1 = tier
        return l1, t1, l2, t2