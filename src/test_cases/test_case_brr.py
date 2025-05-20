import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

from src.instance.instance import Instance
from src.integer_programming.static_model import StaticModel
from src.integer_programming.dynamic_multiple_model import DynamicMultipleModel
from src.test_cases.writer_functions import save_resultsBrr
import numpy as np
import gurobipy as gp
import re


class TestCaseBrr: 
    """
    This class is used to test the BRR problem
    """
    def __init__(self, instance: Instance, variant="static", verbose=False) -> None:
        self.instance = instance
        self.variant = variant
        self.feasible = None
        self.results = {}
        self.Lanes = self.instance.get_warehouse().get_virtual_lanes()
        if self.variant == "static":
            self.model = StaticModel(self.instance, verbose=verbose)
        elif self.variant == "dynamic":
            sys.exit("Dynamic model not implemented yet")
        elif self.variant == "dynamic_multiple":
            self.model = DynamicMultipleModel(self.instance, verbose=verbose)
        else:
            ValueError("The choosen variant is not implemented")
        self.model.solve()
        self.print_inital_state()
        self.print_solution(verbose)

        if verbose: 
            self.model.print_c()
            self.model.print_g()
            self.model.print_b()
            self.model.print_s()

    def print_inital_state(self):
        if self.variant == "static": 
            source = None
            lanes = self.model.Lanes[:-1]
            sink = self.model.Lanes[-1]
        elif self.variant == "dynamic": 
            source = self.model.Lanes[0]
            lanes = self.model.Lanes[1:-1]
            sink = self.model.Lanes[-1]
        elif self.variant == "dynamic_multiple": 
            source = self.model.Lanes[0]
            lanes = self.model.Lanes[1:-1]
            sink = self.model.Lanes[-1]
        else: 
            ValueError("The choosen variant is not implemented")
        print("-" * 85)
        sorted_ul = sorted(self.instance.get_unit_loads(), key=lambda x: x.get_id())
        for ul in sorted_ul: 
            print(ul)
        print("-" * 85)
        if source: 
            print(f"Source:")
            print("{:>6} {:<10}".format(f"AP: {source.get_ap_id()}", str(source.stacks)))
        print(f"Lanes:")
        for lane in lanes: 
            print("{:>6} {:<10}".format(f"AP: {lane.get_ap_id()}", str(lane.stacks)))
        print(f"Sink:")
        print("{:>6} {:<10}".format(f"AP: {sink.get_ap_id()}", str(sink.stacks)))

    
    def print_solution(self, verbose=False): 
        solution = self.model.get_solution()
        self.mipgap = self.model.model.MIPGap
        if solution is not None: 
            self.feasible = True
            solutions = self._split_solution_by_vehicle(solution, verbose)
            total_distance = 0
            for vehicle_id, solution in solutions.items(): 
                print("")
                print(f"Vehicle {vehicle_id}:")
                total_distance += self._print_decision_table(solution, verbose)
            print("{:<25} {:<10} {:>15} {:>15} {:>15}".format("Total all vehicle", "", total_distance, "", ""))
            print("-" * 85)
        else: 
            self.feasible = False

    def _split_solution_by_vehicle(self, solution, verbose=False):
        if solution is not None:
            if self.variant == "static":
                return {"1": solution}
            elif self.variant == "dynamic":
                return {"1": solution}
            elif self.variant == "dynamic_multiple":
            # split the solution into vehicle solutions
                solutions = {}
                total_distance = 0
                for vehicle in self.model.Vehicles: 
                    vehicle_solution = {}
                    for key, value in solution.items(): 
                        vehicle_id = key.split('_')[-1][1:]
                        if vehicle.get_id() == int(vehicle_id):
                            vehicle_solution.update({key: value})
                            total_distance += value
                    solutions[f"v{vehicle.get_id()}"] = vehicle_solution
                return solutions

    def _print_decision_table(self, solution, verbose=False): 
        print("-" * 85)
        print("{:<25} {:<10} {:>15} {:>15} {:>15}".format("Decision", "Move", "Distance", "Travel time", "Time step"))
        print("-" * 85)
        total_distance = 0
        total_travel_time = 0
        time_step = 1  
        for k, v in self.model.get_solution_distances(solution).items():
            decision, move, distance, travel_time = self._get_decision(k, v, verbose)
            if move is not None:
                print("{:<25} {:<10} {:>10} {:>15} {:>15}".format(decision, move, distance, travel_time, time_step))
            total_distance += v
            total_travel_time += travel_time
            time_step += travel_time
        print("-" * 85)
        print("{:<25} {:<10} {:>15} {:>15} {:>15}".format("Total", "", total_distance, total_travel_time, time_step))
        print("-" * 85)
        return total_distance

    def _find_lane(self, ap_id):
        for lane in self.Lanes: 
            if lane.get_ap_id() == int(ap_id): 
                return lane

    def _find_tier(self, lane, tier_id):
        for tier in lane.tiers: 
            if tier.get_id() == int(tier_id): 
                return tier

    def _get_travel_time(self, lane1, tier1, lane2, tier2, handling_time=False):
        """
        Get the travel time between two access points. We can not use the function of the model,
        as we only pass the decision variables to this class 
        """
        # handling_time = tier1 * self.instance.get_handling_time() + tier2 * self.instance.get_handling_time()
        # time = self.instance.get_warehouse().ap_distance[int(ap1)][int(ap2)]/self.instance.get_vehicle_speed() + handling_time
        # return int(max(1, self.instance.get_warehouse().ap_distance[int(ap1)][int(ap2)]/self.instance.get_vehicle_speed()))
        if lane1 == 'source' or lane1 == 'sink':
            l1 = lane1
            t1 = None
        else: 
            l1 = self._find_lane(lane1)
            t1 = self._find_tier(l1, tier1)
        if lane2 == 'sink' or lane2 == 'source':
            l2 = lane2
            t2 = None
        else:
            l2 = self._find_lane(lane2)
            t2 = self._find_tier(l2, tier2)
        return self.model.calculate_travel_time(l1, t1, l2, t2, handling_time)

    def _get_decision(self, k, v, verbose=False):
        """
        Get the decision from the decision variable and the distance from the objective function
        """
        decision = k.split("_")
        if decision[0] == "e":
            if decision[1][1:] == decision[3][1:] and decision[2][1:] == decision[4][1:] and not verbose:
                move = None
                travel_time = 1
            #continue    # Do not print the decision if the AMR does not move and verbose is False
            else:
                move = f"[{decision[1][1:]}, {decision[2][1:]}] \u21AA [{decision[3][1:]}, {decision[4][1:]}]"
                lane1 = decision[1][1:] 
                tier1 = decision[2][1:]
                lane2 = decision[3][1:]
                tier2 = decision[4][1:]
                travel_time = self._get_travel_time(lane1, tier1, lane2, tier2, False)
        elif decision[0] == "x":
            if decision[1][1:] == decision[3][1:] and decision[2][1:] == decision[4][1:] and not verbose:
                move = None
                travel_time = 1
            else: 
                move = f"[{decision[1][1:]}, {decision[2][1:]}] \u2192 [{decision[3][1:]}, {decision[4][1:]}]"
                lane1 = decision[1][1:]
                tier1 = decision[2][1:]
                lane2 = decision[3][1:]
                tier2 = decision[4][1:]
                travel_time = self._get_travel_time(lane1, tier1, lane2, tier2, True)
        elif decision[0] == "y":
            move = f"[{decision[1][1:]}, {decision[2][1:]}] \u2192 [{self.instance.get_warehouse().get_sink().get_ap_id()}, 1]"
            lane1 = decision[1][1:]
            tier1 = decision[2][1:]
            lane2 = 'sink'
            tier2 = None
            travel_time = self._get_travel_time(lane1, tier1, lane2, tier2, True)
        elif decision[0] == "z":
            move = f"[{self.instance.get_warehouse().get_source().get_ap_id()}, 1] \u2192 [{decision[1][1:]}, {decision[2][1:]}]"
            lane1 = decision[1][1:]
            tier1 = decision[2][1:]
            lane2 = 'source'
            tier2 = None
            travel_time = self._get_travel_time(lane1, tier1, lane2, tier2, True)
        return k, move, v, travel_time

    def save_results(self, filename: str):
        """
        Save the results of the test case to a file
        """
        if self.model.model.status == gp.GRB.OPTIMAL or self.model.model.SolCount > 0:
            self.results['objective_value'] = self.model.model.objVal
            self.results['runtime'] = round(self.model.model.runtime, 2)
            self.results['mipgap'] = self.mipgap
            decision_dict = {}
            solutions = self._split_solution_by_vehicle(self.model.get_solution())
            for vehicle, solution in solutions.items():
                timestep = 1
                vehicle_decision_dict = {}
                for k, v in self.model.get_solution_distances(solution).items():
                    decision, move, distance, travel_time = self._get_decision(k, v)
                # for k, v in solution.items():
                    # decision, move, distance, travel_time = self._get_decision(k, v)
                    if move is not None:
                        match = re.search(r"t(\d+)", decision)
                        time = match.group(1)
                        vehicle_decision_dict[time] = {"decision": decision, "move": move, "distance": distance, "travel_time": travel_time}
                    timestep += travel_time
                decision_dict[vehicle] = vehicle_decision_dict
            self.results['decisions'] = decision_dict
                # self.results[f"vehicle_{k}"] = vehicle_dict
            save_resultsBrr(filename, self)
            print(f"Results saved to {filename}")


if __name__ == "__main__": 
    from src.examples_gen.unit_load_gen import UnitLoadGenerator

    instance = Instance(
        layout_file="examples/Size_3x3_Layout_1x1_sink_source.csv",
        fill_level=0.8,
        max_p=0,
        height=1,
        seed=1,
        access_directions={"north": True, "east": True, "south": True, "west": True}, 
        exampleGenerator=UnitLoadGenerator(tw_length=15, fill_level=0.8, seed=1),
    )
    testCase = TestCaseBrr(instance=instance)
    # print(instance.get_warehouse().get_bays_state_int())
    # print(instance.get_warehouse().get_bays_state_ul(instance.unit_loads))
    print(instance.get_warehouse().get_virtual_lanes())
