import sys
import os
wd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(wd, '../..'))

from src.instance.instance import Instance
from src.integer_programming.static_model import StaticModel
from src.integer_programming.dynamic_multiple_model import DynamicMultipleModel
from src.test_cases.writer_functions import save_resultsBrr
import gurobipy as gp
import gurobi_modelanalyzer as gma
import re
import copy
import time
from src.heuristics.astar import AStarSolver

class TestCaseBrr: 
    """
    This class is used to test the BRR problem
    """
    def __init__(self, instance: Instance, variant="static", solution=False, verbose=False, mode="solve") -> None:
        """
        Modes: ['solve', 'check', "heuristic"]
        """
        self.instance = instance
        self.variant = variant
        self.feasible = None
        self.results = {}
        self.Lanes = self.instance.get_buffer().get_virtual_lanes()
        self.verbose = verbose
        if mode == "heuristic":
            self.task_queue = []
            self.move_sequence = [] # to be populated by astar 
            self.amr_assignments = {} # to be populated by vrp
            self.model = None 
            # Runtime tracking
            self.heuristic_start_time = None
            self.heuristic_end_time = None
            self.heuristic_runtime = None
            # Solution comparison
            self.heuristic_objective = None
            self.gurobi_objective = None
            self.mip_gap = None 
        else: 
            if self.variant == "static":
                self.model = StaticModel(self.instance, verbose=verbose)
            elif self.variant == "dynamic":
                sys.exit("Dynamic model not implemented yet")
            elif self.variant == "dynamic_multiple":
                # self.model = DynamicMultipleModel(self.instance, decisions=decisions, verbose=verbose)
                self.model = DynamicMultipleModel(self.instance, verbose=verbose)
            else:
                ValueError("The choosen variant is not implemented")
    
        if mode == "check":
            if self.model and hasattr(self.model, 'model') and self.model.model is not None:
                self.sc = gma.SolCheck(self.model.model)
                # Build solution dictionary with error checking
                self.sol = {}
                for k, v in solution.items():
                    var = self.model.model.getVarByName(k)
                    if var is not None:
                        self.sol[var] = v
                    elif self.verbose:
                        print(f"Warning: Variable '{k}' not found in model")
                
                if not self.sol and self.verbose:
                    print("Warning: No valid variables found for solution checking")
            else:
                raise ValueError("Model not properly initialized for constraint checking")
        elif mode == "solve":
            self.model.solve()
            self.print_inital_state()
            self.print_solution(self.verbose)

        if self.verbose and mode != "heuristic": 
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
        # time = self.instance.get_buffer().ap_distance[int(ap1)][int(ap2)]/self.instance.get_vehicle_speed() + handling_time
        # return int(max(1, self.instance.get_buffer().ap_distance[int(ap1)][int(ap2)]/self.instance.get_vehicle_speed()))
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
            move = f"[{decision[1][1:]}, {decision[2][1:]}] \u2192 [{self.instance.get_buffer().get_sink().get_ap_id()}, 1]"
            lane1 = decision[1][1:]
            tier1 = decision[2][1:]
            lane2 = 'sink'
            tier2 = None
            travel_time = self._get_travel_time(lane1, tier1, lane2, tier2, True)
        elif decision[0] == "z":
            move = f"[{self.instance.get_buffer().get_source().get_ap_id()}, 1] \u2192 [{decision[1][1:]}, {decision[2][1:]}]"
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
        
    def check_solution(self):
        self.sc.test_sol(self.sol)
        return self.sc.Status

    def set_task_queue(self, task_queue):
        """Stores the sorted list of all tasks (storage and retrieval)."""
        print("\n--- Step 1: Task Queue Created ---")
        self.task_queue = task_queue
        if self.verbose:
            print("  Chronological Task Order:")
            for task_ul in self.task_queue:
                task_type = "STORAGE" if "_mock" in str(task_ul.id) else "RETRIEVAL"
                print(f"    {task_ul.get_priority()}. UL {task_ul.id} (Task: {task_type}), Due: {task_ul.retrieval_end}")
        print("-" * 85)

    def solve_heuristic_astar(self, time_limit=120):
        """
        Step 2: Solve using A* algorithm with infinite sink/source concept.
        
        Args:
            time_limit: Maximum time in seconds for A* search (None for no limit)
        """
        print("\n--- Step 2: A* for All Tasks ---")
        
        # We operate on a deep copy of the buffer state
        initial_buffer_state = copy.deepcopy(self.instance.get_buffer())
        
        # Extract all unit load IDs from the task queue
        all_unit_loads = set()
        for task in self.task_queue:
            if "_mock" in str(task.id):
                # Storage task - extract real unit load ID
                real_ul_id = task.real_ul_id
                all_unit_loads.add(real_ul_id)
            else:
                # Retrieval task - use task ID directly
                all_unit_loads.add(task.id)
        
        # Use the A* solver with infinite sink/source concept
        astar_solver = AStarSolver(
            initial_buffer_state=initial_buffer_state,
            all_unit_loads=all_unit_loads,
            dist_matrix=self.instance.get_buffer().ap_distance,
            handling_time=self.instance.get_handling_time(),
            instance=self.instance,
            verbose=self.verbose,
            time_limit=time_limit,
            task_queue=self.task_queue
        )
        
        # Get solution from A*
        move_sequence, astar_states, ul_priority_map = astar_solver.solve()

        # Store the detailed A* states and priority map if they exist
        if astar_states:
            self.results['astar_solution_states'] = astar_states
        if ul_priority_map:
            self.results['ul_priority_map'] = ul_priority_map

        if move_sequence:
            print(f"\n  A* complete. Found solution with {len(move_sequence)} moves.")

            if self.verbose:
                print(f"  First {len(move_sequence)} moves:")
                for j, move in enumerate(move_sequence[:50]):
                    to_pos_repr = move.get('to_pos').ap_id if hasattr(move.get('to_pos'), 'ap_id') else move.get('to_pos')
                    print(f"    {j+1}. {move['type']} UL {move['ul_id']} to {to_pos_repr}")
                if len(move_sequence) > 50:
                    print(f"    ... and {len(move_sequence) - 50} more moves")

            # Set this as the current sequence and try VRP
            self.move_sequence = move_sequence
        else:
            print("  ERROR: A* could not find any solution!")
            self.move_sequence = []
    
    def solve_heuristic_vrp(self, time_limit=None):
        """
        Assigns the generated move sequence to available AMRs using PyVRP.
        
        Args:
            time_limit: Maximum time in seconds for VRP solving (None for no limit)
        """
        print("\n--- Step 3: VRP for AMR Assignment ---")
        
        if not self.move_sequence:
            print("  ERROR: No moves to assign to AMRs!")
            self.amr_assignments = {}
            return self.amr_assignments
        
        # Import the TWVRP solver
        from src.heuristics.twvrp import solve_twvrp
        
        # Number of available AMRs (vehicles) from instance
        num_vehicles = self.instance.get_fleet_size()
        
        # Solve the TWVRP
        buffer_state = self.instance.get_buffer()
        vrp_solution = solve_twvrp(
            buffer=buffer_state,
            moves=self.move_sequence,
            num_vehicles=num_vehicles,
            instance=self.instance,
            time_limit=time_limit,
            solver='ortools', 
            verbose=self.verbose
        )
        
        # Store both the AMR assignments and the full VRP solution
        self.amr_assignments = vrp_solution
        self.vrp_solution = vrp_solution  # Store full solution including warnings
        if self.verbose:
            print(self.amr_assignments)

        print(f"  VRP complete. Assigned {len(self.move_sequence)} moves to {num_vehicles} AMR(s).")
        
        if self.verbose:
            print("  AMR assignments:")
            for vehicle in self.amr_assignments['vehicles']:
                print(f"    AMR {vehicle['vehicle_id']}: {len(vehicle['moves'])} moves, "
                      f"total distance: {vehicle['total_distance']:.1f}")
        
        print("-" * 85)
        return self.amr_assignments
        
    def print_heuristic_solution(self):
        """Prints the final plan from the heuristic with runtime and comparison."""
        print("\n--- Final Heuristic Solution ---")
        
        if not self.amr_assignments:
            print("  No solution available!")
            return
        
        # Calculate objective
        self.calculate_heuristic_objective()
        
        # Print solution details
        total_distance = self.amr_assignments.get('total_distance', 0)
        total_time = self.amr_assignments.get('total_time', 0)
        
        print(f"  Total distance: {total_distance:.1f}")
        print(f"  Total time: {total_time}")
        print(f"  Number of AMRs used: {len(self.amr_assignments['vehicles'])}")
        
        # Print runtime information
        if self.heuristic_runtime is not None:
            print(f"  Heuristic runtime: {self.heuristic_runtime:.3f} seconds")
        
        # Print comparison with Gurobi if available
        if self.mip_gap is not None:
            if self.mip_gap < float('inf'):
                print(f"  Gurobi optimal: {self.gurobi_objective:.1f}")
                print(f"  Heuristic objective: {self.heuristic_objective:.1f}")
                print(f"  MIP gap: {self.mip_gap:.2f}%")
                if self.mip_gap <= 0:
                    print("  ✓ Heuristic found optimal solution!")
                elif self.mip_gap <= 5:
                    print("  ✓ Heuristic solution is very good (≤5% gap)")
                elif self.mip_gap <= 10:
                    print("  ⚠ Heuristic solution is acceptable (≤10% gap)")
                else:
                    print("  ⚠ Heuristic solution has significant gap (>10%)")
            else:
                print("  ⚠ Could not compare with Gurobi solution")
        
        # Print detailed AMR plans
        for vehicle in self.amr_assignments['vehicles']:
            print(f"\n  AMR {vehicle['vehicle_id']} plan:")
            print(f"    Distance: {vehicle['total_distance']:.1f}")
            print(f"    Time: {vehicle['total_time']}")
            print(f"    Moves:")
            
            for i, move in enumerate(vehicle['moves']):
                from_loc = move['from_location']
                to_loc = move['to_location']
                ul_id = move['ul_id']
                move_type = move['move_type']
                distance = move['travel_distance']
                empty_travel = move.get('empty_travel_distance', 0)
                
                if move_type == 'empty':
                    print(f"      {i+1}. EMPTY TRAVEL: {from_loc} → {to_loc} (distance: {distance:.1f})")
                else:
                    if empty_travel > 0:
                        print(f"      {i+1}a. EMPTY TRAVEL to {from_loc} (distance: {empty_travel:.1f})")
                    print(f"      {i+1}b. {move_type.upper()} UL {ul_id}: {from_loc} → {to_loc} (distance: {distance:.1f})")
        
        print("-" * 85)
    
    def start_heuristic_timer(self):
        """Start timing the heuristic solution process."""
        self.heuristic_start_time = time.time()
        print(f"--- Heuristic Started at {time.strftime('%H:%M:%S')} ---")
    
    def end_heuristic_timer(self):
        """End timing and calculate runtime."""
        self.heuristic_end_time = time.time()
        self.heuristic_runtime = self.heuristic_end_time - self.heuristic_start_time
        print(f"--- Heuristic Completed in {self.heuristic_runtime:.3f} seconds ---")
    
    def calculate_heuristic_objective(self):
        """Calculate the objective value of the heuristic solution using real Gurobi-style distances."""
        if self.amr_assignments:
            # First, try to get corrected objective from translated decisions if available
            corrected_objective = self._get_corrected_objective_from_decisions()
            if corrected_objective is not None:
                self.heuristic_objective = corrected_objective
            else:
                # Fallback to VRP solver's calculated distance
                self.heuristic_objective = self.amr_assignments.get('total_distance', 0)
        else:
            self.heuristic_objective = float('inf')
        return self.heuristic_objective
    
    def _get_corrected_objective_from_decisions(self):
        """Calculate objective using the same distance calculation as Gurobi models."""
        if not self.amr_assignments or 'vehicles' not in self.amr_assignments:
            return None
        
        # Import here to avoid circular imports
        from src.test_cases.writer_functions import translate_heuristic_decisions_simple
        
        try:
            # Translate to get real distances
            translated_decisions = translate_heuristic_decisions_simple(self.amr_assignments, self.instance)
            
            # Sum up all the real distances
            total_real_distance = 0
            for vehicle_key, vehicle_decisions in translated_decisions.items():
                for time_key, decision_data in vehicle_decisions.items():
                    total_real_distance += decision_data.get('distance', 0)
            
            return total_real_distance
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate corrected objective: {e}")
            return None
    
    def compare_with_gurobi(self, gurobi_objective: float):
        """Compare heuristic solution with Gurobi optimal solution."""
        self.gurobi_objective = gurobi_objective
        
        if self.heuristic_objective is None:
            self.calculate_heuristic_objective()
        
        if self.gurobi_objective > 0 and self.heuristic_objective < float('inf'):
            # Calculate MIP gap: (heuristic - optimal) / optimal * 100
            self.mip_gap = ((self.heuristic_objective - self.gurobi_objective) / self.gurobi_objective) * 100
        else:
            self.mip_gap = float('inf')
        
        return self.mip_gap
    
    def save_heuristic_results(self, instance_file_path: str, fleet_size_override=None):
        """Save heuristic results to a file matching the Gurobi format."""
        from src.test_cases.writer_functions import save_heuristic_results, generate_heuristic_filename
        
        # Generate the output filename, using fleet size override if provided
        output_filename = generate_heuristic_filename(instance_file_path, fleet_size_override)
        
        # Save the results
        save_heuristic_results(output_filename, self)
        
        # Always print where the file was saved
        print(f"Heuristic results saved to: {output_filename}")
        
        # Add to tracking file
        self._add_to_tracking_file(output_filename)
        
        return output_filename

    def _add_to_tracking_file(self, result_file_path):
        """Add the result file path to the tracking file."""
        import os
        from datetime import datetime
        
        # Create the hashesBRR directory if it doesn't exist
        tracking_dir = "experiments/hashesBRR"
        os.makedirs(tracking_dir, exist_ok=True)
        
        # Path to the tracking file
        tracking_file = os.path.join(tracking_dir, "results_heuristic.txt")
        
        # Create timestamp and extract instance info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract some instance info from the path for easier tracking
        instance_info = "unknown"
        if "unit_loads_" in result_file_path and "fleet_size_" in result_file_path:
            try:
                # Extract unit loads and fleet size
                parts = result_file_path.split('/')
                unit_loads = next((p for p in parts if p.startswith('unit_loads_')), 'unknown')
                fleet_size = next((p for p in parts if p.startswith('fleet_size_')), 'unknown')
                instance_info = f"{unit_loads}, {fleet_size}"
            except:
                instance_info = "parse_error"
        
        # Append the result file path to the tracking file (just the path to make it clickable)
        try:
            with open(tracking_file, 'a') as f:
                f.write(f"{result_file_path}\n")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not write to tracking file: {e}")

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
    print(instance.get_buffer().get_virtual_lanes())
