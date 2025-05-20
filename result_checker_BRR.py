import sys 
import os 
sys.path.insert(0, os.path.abspath('./src'))

import argparse
import json
from src.instance.instance_loader import InstanceLoader
from src.instance.instance import Instance
from src.examples_gen.unit_load_gen import UnitLoadGenerator 
from src.test_cases.test_case_brr import TestCaseBrr
import gurobipy as gp 

def get_decisions(solution):
    """ 
    Extract decisions from the solution.
    """
    all_decision_values = []
    results = solution.get("results", {})
    decisions_by_vehicle = results.get("decisions", {})
    for vehicle_id, vehicle_decisions in decisions_by_vehicle.items():
        for timestamp, decision_details in vehicle_decisions.items():
            decision_value = decision_details.get("decision")
            if decision_value:
                all_decision_values.append(decision_value)
    return all_decision_values

def check_instance(instance, solution, verbose=False): 
    """
    Check if the solution is valid for the given instance.
    """
    print("Checking instance: ", instance)
    try: 
        decisions = get_decisions(solution)
        test_case = TestCaseBrr(instance=instance, variant="dynamic_multiple", decisions=decisions, verbose=verbose)
        return True
    except Exception as e:
        print(f"Error creating test case: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the result of the BRR algorithm")
    parser.add_argument("--solution_path", type=str, required=True, help="Path to the solution file")
    args = parser.parse_args()

    if args.solution_path.endswith(".json"):
        with open(args.solution_path, "r") as f:
            solution = json.load(f)
    else: 
        raise ValueError("The solution file must be a json file")
    try: 
        instance_path = args.solution_path.replace("resultsBRR", "inputsBRR")
        instanceLoader = InstanceLoader(instance_path)
        instance = Instance(instanceLoader=instanceLoader)
    except Exception as e:
        print(f"Error loading instance: {e}")
        sys.exit(1)
    result_legit = check_instance(instance, solution)
    if result_legit: 
        print("Solution is valid.")
    else: 
        print("Solution is not valid.")