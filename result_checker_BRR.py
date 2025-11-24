import sys 
import os 
sys.path.insert(0, os.path.abspath('./src'))

import argparse
import json
from src.instance.instance_loader import InstanceLoader
from src.instance.instance import Instance
from src.test_cases.test_case_brr import TestCaseBrr
import re

def get_decisions(solution):
    """ 
    Extract decisions from the solution.
    Requires the solution to be a dictionary with a "results" key
    that contains a "decisions" key, which is a dictionary mapping vehicle IDs to their decisions.
    Each decision is expected to be a dictionary with a "decision" key.
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

def change_decisions(decisions_list: list[str]) -> list[str]:
    """
    Provides a text-based interactive way to modify a list of decision strings.
    """
    current_decisions = list(decisions_list) # Make a copy

    while True:
        print("\\n--- Current Decisions ---")
        if not current_decisions:
            print("  (No decisions in the list)")
        else:
            for i, decision in enumerate(current_decisions):
                print(f"  {i + 1}: {decision}")
        print("-------------------------")
        
        user_input = input("Enter action ('help' for commands, 'done' to finish): ").strip()
        # Split into command, and up to two arguments (e.g., index and value)
        parts = user_input.split(maxsplit=2) 
        
        command = parts[0].lower() if parts else ""
        
        if command == "done":
            break
        elif command == "help":
            print("\\nAvailable commands:")
            print("  add <decision_string>          - Add a new decision.")
            print("  change <index> <new_decision>  - Change the decision at the specified index (1-based).")
            print("  delete <index>                 - Delete the decision at the specified index (1-based).")
            print("  done                           - Exit and return the modified list.")
            print("  help                           - Show this help message.")
        elif command == "add":
            if len(parts) > 1:
                # The rest of the input is the decision string
                decision_to_add = user_input.split(maxsplit=1)[1]
                current_decisions.append(decision_to_add)
                print(f"Added: '{decision_to_add}'")
            else:
                print("Error: 'add' command requires a decision string. Usage: add <decision_string>")
        elif command in ["change", "delete"]:
            if len(parts) < 2:
                print(f"Error: '{command}' command requires an index. Usage: {command} <index> [new_value_for_change]")
                continue
            try:
                index = int(parts[1]) - 1 # Convert to 0-based index
                
                if not (0 <= index < len(current_decisions)):
                    print("Error: Index out of bounds.")
                    continue
                
                if command == "delete":
                    deleted_item = current_decisions.pop(index)
                    print(f"Deleted: '{deleted_item}'")
                elif command == "change":
                    if len(parts) < 3:
                        print("Error: 'change' command requires an index and a new decision string. Usage: change <index> <new_decision_string>")
                        continue
                    old_value = current_decisions[index]
                    new_value = parts[2]
                    current_decisions[index] = new_value
                    print(f"Changed index {index + 1}: '{old_value}' -> '{new_value}'")
            except ValueError:
                print("Error: Invalid index. Index must be a number.")
        elif not command: # Empty input
            continue
        else:
            print(f"Unknown command: '{command}'. Type 'help' for available commands.")
            
    print("\\n--- Final Decisions ---")
    if not current_decisions:
        print("  (No decisions in the list)")
    else:
        for i, decision in enumerate(current_decisions):
            print(f"  {i + 1}: {decision}")
    print("-----------------------")
    return current_decisions

def create_solution(decisions):
    """
    Create a solution dictionary from the list of decisions.
    The solution is structured to match the expected format for BRR.
    """
    print(decisions)
    solution = {}
    for decision in decisions:
        solution[decision] = 1
    return solution

def check_instance(instance, solution, change=False, verbose=False): 
    """
    Check if the solution is valid for the given instance.
    """
    print("Checking instance: ", instance)
    try: 
        decisions = get_decisions(solution)
        if change:
            decisions = change_decisions(decisions)
        solution = create_solution(decisions)
        testcase = TestCaseBrr(instance=instance, variant="dynamic_multiple", solution=solution, verbose=verbose, mode="check")
        status = testcase.check_solution()
        print(status)
    except Exception as e:
        print(f"Error creating test case: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the result of the BRR algorithm")
    parser.add_argument("--solution-path", type=str, required=True, help="Path to the solution file")
    args = parser.parse_args()

    if args.solution_path.endswith(".json"):
        solution_path = os.path.abspath(args.solution_path)
        with open(args.solution_path, "r") as f:
            solution = json.load(f)
    else: 
        raise ValueError("The solution file must be a json file")
    try: 
        instance_path = solution_path.replace("resultsBRR", "inputsBRR")
        # Remove _heuristic suffix if present
        instance_path = instance_path.replace("_heuristic.json", ".json")
        instanceLoader = InstanceLoader(instance_path)
        instance = Instance(instanceLoader=instanceLoader)
    except FileNotFoundError as e:
        try: 
            instance_path = solution_path.replace("resultsBRR", "inputsBRR")
            # Remove _heuristic suffix and fleet_size directory
            instance_path = instance_path.replace("_heuristic.json", ".json")
            instance_path = re.sub(r"fleet_size_\d+/", "", instance_path)
            instanceLoader = InstanceLoader(instance_path)
            instance = Instance(instanceLoader=instanceLoader)
        except FileNotFoundError as e:
            print(f"Error loading instance: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading instance: {e}")
        sys.exit(1)
    check_instance(instance, solution)