#!/home/maxdisselnmeyer/multibay-reshuffling/.venv/bin/python
import sys
import os
sys.path.insert(0, os.path.abspath('./src'))

### This file is used to start the experiments for the buffer reshuffling and retrieval (BRR) problem


import argparse
import json # Add this import
from src.instance.instance_loader import InstanceLoader
from src.instance.instance import Instance
from src.examples_gen.unit_load_gen import UnitLoadGenerator
from src.test_cases.test_case_brr import TestCaseBrr
import gurobipy as gp

def create_paths(instance):
    file = instance.get_filename()
    fill_level = instance.get_fill_level()
    unit_loads = len(instance.get_unit_loads())
    access_directions = instance.get_access_directions()
    rs_max = instance.get_rs_max()
    as_max = instance.get_as_max()
    time_window_length = instance.get_time_window_length()
    speed = instance.get_vehicle_speed()
    handling_time = instance.get_handling_time()
    fleet_size = instance.get_fleet_size()

    # Common path structure
    common_structure = f"{file}/" \
        f"handling_time_{handling_time}/" \
        f"unit_loads_{unit_loads}/" \
        f"access_directions_" \
        f"{int(access_directions['north'])}" \
        f"{int(access_directions['east'])}" \
        f"{int(access_directions['south'])}" \
        f"{int(access_directions['west'])}/" \
        f"rs_max_{rs_max}/" \
        f"as_max_{as_max}/" \
        f"tw_length_{time_window_length}/" \
        f"speed_{speed}/" \
        f"fleet_size_{fleet_size}"

    input_path = f"experiments/inputsBRR/{common_structure}/"
    os.makedirs(input_path, exist_ok=True)

    result_path = f"experiments/resultsBRR/{common_structure}/"
    os.makedirs(result_path, exist_ok=True)

    feasibility_path = f"experiments/feasibleBRR/{common_structure}/" # New path for feasibility
    os.makedirs(feasibility_path, exist_ok=True) # Create the directory

    hash_path = f"experiments/hashesBRR/"
    os.makedirs(hash_path, exist_ok=True)

    return input_path, result_path, hash_path, feasibility_path # Return the new path


# TODO: parameteroptimization 
def generate_instances():
    files = [
        # 'Size_3x3_Layout_2x2_sink_source',
        'Size_3x3_Layout_1x1_sink_source',
        # 'Size_4x4_Layout_1x1_sink_source',
        # 'Size_5x5_Layout_1x1_sink_source',
        # 'Size_6x6_Layout_1x1_sink_source',
        # 'Size_7x7_Layout_1x1_sink_source',
        # 'Size_8x8_Layout_1x1_sink_source',
        # 'Size_3x3_Layout_2x2_sink_source',
        # 'Size_3x3_Layout_3x3_sink_source',
        # 'Size_4x4_Layout_3x3_sink_source',
        # 'Size_4x4_Layout_2x2_sink_source',
        # 'Size_3x3_Layout_2x2_sink_source',
        # 'Size_3x3_Layout_3x3_sink_source',
        # 'Size_3x3_Layout_1x1_sink',
    ]
    ad = [
        {"north" : True, "east" : True, "south" : True, "west" : True},
        # {"north" : False, "east" : False, "south" : True, "west" : False},
        # {"north" : False, "east" : False, "south" : True, "west" : True},
        # {"north" : False, "east" : True, "south" : True, "west" : False}, 
        # {"north" : True, "east" : False, "south" : True, "west" : False}, 
    ]
    # seeds = [i for i in range(10)]
    seeds = [1]
    # fill_levels = [i/10 for i in range(5, 11)]
    fill_levels = [0.1]
    # time_window_lengths = [i for i in range(30, 61, 10)]
    time_window_lengths = [30]
    vehicle_speeds = [1]
    fleet_sizes = [1]
    rs_maxes = [100]
    as_max = 50
    for file in files: 
        for time_window_length in time_window_lengths:
            for vehicle_speed in vehicle_speeds:
                for rs_max in rs_maxes:
                    for fill_level in fill_levels:
                        for fleet_size in fleet_sizes:
                            for seed in seeds: 
                                for access_directions in ad:
                                    layout_file = f"examples/{file}.csv"
                                    instance = Instance(
                                        layout_file=layout_file,
                                        seed=seed,
                                        access_directions=access_directions, 
                                        max_p=0, 
                                        fill_level=fill_level,
                                        fleet_size=fleet_size,
                                        vehicle_speed=vehicle_speed,
                                        handling_time=1,
                                        exampleGenerator=UnitLoadGenerator(tw_length=time_window_length, fill_level=fill_level, seed=seed, rs_max=rs_max, as_max=as_max, source=True),
                                        rs_max=rs_max,
                                        as_max=as_max,
                                        time_window_length=time_window_length,
                                    )
                                    yield instance


def solve_instance(instance, input_path, result_path, hash_path, feasibility_path, verbose): # Add feasibility_path
    """Solves a single instance, handling potential Gurobi errors."""

    instance_path = f"{input_path}{instance.get_seed()}.json"
    result_file_path = f"{result_path}{instance.get_seed()}.json" # Renamed to avoid conflict
    feasibility_file_path = f"{feasibility_path}{instance.get_seed()}.json" # Path for the feasibility json
    instance.save_instance(instance_path)

    if not instance.check_if_solved(hash_path, instance_path):
        print("Solving instance: ", instance)
        feasibility_status = {} # Initialize feasibility status dict
        try:
            test_case = TestCaseBrr(instance=instance, variant="dynamic_multiple", verbose=verbose)
            is_feasible = test_case.feasible # Store feasibility
            feasibility_status = {'feasible': is_feasible} # Set status

            if is_feasible:
                instance.save_hash('feasible', hash_path, instance_path)
                result_file_list = f"{hash_path}/results.txt" # Renamed for clarity
                with open(result_file_list, "a") as file:
                    file.write(result_file_path + "\n")
                test_case.save_results(result_file_path)
            else:
                instance.save_hash('infeasible', hash_path, instance_path)

        except gp.GurobiError as e:
            print(f"Gurobi Error in instance {instance.get_seed()}: {e}")
            feasibility_status = {'feasible': None, 'error': 'GurobiError', 'message': str(e)} # Record error
        except Exception as e:
            print(f"Unexpected error in instance {instance.get_seed()}: {e}")
            feasibility_status = {'feasible': None, 'error': 'Exception', 'message': str(e)} # Record error
        finally:
            # Save feasibility status regardless of outcome (unless already solved)
            try:
                with open(feasibility_file_path, 'w') as f:
                    json.dump(feasibility_status, f, indent=4) # Save as JSON
                print(f"Saved feasibility status to {feasibility_file_path}")
            except Exception as e:
                print(f"Error saving feasibility status for instance {instance.get_seed()}: {e}")
            # Add any other cleanup code here if necessary
            pass
    else:
        print("Instance already solved. Skipping.")
        # Optionally, you could still check if the feasibility file exists and create it
        # if needed, based on the hash file content, but the current logic skips entirely.

if __name__ == "__main__":
    """
    parser is used to optionally pass a json file containing the instance information when starting the script
    """
    instances = []    # Use this list to pass the instances to the rest of the code

    # If an instance file is passed during call of the file
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, help="Path to the instance json file")
    parser.add_argument("--verbose", action="store_true", help="Prints the results of the model")
    parser.add_argument("--check-existing", action="store_true", help="Check existing instances in inputsBRR instead of generating new ones") # Add this flag
    args = parser.parse_args()
    verbose = args.verbose
    instances = [] # Initialize instances list

    if args.instance:
        print(f"Loading specific instance: {args.instance}")
        try:
            # Create the loader
            instanceLoader = InstanceLoader(args.instance)
            # Pass the loader object directly to the Instance constructor
            instance = Instance(instanceLoader=instanceLoader)
            instances.append(instance)
        # Keep broader exception handling for loader/instance issues
        except Exception as e:
            print(f"Error loading or initializing instance from {args.instance}: {e}")
            sys.exit(1)

    # If no specific instance file is passed
    else:
        # Decide whether to generate or load based on the flag
        if args.check_existing:
            instances_to_process = []
            base_input_path = "experiments/inputsBRR"
            print(f"Checking existing instances flag set. Scanning {base_input_path}...")
            if not os.path.isdir(base_input_path):
                print(f"Error: Directory {base_input_path} not found.")
                sys.exit(1)

            for root, dirs, files in os.walk(base_input_path):
                for filename in files:
                    if filename.endswith(".json"):
                        instance_json_path = os.path.join(root, filename)
                        try:
                            # Create the loader
                            loader = InstanceLoader(instance_json_path)
                            # Pass the loader object directly to the Instance constructor
                            instance = Instance(instanceLoader=loader)
                            instances_to_process.append(instance)
                        # Keep broader exception handling for loader/instance issues
                        except Exception as e:
                            print(f"Warning: Error loading or initializing instance from {instance_json_path}: {e}. Skipping file.")

            print(f"Found {len(instances_to_process)} existing instances to process.")
            instances = instances_to_process
        else:
            print("Generating new instances...")
            # This uses the 'else' branch of Instance.__init__ correctly with exampleGenerator
            instances = generate_instances()

    # Process the collected instances (either loaded or generated)
    instance_iterator = instances # Handle list or generator

    # Check if any instances were loaded/generated before proceeding
    # A simple way is to proceed and let the loop handle it, adding a counter.
    print(f"\nProcessing instances...")
    count = 0
    for instance in instance_iterator:
        count += 1
        print(f"\n--- Processing instance {count} (Seed: {instance.get_seed()}) ---")
        try:
            # Unpack the new feasibility_path
            input_path, result_path, hash_path, feasibility_path = create_paths(instance)
            # Pass feasibility_path to the solve function
            solve_instance(instance, input_path, result_path, hash_path, feasibility_path, verbose)
        except Exception as e:
            # Catch potential errors during path creation or solving for a specific instance
            print(f"Error processing instance with seed {instance.get_seed()}: {e}")

    if count == 0:
        print("No instances were generated or loaded to process.")

    print("\n--- Finished processing all instances ---")
