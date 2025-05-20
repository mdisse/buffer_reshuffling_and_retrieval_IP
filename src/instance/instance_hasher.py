import hashlib
import json
import os

def hash_instance(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    sorted_data = json.dumps(data, sort_keys=True)
    return hashlib.sha256(sorted_data.encode()).hexdigest()

def collect_hashes(input_path):
    files = [f"{input_path}/feasible.txt", f"{input_path}/infeasible.txt"]
    hashes = set()

    for file in files:
        if not os.path.exists(file):
            open(file, 'w').close()

        with open(file, 'r') as f:
            for line in f:
                hashes.add(line.strip())
    return hashes