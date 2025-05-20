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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the result of the BRR algorithm")
    parser.add_argument("--solution", type=str, required=True, help="Path to the solution file")
    args = parser.parse_args()
