#!/usr/bin/env python3
"""
Test script to verify VRP failure output messages are accurate.
"""

import json
from src.test_cases.test_case_brr import TestCaseBrr

# Create instance 3 manually
instance_data = {
    "fleet_size": 2,
    "amr_speed": 1.0,
    "handling_time": 1,
    "buffer": {
        "n_rows": 3,
        "n_cols": 3,
        "lane_depth": 3,
        "access_directions": "1010"
    },
    "unit_loads": [
        {
            "id": 1,
            "retrieval_start": 173,
            "retrieval_end": 194,
            "storage_start": 45,
            "storage_end": 68
        },
        {
            "id": 2,
            "retrieval_start": 124,
            "retrieval_end": 147,
            "storage_start": 19,
            "storage_end": 36
        },
        {
            "id": 3,
            "retrieval_start": 84,
            "retrieval_end": 103,
            "storage_start": 7,
            "storage_end": 23
        },
        {
            "id": 4,
            "retrieval_start": 130,
            "retrieval_end": 153,
            "storage_start": 74,
            "storage_end": 99
        },
        {
            "id": 5,
            "retrieval_start": 104,
            "retrieval_end": 121,
            "storage_start": 0,
            "storage_end": 22
        }
    ]
}

print("="*80)
print("Testing VRP Failure Output Messages")
print("="*80)

# Create test case
test_case = TestCaseBrr(
    instance_data=instance_data,
    instance_name="test_instance_3"
)

# Run A* to get moves
print("\n1. Running A* to generate move sequence...")
test_case.solve_with_heuristic(
    max_expansions=5000,
    a_star_time_limit=60,
    vrp_time_limit=60
)

# The output should now correctly identify VRP failure
print("\n2. Checking final solution output...")
test_case.print_heuristic_solution()

print("\n" + "="*80)
print("Test Complete")
print("="*80)
print("\nExpected behavior:")
print("  - Should show '❌ VRP FAILED' when solver fails")
print("  - Should NOT show '✓ Heuristic found optimal solution!' on failure")
print("  - Should show error message from VRP solver")
