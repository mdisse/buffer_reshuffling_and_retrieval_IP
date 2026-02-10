"""
TWVRP Integration Module

This module provides a unified interface for different TWVRP solver implementations.
Currently supports:
- OR-Tools VRP solver (twvrp_ortools.py) - Routing-based approach with flexible constraints
- OR-Tools CP-SAT Scheduling solver (twvrp_scheduling.py) - Constraint programming approach,
  better for tight time windows and complex precedence constraints

All solvers take the same inputs and produce compatible outputs, allowing easy switching
between implementations based on problem characteristics and performance requirements.
"""

from typing import List, Dict, Optional


def solve_twvrp(buffer, moves: List[Dict], num_vehicles: int = 1, instance=None, 
                time_limit=None, solver: str = 'scheduling', verbose: bool = False) -> Dict:
    """
    Solve TWVRP using the specified solver.
    
    Args:
        buffer: Buffer object for distance calculations
        moves: List of moves from A* search
        num_vehicles: Number of available vehicles
        instance: Instance object for handling time access
        time_limit: Maximum time in seconds for VRP solving (None for no limit)
        solver: Which solver to use ('ortools', 'scheduling')
        
    Returns:
        Dictionary containing the VRP solution
        
    Raises:
        ValueError: If an unknown solver is specified
        ImportError: If the required solver package is not available
    """
    if solver.lower() == 'scheduling':
        try:
            from .scheduling import solve_twvrp_with_scheduling
            return solve_twvrp_with_scheduling(buffer, moves, num_vehicles, instance, time_limit, verbose=verbose)
        except ImportError as e:
            raise ImportError(f"OR-Tools CP-SAT not available: {e}")
    else:
        raise ValueError(f"Unknown solver '{solver}'. Available solvers: 'scheduling'")


def solve_twvrp_with_scheduling(buffer, moves: List[Dict], num_vehicles: int = 1,
                                instance=None, time_limit=None, verbose: bool = False) -> Dict:
    """Solve TWVRP using OR-Tools CP-SAT Scheduling solver."""
    return solve_twvrp(buffer, moves, num_vehicles, instance, time_limit, solver='scheduling', verbose=verbose)
