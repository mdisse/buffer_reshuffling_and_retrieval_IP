"""
TWVRP Integration Module

This module provides a unified interface for different TWVRP solver implementations.
Currently supports:
- OR-Tools-based solver (twvrp_ortools.py) - Flexible constraints, good for complex time windows

Both solvers take the same inputs and produce compatible outputs, allowing easy switching
between implementations based on problem characteristics and performance requirements.
"""

from typing import List, Dict, Optional


def solve_twvrp(buffer, moves: List[Dict], num_vehicles: int = 1, instance=None, 
                time_limit=None, solver: str = 'ortools', verbose: bool = False) -> Dict:
    """
    Solve TWVRP using the specified solver.
    
    Args:
        buffer: Buffer object for distance calculations
        moves: List of moves from A* search
        num_vehicles: Number of available vehicles
        instance: Instance object for handling time access
        time_limit: Maximum time in seconds for VRP solving (None for no limit)
        solver: Which solver to use ('pyvrp' or 'ortools')
        
    Returns:
        Dictionary containing the VRP solution
        
    Raises:
        ValueError: If an unknown solver is specified
        ImportError: If the required solver package is not available
    """
    if solver.lower() == 'ortools':
        try:
            from .twvrp_ortools import solve_twvrp_with_ortools
            return solve_twvrp_with_ortools(buffer, moves, num_vehicles, instance, time_limit, verbose=verbose)
        except ImportError as e:
            raise ImportError(f"OR-Tools not available: {e}")
    
    else:
        raise ValueError(f"Unknown solver '{solver}'. Available solvers: 'pyvrp', 'ortools'")


def solve_twvrp_with_ortools(buffer, moves: List[Dict], num_vehicles: int = 1, 
                            instance=None, time_limit=None, verbose: bool = False) -> Dict:
    """Solve TWVRP using OR-Tools."""
    return solve_twvrp(buffer, moves, num_vehicles, instance, time_limit, solver='ortools', verbose=verbose)