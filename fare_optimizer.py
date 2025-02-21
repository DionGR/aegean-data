import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable
from datetime import datetime

class FareOptimizer:
    def __init__(self, passenger_predictor: Callable[[float], float], max_passengers: int = 1000):
        """
        Initialize the fare optimizer.
        
        Args:
            passenger_predictor: A function that takes AVG_FARE as input and returns predicted PASSENGERS
            max_passengers: Maximum allowed number of passengers (default: 1000)
        """
        self.passenger_predictor = passenger_predictor
        self.max_passengers = max_passengers
        
    def objective_function(self, avg_fare: float) -> float:
        """
        Calculate the objective function: PASSENGERS*AVG_FARE + PASSENGERS
        Returns negative infinity if constraints are violated.
        Note: We return negative value because scipy.optimize minimizes the function
        
        Args:
            avg_fare: The average fare price
            
        Returns:
            Negative value of the objective function or -inf if constraints violated
        """
        if avg_fare <= 0:
            return float('inf')  # Invalid fare
            
        passengers = self.passenger_predictor(avg_fare)
        
        if passengers > self.max_passengers:
            return float('inf')  # Too many passengers
            
        return -(passengers * avg_fare + passengers)  # Negative because we want to maximize
    
    def optimize(self, initial_guess: float = 100.0) -> tuple[float, float]:
        """
        Find the optimal average fare that maximizes the objective function.
        
        Args:
            initial_guess: Initial fare value to start optimization from
            
        Returns:
            Tuple of (optimal_fare, optimal_value)
        """
        # Using minimize_scalar with method='brent' which doesn't require bounds
        # but will still respect our constraints through the objective function
        result = minimize_scalar(
            self.objective_function,
            method='brent',
            options={'maxiter': 10000}
        )
        
        if not result.success:
            raise ValueError("Optimization failed to converge")
            
        optimal_fare = result.x
        optimal_value = -result.fun  # Convert back to positive value
        
        # Validate final result
        if optimal_fare <= 0:
            raise ValueError("Optimization resulted in invalid fare (≤ 0)")
            
        passengers = self.passenger_predictor(optimal_fare)
        if passengers > self.max_passengers:
            raise ValueError(f"Optimization resulted in too many passengers (>{self.max_passengers})")
        
        return optimal_fare, optimal_value