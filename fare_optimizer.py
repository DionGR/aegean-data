import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable
from datetime import datetime

class FareOptimizer:
    def __init__(self, passenger_predictor: Callable[[float], float], max_passengers: int = 1000,
                 min_fare: float = None, max_fare: float = None):
        """
        Initialize the fare optimizer with bounds checking.
        """
        self.passenger_predictor = passenger_predictor
        self.max_passengers = max_passengers
        self.min_fare = max(1.0, min_fare) if min_fare is not None else 1.0
        self.max_fare = min(1e4, max_fare) if max_fare is not None else 1e4
        
    def objective_function(self, avg_fare: float) -> float:
        """
        Calculate the objective function with error handling.
        """
        try:
            # Ensure fare is within reasonable bounds
            if not self.min_fare <= avg_fare <= self.max_fare:
                return float('inf')
            
            # Get predicted passengers
            passengers = self.passenger_predictor(avg_fare)
            
            # Handle invalid passenger numbers
            if not isinstance(passengers, (int, float)) or np.isnan(passengers):
                return float('inf')
            
            # Check passenger constraints
            if passengers <= 0 or passengers > self.max_passengers:
                return float('inf')
            
            # Calculate revenue
            revenue = passengers * avg_fare
            
            return -revenue  
            
        except Exception as e:
            # Return infinity for any calculation errors
            return float('inf')
    
    def optimize(self, initial_guess: float = 100.0) -> tuple[float, float, float]:
        """
        Find the optimal fare with robust error handling.
        """
        try:
            # Ensure initial guess is within bounds
            initial_guess = max(self.min_fare, min(initial_guess, self.max_fare))
            
            # Use minimize_scalar with bounds
            result = minimize_scalar(
                self.objective_function,
                bounds=(self.min_fare, self.max_fare),
                method='bounded',
                options={'maxiter': 1000}
            )
            
            if not result.success:
                # Fall back to grid search if optimization fails
                fares = np.linspace(self.min_fare, self.max_fare, 100)
                revenues = [-self.objective_function(f) for f in fares]
                best_idx = np.argmax(revenues)
                optimal_fare = fares[best_idx]
                optimal_value = revenues[best_idx]
            else:
                optimal_fare = result.x
                optimal_value = -result.fun
            
            # Get final passenger prediction
            passengers = self.passenger_predictor(optimal_fare)
            
            # Validate results
            if np.isnan(optimal_value) or np.isnan(passengers):
                raise ValueError("Optimization resulted in NaN values")
            
            return optimal_fare, optimal_value, passengers
            
        except Exception as e:
            # Fallback to a reasonable default if optimization fails completely
            default_fare = initial_guess
            passengers = self.passenger_predictor(default_fare)
            revenue = passengers * default_fare
            return default_fare, revenue, passengers