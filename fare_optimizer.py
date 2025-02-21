import numpy as np
from scipy.optimize import minimize_scalar
from typing import Callable
from datetime import datetime
from historical_analyzer import HistoricalAnalyzer

class FareOptimizer:
    def __init__(self, passenger_predictor: Callable[[float], float],
                 historical_analyzer: HistoricalAnalyzer,
                 month: int,
                 flight_type: str,
                 std_multiplier: float = 2.0,
                 seats: int = 500000):
        """
        Initialize the fare optimizer with historical constraints.
        """
        self.passenger_predictor = passenger_predictor
        
        # Get historical constraints
        fare_constraints, _, _ = \
            historical_analyzer.get_monthly_constraints(month, flight_type, std_multiplier)
        
        # Set constraints based on historical data
        self.min_fare = fare_constraints['min']
        self.max_fare = fare_constraints['max']
        self.expected_fare = fare_constraints['mean']
        
        self.min_passengers = 0
        self.max_passengers = seats
                
    def objective_function(self, avg_fare: float) -> float:
        """
        Calculate the objective function with historical constraints.
        """
        try:
            # Ensure fare is within historical bounds
            if not self.min_fare <= avg_fare <= self.max_fare:
                return float('inf')
            
            # Get predicted passengers
            passengers = self.passenger_predictor(avg_fare)
            
            # Handle invalid passenger numbers
            if not isinstance(passengers, (int, float)) or np.isnan(passengers):
                return float('inf')
            
            # Calculate revenue
            revenue = passengers * avg_fare
            
            # Apply penalties based on deviation from historical patterns
            fare_penalty = ((avg_fare - self.expected_fare) / self.expected_fare) ** 2
            
            # Adjusted revenue with penalties
            adjusted_revenue = revenue * (1 - 0.1 * (fare_penalty))
            
            return -adjusted_revenue
            
        except Exception as e:
            return float('inf')
    
    def optimize(self, initial_guess: float = None) -> tuple[float, float, float]:
        """
        Find the optimal fare with historical constraints.
        """
        try:
            # Use expected fare as initial guess if none provided
            if initial_guess is None:
                initial_guess = self.expected_fare
            else:
                initial_guess = max(self.min_fare, min(initial_guess, self.max_fare))
            
            # Use minimize_scalar with bounds
            result = minimize_scalar(
                self.objective_function,
                bounds=(self.min_fare, self.max_fare),
                method='bounded',
                options={'maxiter': 1000}
            )
            
            if not result.success:
                # Fall back to grid search with historical constraints
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
            
            # Validate results against historical patterns
            if optimal_fare < self.min_fare or optimal_fare > self.max_fare:
                optimal_fare = self.expected_fare
            
            if passengers < self.min_passengers or passengers > self.max_passengers:
                optimal_value = passengers * optimal_fare

            return optimal_fare, optimal_value, passengers
            
        except Exception as e:
            # Fallback to historical averages if optimization fails
            return self.expected_fare, self.expected_fare 