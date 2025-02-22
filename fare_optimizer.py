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
                 std_multiplier: float = 3.0,
                 seats: int = 500000):
        """
        Initialize the fare optimizer with historical constraints and capacity limits.
        """
        self.passenger_predictor = passenger_predictor
        self.capacity = seats
        
        # Get historical constraints
        fare_constraints, pax_constraints = historical_analyzer.get_monthly_constraints(month, flight_type, std_multiplier)
        
        # Set constraints based on historical data
        self.min_fare = fare_constraints['min']
        self.max_fare = fare_constraints['max']
        self.expected_fare = fare_constraints['mean']
        
        self.min_passengers = pax_constraints['min']
        self.max_passengers = pax_constraints['max']
        
    def find_capacity_matching_fare(self, current_fare: float) -> float:
        """
        Find the fare that matches capacity when demand exceeds it.
        Uses binary search to find the optimal fare that brings demand down to capacity.
        """
        left = current_fare
        right = self.max_fare * 1.5  # Allow some flexibility above max_fare
        target = self.capacity
        tolerance = 1.0  # Acceptable difference in passengers
        
        while (right - left) > 0.1:  # 0.1 precision for fare
            mid_fare = (left + right) / 2
            predicted_pax = self.passenger_predictor(mid_fare)
            
            if abs(predicted_pax - target) <= tolerance:
                return mid_fare
            elif predicted_pax > target:
                left = mid_fare
            else:
                right = mid_fare
                
        return left

    def objective_function(self, avg_fare: float) -> float:
        """
        Calculate the objective function that maximizes revenue while respecting capacity.
        Only increases fares if demand exceeds capacity.
        """
        try:
            # Initial check for fare bounds
            if avg_fare < self.min_fare:
                return float('inf')
            
            # Get predicted passengers at current fare
            predicted_pax = self.passenger_predictor(avg_fare)
            
            # Handle invalid predictions
            if not isinstance(predicted_pax, (int, float)) or np.isnan(predicted_pax):
                return float('inf')
            
            # If demand exceeds capacity, increase fare to match capacity
            if predicted_pax > self.capacity:
                # Find fare that brings demand to capacity
                optimal_fare = self.find_capacity_matching_fare(avg_fare)
                predicted_pax = min(self.passenger_predictor(optimal_fare), self.capacity)
                revenue = predicted_pax * optimal_fare
            else:
                revenue = predicted_pax * avg_fare
            
            # Apply a small penalty for extreme deviations from expected fare
            # Only when we're not at capacity
            if predicted_pax < (self.capacity * 0.95):
                fare_deviation = abs(avg_fare - self.expected_fare) / self.expected_fare
                penalty = 0.05 * fare_deviation  # Reduced penalty impact
                revenue *= (1 - penalty)
            
            return -revenue  # Negative because we're minimizing
            
        except Exception as e:
            return float('inf')
    
    def optimize(self, initial_guess: float = None) -> tuple[float, float, float]:
        """
        Find the optimal fare that maximizes revenue while respecting capacity constraints.
        """
        try:
            # Use expected fare as initial guess if none provided
            if initial_guess is None:
                initial_guess = self.expected_fare
            
            # First optimize normally within historical bounds
            result = minimize_scalar(
                self.objective_function,
                bounds=(self.min_fare, self.max_fare),
                method='bounded',
                options={'maxiter': 500}
            )
            
            if not result.success:
                # Fall back to grid search
                fares = np.linspace(self.min_fare, self.max_fare, 100)
                revenues = [-self.objective_function(f) for f in fares]
                best_idx = np.argmax(revenues)
                optimal_fare = fares[best_idx]
                optimal_value = revenues[best_idx]
            else:
                optimal_fare = result.x
                optimal_value = -result.fun
            
            # Get predicted passengers at optimal fare
            predicted_pax = self.passenger_predictor(optimal_fare)
            
            # If demand exceeds capacity, find the fare that matches capacity
            if predicted_pax > self.capacity:
                optimal_fare = self.find_capacity_matching_fare(optimal_fare)
                predicted_pax = min(self.passenger_predictor(optimal_fare), self.capacity)
                optimal_value = predicted_pax * optimal_fare
            
            return optimal_fare, optimal_value, predicted_pax
            
        except Exception as e:
            predicted_pax = min(self.passenger_predictor(self.expected_fare), self.capacity)
            return self.expected_fare, predicted_pax * self.expected_fare, predicted_pax