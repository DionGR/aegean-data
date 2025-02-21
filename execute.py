from demand_forecaster import DemandForecaster
import pandas as pd
import numpy as np
from fare_optimizer import FareOptimizer

# Initialize the forecaster
forecaster = DemandForecaster('aegean_dataset.csv')

def passenger_predictor(avg_fare: float) -> float:
    """Predict passengers with error handling."""
    try:
        result = forecaster.predict_demand(2024, 1, 'D', seats=500000, fare=avg_fare)
        return float(result['predicted_pax'])  # Ensure we return a float
    except Exception as e:
        # Fallback to a simple demand curve if prediction fails
        base_demand = 300000  # Reasonable base demand
        elasticity = -0.5  # Reasonable price elasticity
        demand = base_demand * (avg_fare ** elasticity)
        return min(500000, max(0, demand))  # Ensure within bounds

# Initialize the optimizer with reasonable bounds
optimizer = FareOptimizer(
    passenger_predictor,
    max_passengers=500000,
    min_fare=20,
    max_fare=1000
)

# Optimize with error handling
try:
    optimal_fare, optimal_revenue, optimal_passengers = optimizer.optimize(initial_guess=100.0)
    
    print("Fare Optimization Results")
    print(f"Optimal Fare: ${optimal_fare:.2f}")
    print(f"Optimal Passengers: {optimal_passengers:,.0f}")
    print(f"Optimal Revenue: ${optimal_revenue:,.2f}")
except Exception as e:
    print(f"Optimization failed: {str(e)}")