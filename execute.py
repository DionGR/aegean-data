from demand_forecaster import DemandForecaster
from historical_analyzer import HistoricalAnalyzer
from fare_optimizer import FareOptimizer
import pandas as pd
import numpy as np

# Initialize the analyzers
historical_analyzer = HistoricalAnalyzer('aegean_dataset.csv')
forecaster = DemandForecaster('aegean_dataset.csv')

# Setup for January domestic flights
MONTH = 1
FLIGHT_TYPE = 'D'
seats = 200000

def passenger_predictor(avg_fare: float) -> float:
    result = forecaster.predict_demand(2024, MONTH, FLIGHT_TYPE, fare=avg_fare)
    return float(result['predicted_pax'])


# Initialize the optimizer with historical constraints
optimizer = FareOptimizer(
    passenger_predictor,
    historical_analyzer,
    month=MONTH,
    flight_type=FLIGHT_TYPE,
    seats=seats
)

# Get historical constraints for reference
fare_constraints, _, _ = \
    historical_analyzer.get_monthly_constraints(MONTH, FLIGHT_TYPE)

# Optimize with error handling
try:
    optimal_fare, optimal_revenue, optimal_passengers = optimizer.optimize()
    
    print("Model Results")
    print(f"Optimal Fare: {optimal_fare:.2f}")
    print(f"Optimal Revenue: {optimal_revenue:.2f}")
    print(f"Optimal Passengers: {optimal_passengers:.2f}")

except Exception as e:
    print(f"Optimization failed: {str(e)}")