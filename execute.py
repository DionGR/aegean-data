from demand_forecaster import DemandForecaster
import pandas as pd
import numpy as np
from FINAL.fare_optimizer import FareOptimizer

forecaster = DemandForecaster('aegean_dataset.csv')

def example_passenger_predictor(avg_fare: float) -> float:
    """
    Example black-box model that predicts passengers based on average fare.
    This example ensures passengers never exceed 1000 and decreases with increasing fare.
    Replace this with your actual model.
    """

    results_dict = forecaster.predict_demand(2024, 1, 'D', seats=500000, fare=avg_fare)
    passengers = results_dict['predicted_pax']
    return passengers

optimizer = FareOptimizer(example_passenger_predictor, max_passengers=500000)
optimal_fare, optimal_value = optimizer.optimize()
optimal_passengers = example_passenger_predictor(optimal_fare)

print("Fare Optimization Results")
print(f"Optimal Fare: ${optimal_fare:.2f}")
print(f"Optimal Passengers: {optimal_passengers:.0f}")



