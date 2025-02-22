import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime

class HistoricalAnalyzer:
    def __init__(self, csv_path: str):
        """Initialize with historical data."""
        self.data = pd.read_csv(csv_path)
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess the data and calculate monthly statistics."""
        # Create separate dataframes for domestic and international
        self.domestic_df = self.data[self.data['D/I'] == 'D']
        self.international_df = self.data[self.data['D/I'] == 'I']
        
        # Calculate monthly statistics for both types
        self.monthly_stats = {
            'D': self._calculate_monthly_stats(self.domestic_df),
            'I': self._calculate_monthly_stats(self.international_df)
        }
    
    def _calculate_monthly_stats(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """Calculate statistics for each month."""
        monthly_stats = {}
        
        for month in range(1, 13):
            month_data = df[df['MONTH'] == month]
            
            # Calculate statistics with error handling
            try:
                monthly_stats[month] = {
                    'fare': {
                        'mean': month_data['FARE'].mean(),
                        'std': month_data['FARE'].std(),
                        'min': month_data['FARE'].min(),
                        'max': month_data['FARE'].max()
                    },
                    'pax': {
                        'mean': month_data['PAX'].mean(),
                        'std': month_data['PAX'].std(),
                        'min': month_data['PAX'].min(),
                        'max': month_data['PAX'].max()
                    },
                    'load_factor': {
                        'mean': (month_data['PAX'] / month_data['SEATS']).mean(),
                        'std': (month_data['PAX'] / month_data['SEATS']).std()
                    }
                }
            except Exception as e:
                print(f"Error calculating statistics for month {month}: {str(e)}")
                # Provide fallback values
                monthly_stats[month] = {
                    'fare': {'mean': 100, 'std': 20, 'min': 50, 'max': 200},
                    'pax': {'mean': 300000, 'std': 50000, 'min': 200000, 'max': 400000},
                    'load_factor': {'mean': 0.8, 'std': 0.1}
                }
        
        return monthly_stats
    
    def get_monthly_constraints(self, month: int, flight_type: str, 
                              std_multiplier: float = 3.0) -> Tuple[Dict, Dict]:
        """
        Get constraints for a specific month and flight type.
        Returns fare_constraints and pax_constraints.
        """
        if month not in range(1, 13) or flight_type not in ['D', 'I']:
            raise ValueError("Invalid month or flight type")
            
        stats = self.monthly_stats[flight_type][month]
        
        # Calculate fare constraints
        fare_constraints = {
            'min': max(1.0, stats['fare']['mean'] - std_multiplier * stats['fare']['std']),
            'max': stats['fare']['mean'] + std_multiplier * stats['fare']['std'],
            'mean': stats['fare']['mean']
        }
        
        # Calculate passenger constraints
        pax_constraints = {
            'min': max(1.0, stats['pax']['mean'] - std_multiplier * stats['pax']['std']),
            'max': stats['pax']['mean'] + std_multiplier * stats['pax']['std'],
            'mean': stats['pax']['mean']
        }
        
        # # Add load factor constraints
        # load_factor_constraints = {
        #     'min': max(0.3, stats['load_factor']['mean'] - std_multiplier * stats['load_factor']['std']),
        #     'max': min(0.95, stats['load_factor']['mean'] + std_multiplier * stats['load_factor']['std']),
        #     'mean': stats['load_factor']['mean']
        # }
        
        return fare_constraints, pax_constraints