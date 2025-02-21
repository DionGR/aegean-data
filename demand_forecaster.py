import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # Suppress Prophet warnings

class DemandForecaster:
    def __init__(self, csv_path):
        """Initialize the DemandForecaster with data from csv_path."""
        self.data = pd.read_csv(csv_path)
        self.models = {
            'D': None,  # Domestic flights model
            'I': None   # International flights model
        }
        self._preprocess_data()
        self._train_models()
    
    def _preprocess_data(self):
        """Preprocess the data for Prophet format."""
        # Create separate dataframes for domestic and international flights
        self.domestic_df = self.data[self.data['D/I'] == 'D'].copy()
        self.international_df = self.data[self.data['D/I'] == 'I'].copy()
        
        # Convert YEAR and MONTH to datetime for both datasets
        for df in [self.domestic_df, self.international_df]:
            df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + 
                                    df['MONTH'].astype(str) + '-01')
            df['y'] = df['PAX']
            
            # Add derived features
            df['load_factor'] = df['PAX'] / df['SEATS']
            
            # Create month feature for seasonality
            df['month'] = df['ds'].dt.month

    def _train_models(self):
        """Train separate Prophet models for domestic and international flights."""
        # Train domestic flights model
        self.models['D'] = self._create_and_fit_model(self.domestic_df)
        
        # Train international flights model
        self.models['I'] = self._create_and_fit_model(self.international_df)
    
    def _create_and_fit_model(self, df):
        """Create and fit a Prophet model with custom features."""
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add custom seasonality using the SEASONALITY column
        model.add_regressor('seasonality')
        model.add_regressor('comp_fare')
        model.add_regressor('seats')
        model.add_regressor('fare')
        
        # Prepare training data
        train_df = pd.DataFrame({
            'ds': df['ds'],
            'y': df['y'],
            'seasonality': df['SEASONALITY'],
            'seats': df['SEATS'],
            'comp_fare': df['COMP_FARE'],
            'fare': df['FARE']
        })
        
        # Fit the model
        model.fit(train_df)
        return model

    def predict_demand(self, year, month, flight_type, seats=None, fare=None):
        """
        Predict passenger demand for a specific year, month, and flight type.
        
        Parameters:
        -----------
        year : int
            The year for prediction
        month : int
            The month for prediction (1-12)
        flight_type : str
            'D' for domestic or 'I' for international
        seats : int, optional
            Number of seats constraint
            
        Returns:
        --------
        dict
            Dictionary containing the predicted PAX and additional metrics
        """
        if flight_type not in ['D', 'I']:
            raise ValueError("flight_type must be 'D' or 'I'")
            
        # Create prediction dataframe
        future_date = pd.DataFrame({
            'ds': [datetime(year, month, 1)]
        })
        
        # Get historical data for the same month
        historical_data = (self.domestic_df if flight_type == 'D' 
                         else self.international_df)
        same_month = historical_data[historical_data['MONTH'] == month]
        
        # Calculate average metrics for the month
        avg_seasonality = same_month['SEASONALITY'].mean()
        avg_comp_fare = same_month['COMP_FARE'].mean()
        
        # Use provided seats or average seats for the month
        seats_value = seats if seats is not None else same_month['SEATS'].mean()
        
        # Add required features for prediction
        future_date['seasonality'] = avg_seasonality
        future_date['comp_fare'] = avg_comp_fare
        future_date['seats'] = seats_value
        future_date['fare'] = fare if fare is not None else avg_comp_fare
        
        # Make prediction
        forecast = self.models[flight_type].predict(future_date)
        
        # Calculate load factor
        predicted_pax = max(0, forecast['yhat'].iloc[0])  # Ensure non-negative
        if seats:
            predicted_pax = min(predicted_pax, seats)  # Apply seats constraint
        load_factor = predicted_pax / seats_value if seats_value > 0 else 0
        
        return {
            'predicted_pax': int(round(predicted_pax)),
            'load_factor': round(load_factor, 3),
            'lower_bound': max(0, int(round(forecast['yhat_lower'].iloc[0]))),
            'upper_bound': int(round(forecast['yhat_upper'].iloc[0])),
            'seasonality': avg_seasonality
        }
