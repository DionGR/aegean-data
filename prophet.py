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
            df['fare_ratio'] = df['FARE'] / df['COMPETITION_FARE']
            
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
        model.add_regressor('fare_ratio')
        model.add_regressor('seats')
        
        # Prepare training data
        train_df = pd.DataFrame({
            'ds': df['ds'],
            'y': df['y'],
            'seasonality': df['SEASONALITY'],
            'fare_ratio': df['fare_ratio'],
            'seats': df['SEATS']
        })
        
        # Fit the model
        model.fit(train_df)
        return model

    def generate_forecast_range(self, start_year, start_month, end_year, end_month, seats_constraint=None, output_file='forecast_results.csv'):
        """
        Generate forecasts for a range of months and save to CSV.
        
        Parameters:
        -----------
        start_year : int
            Starting year for forecast
        start_month : int
            Starting month for forecast (1-12)
        end_year : int
            Ending year for forecast
        end_month : int
            Ending month for forecast (1-12)
        seats_constraint : dict, optional
            Dictionary with 'D' and 'I' keys containing yearly seats constraints
        output_file : str
            Path to save the output CSV file
        """
        # Generate date range
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        
        # Initialize results list
        forecast_results = []
        
        # Generate predictions for both domestic and international flights
        for flight_type in ['D', 'I']:
            # Get historical data
            historical_data = (self.domestic_df if flight_type == 'D' 
                             else self.international_df)
            
            # Create future dataframe for all dates
            future_dates = pd.DataFrame({'ds': dates})
            
            # Add features for prediction
            future_dates['month'] = future_dates['ds'].dt.month
            
            # Calculate average metrics per month from historical data
            monthly_averages = historical_data.groupby('MONTH').agg({
                'SEASONALITY': 'mean',
                'fare_ratio': 'mean',
                'SEATS': 'mean'
            }).reset_index()
            
            # Map monthly averages to future dates
            future_dates = future_dates.merge(
                monthly_averages,
                left_on='month',
                right_on='MONTH',
                how='left'
            )
            
            # Apply seats constraint if provided
            if seats_constraint and flight_type in seats_constraint:
                yearly_seats = seats_constraint[flight_type]
                # Distribute seats based on historical monthly distribution
                monthly_seat_dist = (historical_data.groupby('MONTH')['SEATS']
                                   .mean() / historical_data.groupby('MONTH')['SEATS'].mean().sum())
                future_dates['SEATS'] = future_dates['month'].map(
                    monthly_seat_dist * yearly_seats)
            
            # Prepare final prediction dataframe
            pred_df = pd.DataFrame({
                'ds': future_dates['ds'],
                'seasonality': future_dates['SEASONALITY'],
                'fare_ratio': future_dates['fare_ratio'],
                'seats': future_dates['SEATS']
            })
            
            # Make predictions
            forecast = self.models[flight_type].predict(pred_df)
            
            # Process results
            for idx, row in forecast.iterrows():
                date = row['ds']
                predicted_pax = max(0, row['yhat'])  # Ensure non-negative
                
                # Apply seats constraint
                if seats_constraint and flight_type in seats_constraint:
                    seats = pred_df.loc[idx, 'seats']
                    predicted_pax = min(predicted_pax, seats)
                
                forecast_results.append({
                    'YEAR': date.year,
                    'MONTH': date.month,
                    'D/I': flight_type,
                    'PAX_PRED': int(round(predicted_pax))
                })
        
        # Convert results to DataFrame and sort
        results_df = pd.DataFrame(forecast_results)
        results_df = results_df.sort_values(['YEAR', 'D/I', 'MONTH'])
        
        results_pax_only_df = results_df[['PAX_PRED']]
        results_pax_only_df.to_csv(output_file, index=False)
        
        return results_df

    def predict_demand(self, year, month, flight_type, seats=None):
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
        avg_fare_ratio = same_month['fare_ratio'].mean()
        
        # Use provided seats or average seats for the month
        seats_value = seats if seats is not None else same_month['SEATS'].mean()
        
        # Add required features for prediction
        future_date['seasonality'] = avg_seasonality
        future_date['fare_ratio'] = avg_fare_ratio
        future_date['seats'] = seats_value
        
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
        
# Initialize the forecaster
forecaster = DemandForecaster('aegean_dataset.csv')

# Example: Generate forecasts from January 2024 to December 2025
# Optional: Add seats constraints for each year and flight type
seats_constraint = {
    'D': 7000000,  # Yearly seats constraint for domestic flights
    'I': 12000000  # Yearly seats constraint for international flights
}

# Generate forecasts and save to CSV
forecast_df = forecaster.generate_forecast_range(
    start_year=2024,
    start_month=1,
    end_year=2024,
    end_month=12,
    seats_constraint=seats_constraint,
    output_file='forecast_results.csv'
)

# Display the first few rows of the forecast
print("\nFirst few rows of the forecast:")
print(forecast_df.head(10))

# Display some summary statistics
print("\nSummary statistics by flight type:")
print(forecast_df.groupby('D/I')['PAX_PRED'].describe())