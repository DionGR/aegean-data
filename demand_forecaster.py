import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')  # Suppress Prophet warnings

class DemandForecaster:
    def __init__(self, csv_path):
        """Initialize the DemandForecaster with data from csv_path."""
        self.data = pd.read_csv(csv_path)
        self.models = {
            'D': None,  # Domestic flights model
            'I': None   # International flights model
        }
        self.scalers = {
            'D': StandardScaler(),
            'I': StandardScaler()
        }
        # Store mean values for each flight type
        self.mean_values = {'D': {}, 'I': {}}
        self._preprocess_data()
        self._train_models()
        
    def _preprocess_data(self):
        """Preprocess the data for Prophet format with NaN handling."""
        # Create separate dataframes for domestic and international flights
        self.domestic_df = self.data[self.data['D/I'] == 'D'].copy()
        self.international_df = self.data[self.data['D/I'] == 'I'].copy()
        
        for flight_type, df in [('D', self.domestic_df), ('I', self.international_df)]:
            # Create datetime column for Prophet (ds)
            df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str).str.zfill(2) + '-01')
            
            # Set target variable (y)
            df['y'] = df['PAX']
            
            # Store mean values for later use
            self.mean_values[flight_type] = {
                'FARE': df['FARE'].mean(),
                'COMPETITION_FARE': df['COMPETITION_FARE'].mean(),
                'SEASONALITY': df['SEASONALITY'].mean(),
                'PAX': df['PAX'].mean()
            }
            
            df['month'] = df['ds'].dt.month
            
            # Add price elasticity features
            df = self._calculate_price_elasticity_features(df)
            
            # Scale features
            features_to_scale = ['FARE', 'COMPETITION_FARE', 'fare_ratio', 'fare_seasonality']
            scaled_features = self.scalers[flight_type].fit_transform(df[features_to_scale])
            df[features_to_scale] = scaled_features

    def _calculate_price_elasticity_features(self, df):
        """Calculate price elasticity related features with NaN handling."""
        # Handle potential division by zero or NaN in competition fare
        df['fare_ratio'] = np.where(
            df['COMPETITION_FARE'] > 0,
            df['FARE'] / df['COMPETITION_FARE'],
            df['FARE'] / df['COMPETITION_FARE'].mean()
        )
        
        # Handle negative or zero fares for log transformation
        min_fare = 0.01  # Minimum fare to prevent log(0)
        df['log_fare'] = np.log1p(np.maximum(df['FARE'], min_fare))
        df['log_comp_fare'] = np.log1p(np.maximum(df['COMPETITION_FARE'], min_fare))
        
        # Calculate fare interaction with seasonality
        df['fare_seasonality'] = df['FARE'] * df['SEASONALITY']
        
        # Replace any remaining NaN values with column means
        for col in ['fare_ratio', 'log_fare', 'log_comp_fare', 'fare_seasonality']:
            df[col] = df[col].fillna(df[col].mean())
        
        return df

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
        
        # Add regressors
        model.add_regressor('seasonality')
        model.add_regressor('fare_ratio')
        model.add_regressor('log_fare')
        model.add_regressor('log_comp_fare')
        model.add_regressor('fare_seasonality')
        
        # Prepare training data
        train_df = pd.DataFrame({
            'ds': df['ds'],
            'y': df['y'],
            'seasonality': df['SEASONALITY'],
            'fare_ratio': df['fare_ratio'],
            'log_fare': df['log_fare'],
            'log_comp_fare': df['log_comp_fare'],
            'fare_seasonality': df['fare_seasonality']
        })
        
        # Fit the model
        model.fit(train_df)
        return model

    def predict_demand(self, year, month, flight_type, fare=None):
        """Predict passenger demand with robust NaN handling."""
        if flight_type not in ['D', 'I']:
            raise ValueError("flight_type must be 'D' or 'I'")
            
        # Create prediction dataframe
        future_date = pd.DataFrame({
            'ds': [pd.to_datetime(f"{year}-{month:02d}-01")]
        })
        
        # Use mean values as defaults
        mean_values = self.mean_values[flight_type]
        fare_value = fare if fare is not None else mean_values['FARE']
        
        fare_value = max(0.01, fare_value)
        
        # Calculate price elasticity features with safety checks
        avg_comp_fare = mean_values['COMPETITION_FARE']
        fare_ratio = fare_value / max(0.01, avg_comp_fare)
        log_fare = np.log1p(fare_value)
        log_comp_fare = np.log1p(max(0.01, avg_comp_fare))
        fare_seasonality = fare_value * mean_values['SEASONALITY']
        
        # Scale features
        scaled_features = self.scalers[flight_type].transform([[
            fare_value, avg_comp_fare, fare_ratio, fare_seasonality
        ]])
        
        # Prepare prediction features
        future_date['seasonality'] = mean_values['SEASONALITY']
        future_date['fare_ratio'] = scaled_features[0][2]
        future_date['log_fare'] = log_fare
        future_date['log_comp_fare'] = log_comp_fare
        future_date['fare_seasonality'] = scaled_features[0][3]
        
        try:
            # Make prediction
            forecast = self.models[flight_type].predict(future_date)
            predicted_pax = max(0, forecast['yhat'].iloc[0])
            
            # Apply price elasticity adjustment
            if fare_value > 3 * avg_comp_fare:
                fare_multiplier = np.exp(-0.5 * (fare_value / avg_comp_fare - 3))
                predicted_pax *= fare_multiplier
            
            # Calculate load factor
            load_factor = min(1.0, max(0.0, load_factor))
            
            return {
                'predicted_pax': int(round(predicted_pax)),
                'load_factor': round(load_factor, 3),
                'lower_bound': max(0, int(round(forecast['yhat_lower'].iloc[0]))),
                'upper_bound': int(round(forecast['yhat_upper'].iloc[0])),
                'seasonality': mean_values['SEASONALITY']
            }
            
        except Exception as e:
            # Fallback to historical mean values
            mean_pax = mean_values['PAX']
            return {
                'predicted_pax': int(round(mean_pax)),
                'lower_bound': int(round(0.8 * mean_pax)),
                'upper_bound': int(round(1.2 * mean_pax)),
                'seasonality': mean_values['SEASONALITY']
            }