import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class SeasonalityCalculator:
    def __init__(self):
        """Initialize the seasonality calculator with Greek calendar-based patterns."""
        self._setup_greek_calendar()
        
    def _setup_greek_calendar(self):
        """Define Greek calendar patterns that affect air travel demand."""
        # Base monthly patterns (normalized values based on typical Greek travel patterns)
        self.base_patterns = {
            'D': {  # Domestic base patterns
                1: 0.4,   # Low season (winter)
                2: 0.2,   # Lowest season
                3: 0.33,  # Early spring
                4: 0.45,  # Easter period
                5: 0.65,  # Late spring
                6: 1.0,   # Start of summer
                7: 1.0,   # Peak summer
                8: 1.0,   # Peak summer
                9: 0.8,   # Late summer
                10: 0.53,  # Fall
                11: 0.31,  # Late fall
                12: 0.5,   # Holiday season
            },
            'I': {  # International base patterns
                1: 0.2,   # Low season
                2: 0.2,   # Low season
                3: 0.3,   # Early spring
                4: 0.4,   # Easter period
                5: 0.6,   # Late spring
                6: 1.0,   # Peak summer start
                7: 1.0,   # Peak summer
                8: 1.0,   # Peak summer
                9: 0.8,   # Late summer
                10: 0.5,  # Fall
                11: 0.3,  # Late fall
                12: 0.3,  # Winter
            }
        }
        
        # Greek holidays and events
        self.holidays = {
            # National holidays
            "New_Year": {"month": 1, "day": 1, "duration": 3},
            "Epiphany": {"month": 1, "day": 6, "duration": 1},
            "Clean_Monday": {"month": 3, "day": 15, "duration": 3},  # Approximate
            "Independence_Day": {"month": 3, "day": 25, "duration": 1},
            "Labor_Day": {"month": 5, "day": 1, "duration": 1},
            "Holy_Spirit": {"month": 6, "day": 20, "duration": 1},  # Approximate
            "Assumption": {"month": 8, "day": 15, "duration": 3},
            "Oxi_Day": {"month": 10, "day": 28, "duration": 1},
            "Christmas": {"month": 12, "day": 25, "duration": 5},
        }
        
        # Seasonal periods
        self.seasons = {
            "Winter_Low": {
                "months": [1, 2],
                "impact": {"D": 0.2, "I": 0.2}
            },
            "Spring_Shoulder": {
                "months": [3, 4, 5],
                "impact": {"D": 0.4, "I": 0.3}
            },
            "Summer_Peak": {
                "months": [7, 8],
                "impact": {"D": 1.0, "I": 1.0}
            },
            "Summer_Shoulder": {
                "months": [6, 9],
                "impact": {"D": 0.8, "I": 0.8}
            },
            "Fall_Shoulder": {
                "months": [10],
                "impact": {"D": 0.5, "I": 0.5}
            },
            "Winter_Holiday": {
                "months": [12],
                "impact": {"D": 0.5, "I": 0.3}
            }
        }

    def _calculate_holiday_effect(self, month: int, flight_type: str) -> float:
        """Calculate holiday effect for a given month."""
        holiday_impact = 0.0
        
        # Check if month contains holidays
        for holiday, details in self.holidays.items():
            if details["month"] == month:
                # Domestic flights are more affected by Greek holidays
                if flight_type == 'D':
                    holiday_impact += 0.2 * (details["duration"] / 30)  # Normalized by month length
                else:
                    holiday_impact += 0.1 * (details["duration"] / 30)  # Less impact on international
                    
        return holiday_impact

    def _calculate_seasonal_effect(self, month: int, flight_type: str) -> float:
        """Calculate seasonal effect for a given month."""
        for season, details in self.seasons.items():
            if month in details["months"]:
                return details["impact"][flight_type]
        return self.base_patterns[flight_type][month]

    def get_seasonality(self, month: int, flight_type: str) -> float:
        """
        Get the final seasonality value for a specific month and flight type.
        Combines base patterns with holiday effects.
        """
        if flight_type not in ['D', 'I']:
            raise ValueError("flight_type must be 'D' or 'I'")
        if month < 1 or month > 12:
            raise ValueError("month must be between 1 and 12")
        
        # Get base seasonality from patterns
        base_seasonality = self.base_patterns[flight_type][month]
        
        # Add holiday effect
        holiday_effect = self._calculate_holiday_effect(month, flight_type)
        
        # Add seasonal effect
        seasonal_effect = self._calculate_seasonal_effect(month, flight_type)
        
        # Combine effects (base + holiday adjustment + seasonal adjustment)
        final_seasonality = base_seasonality + holiday_effect + (seasonal_effect - base_seasonality) * 0.5
        
        return round(final_seasonality, 2)