# F1 Race Prediction Interface
import pandas as pd
import numpy as np
import fastf1 as f1
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

class F1RacePredictor:
    """Interface for predicting F1 race winners"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.driver_mapping = {}
        self.team_mapping = {}
        self.historical_data = None
        
        if model_path:
            self.load_model(model_path)
    
    def save_model(self, filepath="f1_model.pkl"):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'driver_mapping': self.driver_mapping,
            'team_mapping': self.team_mapping
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath="f1_model.pkl"):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.driver_mapping = model_data['driver_mapping']
        self.team_mapping = model_data['team_mapping']
        print(f"Model loaded from {filepath}")
    
    def get_qualifying_data(self, year, race_round):
        """Get qualifying data from FastF1 API"""
        try:
            # Get qualifying session
            qualifying = f1.get_session(year, race_round, 'Q')
            qualifying.load()
            
            # Get race session for additional data
            race = f1.get_session(year, race_round, 'R')
            race.load()
            
            # Create DataFrame with qualifying results
            results = qualifying.results
            
            qualifying_data = pd.DataFrame({
                'Driver': results['Abbreviation'],
                'Team': results['TeamName'],
                'Grid': results['Position'],  # Qualifying position becomes grid position
                'Q1': results['Q1'],
                'Q2': results['Q2'], 
                'Q3': results['Q3']
            })
            
            # Get best qualifying time for each driver
            qualifying_data['BestQualifyingTime'] = qualifying_data[['Q1', 'Q2', 'Q3']].min(axis=1, skipna=True)
            qualifying_data['BestQualifyingTime'] = pd.to_timedelta(qualifying_data['BestQualifyingTime']).dt.total_seconds()
            
            return qualifying_data
            
        except Exception as e:
            print(f"Error getting qualifying data: {e}")
            return None
    
    def prepare_race_prediction_data(self, qualifying_data):
        """Prepare data for race prediction using qualifying results"""
        
        # Load historical data for feature engineering
        if self.historical_data is None:
            try:
                self.historical_data = pd.read_csv("F1_Final_Dataset.csv")
            except:
                self.historical_data = pd.read_csv("NewSetborn.csv")
        
        race_data = qualifying_data.copy()
        
        # Add driver and team encodings
        race_data['Driver_enc'] = race_data['Driver'].map(self.driver_mapping).fillna(-1)
        race_data['Team_enc'] = race_data['Team'].map(self.team_mapping).fillna(-1)
        
        # Estimate features based on historical data
        for idx, row in race_data.iterrows():
            driver = row['Driver']
            team = row['Team']
            
            # Get historical data for this driver
            driver_history = self.historical_data[self.historical_data['Driver'] == driver]
            team_history = self.historical_data[self.historical_data['Team'] == team]
            
            # Fill in estimated features based on historical averages
            if len(driver_history) > 0:
                race_data.loc[idx, 'FastestLap'] = driver_history['FastestLap'].median()
                race_data.loc[idx, 'AverageTime'] = driver_history['AverageTime'].median()
                race_data.loc[idx, 'pitstops'] = driver_history['pitstops'].median()
                race_data.loc[idx, 'driver_recent_wins'] = driver_history['Win'].tail(5).sum()
                race_data.loc[idx, 'driver_recent_form'] = driver_history['Finishpos'].tail(5).mean()
                race_data.loc[idx, 'driver_career_win_rate'] = driver_history['Win'].mean()
            else:
                # Use overall averages for new drivers
                race_data.loc[idx, 'FastestLap'] = self.historical_data['FastestLap'].median()
                race_data.loc[idx, 'AverageTime'] = self.historical_data['AverageTime'].median()
                race_data.loc[idx, 'pitstops'] = self.historical_data['pitstops'].median()
                race_data.loc[idx, 'driver_recent_wins'] = 0
                race_data.loc[idx, 'driver_recent_form'] = 10  # Mid-field average
                race_data.loc[idx, 'driver_career_win_rate'] = 0
            
            if len(team_history) > 0:
                race_data.loc[idx, 'team_recent_wins'] = team_history['Win'].tail(10).sum()
                race_data.loc[idx, 'team_recent_form'] = team_history['Finishpos'].tail(10).mean()
            else:
                race_data.loc[idx, 'team_recent_wins'] = 0
                race_data.loc[idx, 'team_recent_form'] = 10
        
        # Add grid-based features
        race_data['DNF'] = 0  # Assume no DNF for prediction
        race_data['front_row_start'] = (race_data['Grid'] <= 2).astype(int)
        race_data['top10_start'] = (race_data['Grid'] <= 10).astype(int)
        race_data['grid_advantage'] = (race_data['Grid'] <= 3).astype(int)
        race_data['pace_advantage'] = (race_data['BestQualifyingTime'] < race_data['BestQualifyingTime'].quantile(0.3)).astype(int)
        
        # Fill any remaining missing values
        race_data = race_data.fillna(race_data.median())
        
        return race_data
    
    def predict_race_winner(self, year, race_round):
        """Predict race winner for upcoming race"""
        
        if self.model is None:
            raise ValueError("No model loaded! Please train or load a model first.")
        
        # Get qualifying data
        print(f"Getting qualifying data for {year} Round {race_round}...")
        qualifying_data = self.get_qualifying_data(year, race_round)
        
        if qualifying_data is None:
            print("Could not get qualifying data. Using manual input method.")
            return self.predict_with_manual_input()
        
        # Prepare prediction data
        race_data = self.prepare_race_prediction_data(qualifying_data)
        
        # Select features that match training data
        available_features = [col for col in self.feature_columns if col in race_data.columns]
        missing_features = [col for col in self.feature_columns if col not in race_data.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Add missing features with default values
            for feature in missing_features:
                race_data[feature] = 0
        
        X_race = race_data[self.feature_columns]
        
        # Scale features
        X_race_scaled = self.scaler.transform(X_race)
        
        # Make predictions
        win_probabilities = self.model.predict_proba(X_race_scaled)[:, 1]
        
        # Create results
        results = pd.DataFrame({
            'Position': range(1, len(race_data) + 1),
            'Driver': race_data['Driver'],
            'Team': race_data['Team'],
            'Grid_Position': race_data['Grid'],
            'Win_Probability': win_probabilities,
            'Win_Percentage': win_probabilities * 100
        }).sort_values('Win_Probability', ascending=False).reset_index(drop=True)
        
        results['Predicted_Position'] = range(1, len(results) + 1)
        
        return results
    
    def predict_with_manual_input(self):
        """Manual input method for prediction when API data unavailable"""
        print("\nManual Race Prediction Mode")
        print("Please enter the following information:")
        
        drivers_data = []
        
        while True:
            print(f"\nDriver {len(drivers_data) + 1}:")
            driver_name = input("Driver name (or 'done' to finish): ").strip()
            
            if driver_name.lower() == 'done':
                break
            
            team_name = input("Team name: ").strip()
            grid_pos = int(input("Grid position: "))
            
            # Use historical averages or user estimates
            driver_data = {
                'Driver': driver_name,
                'Team': team_name,
                'Grid': grid_pos,
                'Driver_enc': self.driver_mapping.get(driver_name, -1),
                'Team_enc': self.team_mapping.get(team_name, -1),
                'FastestLap': 90.0,  # Default estimate
                'AverageTime': 95.0,  # Default estimate
                'pitstops': 2,       # Default pit stops
                'DNF': 0,
                'driver_recent_wins': 0,
                'team_recent_wins': 0,
                'grid_advantage': 1 if grid_pos <= 3 else 0,
                'pace_advantage': 1 if grid_pos <= 5 else 0,
                'front_row_start': 1 if grid_pos <= 2 else 0,
                'top10_start': 1 if grid_pos <= 10 else 0
            }
            
            drivers_data.append(driver_data)
        
        if not drivers_data:
            print("No driver data entered.")
            return None
        
        # Create DataFrame and predict
        race_data = pd.DataFrame(drivers_data)
        
        # Fill missing features with defaults
        for feature in self.feature_columns:
            if feature not in race_data.columns:
                race_data[feature] = 0
        
        X_race = race_data[self.feature_columns]
        X_race_scaled = self.scaler.transform(X_race)
        
        win_probabilities = self.model.predict_proba(X_race_scaled)[:, 1]
        
        results = pd.DataFrame({
            'Driver': race_data['Driver'],
            'Team': race_data['Team'],
            'Grid_Position': race_data['Grid'],
            'Win_Probability': win_probabilities,
            'Win_Percentage': win_probabilities * 100
        }).sort_values('Win_Probability', ascending=False).reset_index(drop=True)
        
        results['Predicted_Position'] = range(1, len(results) + 1)
        
        return results
    
    def get_season_predictions(self, year, start_round=1, end_round=24):
        """Get predictions for multiple races in a season"""
        season_predictions = {}
        
        for race_round in range(start_round, end_round + 1):
            try:
                print(f"\nPredicting Race {race_round}...")
                predictions = self.predict_race_winner(year, race_round)
                season_predictions[race_round] = predictions
                
                # Show top 3 predictions
                print(f"Top 3 predictions for Race {race_round}:")
                print(predictions.head(3)[['Driver', 'Team', 'Win_Percentage']].to_string(index=False))
                
            except Exception as e:
                print(f"Could not predict Race {race_round}: {e}")
                continue
        
        return season_predictions
    
    def compare_predictions_with_results(self, year, race_round):
        """Compare predictions with actual race results"""
        try:
            # Get predictions
            predictions = self.predict_race_winner(year, race_round)
            
            # Get actual results
            race = f1.get_session(year, race_round, 'R')
            race.load()
            actual_results = race.results
            
            # Create comparison
            comparison = pd.DataFrame({
                'Driver': actual_results['Abbreviation'],
                'Actual_Position': actual_results['Position'],
                'Actual_Winner': (actual_results['Position'] == 1).astype(int)
            })
            
            # Merge with predictions
            comparison = comparison.merge(
                predictions[['Driver', 'Predicted_Position', 'Win_Probability']], 
                on='Driver', 
                how='left'
            )
            
            # Calculate accuracy metrics
            winner_predicted = comparison[comparison['Actual_Winner'] == 1]['Win_Probability'].iloc[0] if len(comparison[comparison['Actual_Winner'] == 1]) > 0 else 0
            top3_accuracy = len(comparison[(comparison['Actual_Position'] <= 3) & (comparison['Predicted_Position'] <= 3)]) / 3
            
            print(f"\nPrediction Accuracy for Race {race_round}:")
            print(f"Winner Probability: {winner_predicted:.3f}")
            print(f"Top 3 Accuracy: {top3_accuracy:.3f}")
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing predictions: {e}")
            return None

# Utility functions for race prediction
def create_prediction_report(predictions, race_name=""):
    """Create a formatted prediction report"""
    
    report = f"""
{'='*60}
F1 RACE PREDICTION REPORT - {race_name}
{'='*60}

TOP 10 PREDICTIONS:
{'-'*40}
"""
    
    for idx, row in predictions.head(10).iterrows():
        report += f"{row['Predicted_Position']:2d}. {row['Driver']:15s} ({row['Team']:20s}) - {row['Win_Percentage']:5.1f}%\n"
    
    report += f"""
{'-'*40}
RACE WINNER PREDICTION: {predictions.iloc[0]['Driver']} ({predictions.iloc[0]['Win_Percentage']:.1f}% probability)

PODIUM PREDICTIONS:
1st: {predictions.iloc[0]['Driver']} ({predictions.iloc[0]['Win_Percentage']:.1f}%)
2nd: {predictions.iloc[1]['Driver']} ({predictions.iloc[1]['Win_Percentage']:.1f}%)
3rd: {predictions.iloc[2]['Driver']} ({predictions.iloc[2]['Win_Percentage']:.1f}%)

DARK HORSES (High probability from low grid positions):
"""
    
    # Find drivers with high win probability but low grid position
    dark_horses = predictions[(predictions['Grid_Position'] > 10) & (predictions['Win_Percentage'] > 5)]
    for _, row in dark_horses.iterrows():
        report += f"- {row['Driver']} (Grid P{row['Grid_Position']}, {row['Win_Percentage']:.1f}% win chance)\n"
    
    if len(dark_horses) == 0:
        report += "- No significant dark horses identified\n"
    
    report += f"\n{'='*60}"
    
    return report

# Example usage and testing
def main():
    """Main function to demonstrate the race predictor"""
    
    # Example of how to use the predictor
    predictor = F1RacePredictor()
    
    # First, you would need to train and save a model
    print("To use this predictor, first train your model using the enhanced training script")
    print("Then save it and load it here.")
    
    # Example of manual prediction
    print("\nExample: Manual prediction mode")
    
    # Simulate some driver data
    example_data = [
        {'Driver': 'VER', 'Team': 'Red Bull Racing', 'Grid': 1},
        {'Driver': 'LEC', 'Team': 'Ferrari', 'Grid': 2},
        {'Driver': 'HAM', 'Team': 'Mercedes', 'Grid': 3},
        {'Driver': 'RUS', 'Team': 'Mercedes', 'Grid': 4},
        {'Driver': 'NOR', 'Team': 'McLaren', 'Grid': 5}
    ]
    
    print("Example prediction data:")
    for driver in example_data:
        print(f"- {driver['Driver']} ({driver['Team']}) - Grid P{driver['Grid']}")
    
    # If you have a trained model, you could use:
    # predictor.load_model("your_model.pkl")
    # predictions = predictor.predict_race_winner(2024, 10)  # Example for race 10 of 2024
    # print(create_prediction_report(predictions, "Example Grand Prix"))

if __name__ == "__main__":
    main()