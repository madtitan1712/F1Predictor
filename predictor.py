import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_mappings():
    """Load the trained model and encoding mappings"""
    if not os.path.exists("f1_prediction_model.pkl"):
        print("Error: Model file not found. Please run main_enhanced.py first.")
        return None, None
        
    if not os.path.exists("encoding_mappings.json"):
        print("Error: Encoding mappings not found. Please run DataClean_enhanced.py first.")
        return None, None
    
    # Load model
    with open("f1_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Load mappings
    with open("encoding_mappings.json", "r") as f:
        mappings = json.load(f)
        
    return model, mappings

def get_track_mapping():
    """Get a list of available tracks for user selection"""
    if os.path.exists("encoding_mappings.json"):
        with open("encoding_mappings.json", "r") as f:
            mappings = json.load(f)
        
        track_mapping = mappings.get("track_mapping", {})
        return {v: k for k, v in track_mapping.items()}
    
    return {}

def get_driver_team_pairs():
    """Get a list of driver-team pairs from the cleaned data"""
    if os.path.exists("F1DatasetCleaned.csv"):
        data = pd.read_csv("F1DatasetCleaned.csv")
        latest_season_data = data[data["Season"] == data["Season"].max()]
        
        driver_teams = latest_season_data[["Driver", "Team"]].drop_duplicates()
        driver_team_dict = {row["Driver"]: row["Team"] for _, row in driver_teams.iterrows()}
        return driver_team_dict
    
    return {}

def predict_race_winner(grid_positions, track, fastest_lap_data=None, average_time_data=None, pitstop_data=None):
    """
    Predict the winner of a race based on grid positions and track
    
    Parameters:
    grid_positions: dict - {driver_name: grid_position}
    track: str - track name
    fastest_lap_data: dict - {driver_name: fastest_lap_time_in_seconds} (optional)
    average_time_data: dict - {driver_name: average_lap_time_in_seconds} (optional)
    pitstop_data: dict - {driver_name: number_of_pitstops} (optional)
    
    Returns:
    dict - Predicted probabilities for each driver to win
    """
    # Load model and mappings
    model, mappings = load_model_and_mappings()
    if not model or not mappings:
        return {}
    
    driver_mapping = mappings.get("driver_mapping", {})
    team_mapping = mappings.get("team_mapping", {})
    track_mapping = mappings.get("track_mapping", {})
    
    # Check if track exists in our data
    if track not in track_mapping:
        print(f"Warning: Track '{track}' not found in training data.")
        similar_tracks = [t for t in track_mapping.keys() if track.lower() in t.lower()]
        
        if similar_tracks:
            print(f"Using closest match: {similar_tracks[0]}")
            track = similar_tracks[0]
        else:
            print("No similar track found. Using a default track.")
            # Use the first track as default
            track = list(track_mapping.keys())[0]
    
    # Get default values from the training data
    if os.path.exists("F1DatasetCleaned.csv"):
        training_data = pd.read_csv("F1DatasetCleaned.csv")
        avg_fastest_lap = training_data["FastestLap"].mean()
        avg_average_time = training_data["AverageTime"].mean()
    else:
        avg_fastest_lap = 90.0  # Default value if training data is not available
        avg_average_time = 95.0  # Default value if training data is not available
    
    # Get driver-team pairs
    driver_team_dict = get_driver_team_pairs()
    
    # Prepare prediction data
    prediction_data = []
    
    # For each driver in the grid
    for driver, grid_pos in grid_positions.items():
        # Check if driver exists in our mapping
        if driver not in driver_mapping:
            print(f"Warning: Driver '{driver}' not found in mapping. Skipping.")
            continue
        
        # Get team for this driver
        team = driver_team_dict.get(driver)
        if not team or team not in team_mapping:
            print(f"Warning: Team for driver '{driver}' not found or not mapped.")
            if len(team_mapping) > 0:
                # Use the first team as default
                team = list(team_mapping.keys())[0]
            else:
                print(f"Error: No team mapping available. Skipping driver '{driver}'.")
                continue
        
        # Create prediction features
        driver_data = {
            "Driver_enc": driver_mapping[driver],
            "Team_enc": team_mapping[team],
            "Track_enc": track_mapping[track],
            "Grid": grid_pos,
            "FastestLap": fastest_lap_data.get(driver, avg_fastest_lap) if fastest_lap_data else avg_fastest_lap,
            "AverageTime": average_time_data.get(driver, avg_average_time) if average_time_data else avg_average_time,
            "pitstops": pitstop_data.get(driver, 1) if pitstop_data else 1,  # Default to 1 pitstop
            "DNF": 0  # Assuming no DNFs in prediction
        }
        prediction_data.append(driver_data)
    
    if not prediction_data:
        print("Error: No valid drivers to predict.")
        return {}
    
    # Convert to DataFrame
    pred_df = pd.DataFrame(prediction_data)
    
    # Make predictions
    win_probas = model.predict_proba(pred_df)[:, 1]  # Probability of winning
    
    # Create results dictionary
    results = {}
    i = 0
    for driver, _ in grid_positions.items():
        if driver in driver_mapping:  # Only include drivers that were in the prediction
            results[driver] = float(win_probas[i])
            i += 1
    
    # Sort by probability
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_results
def list_available_tracks():
    """List all available tracks in the dataset"""
    track_dict = get_track_mapping()
    if track_dict:
        print("\nAvailable tracks:")
        for i, track_name in enumerate(track_dict.values()):
            print(f"{i+1}. {track_name}")
    else:
        print("No track data available.")

def list_available_drivers():
    """List all available drivers in the dataset"""
    mappings = load_model_and_mappings()[1]
    if mappings:
        driver_mapping = mappings.get("driver_mapping", {})
        print("\nAvailable drivers:")
        sorted_drivers = sorted(driver_mapping.keys())
        for i, driver in enumerate(sorted_drivers):
            print(f"{i+1}. {driver}")
    else:
        print("No driver data available.")
if __name__ == "__main__":
    print("F1 Race Predictor - Track-Based Prediction")
    print("==========================================")
    
    # Show available tracks
    list_available_tracks()
    track = input("\nEnter track name: ")
    
    # Show available drivers
    list_available_drivers()
    
    # Get grid positions
    print("\nEnter grid positions (driver code and position, e.g., 'VER 1')")
    print("Enter 'done' when finished.")
    
    grid = {}
    while True:
        entry = input("> ")
        if entry.lower() == 'done':
            break
        
        try:
            driver, position = entry.split()
            grid[driver.upper()] = int(position)
        except ValueError:
            print("Invalid format. Use 'DRIVER POSITION' (e.g., 'VER 1')")
    
    if not grid:
        print("No grid positions entered. Using example grid.")
        grid = {
            "VER": 1,
            "HAM": 2,
            "LEC": 3,
            "NOR": 4,
            "PER": 5
        }
    results = predict_race_winner(grid_positions=grid, track=track)
    
    if results:
        print("\nPredicted Race Results:")
        print("======================")
        for i, (driver, probability) in enumerate(results.items()):
            print(f"{i+1}. {driver}: {probability:.4f} ({probability*100:.1f}% chance to win)")
