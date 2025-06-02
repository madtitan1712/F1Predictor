import pandas as pd
import numpy as np
import os

def getcleanedData():
    """Clean and process F1 race data for modeling"""
    
    # Check if the dataset exists
    if not os.path.exists("F1Dataset.csv"):
        print("Error: F1Dataset.csv not found.")
        print("Please run DataFarmer_enhanced.py first to collect the data.")
        return None
    
    print("Loading F1 dataset...")
    try:
        # Load the dataset with error handling for different encodings
        x = pd.read_csv("F1Dataset.csv")
        print(f"Loaded dataset with {len(x)} rows and {len(x.columns)} columns")
        print(f"Columns: {x.columns.tolist()}")
        # Checking for required columns 
        required_columns = ["Driver", "Team", "Grid", "Finishpos", "FastestLap", "AverageTime", "Circuit"]
        missing_columns = [col for col in required_columns if col not in x.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return None
        # Checking for DNF status..
        print("Processing DNF status...")
        x["DNF"] = ((x["FastestLap"].isna()) | (x["AverageTime"].isna())).astype(int)
        # Convert lap times to seconds with 2 ways of Error handling...
        # One way is using the to_timedelta function... If that fails
        # Converting manually is another go 
        print("Converting lap times to seconds...")
        try:
            # Try direct conversion to timedelta
            x["FastestLap"] = pd.to_timedelta(x["FastestLap"]).dt.total_seconds()
        except Exception as e:
            print(f"Warning: Error converting FastestLap: {str(e)}")
            print("Using alternative conversion method...")
            
            # Manual Conversion Function
            def safe_convert_to_seconds(time_str):
                if pd.isna(time_str):
                    return np.nan
                
                try:
                    # Try direct conversion
                    return pd.to_timedelta(time_str).total_seconds()
                except:
                    try:
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            minutes = float(parts[0])
                            seconds = float(parts[1])
                            return minutes * 60 + seconds
                        else:
                            return float(time_str)  
                    except:
                        print(f"Warning: Could not parse time: {time_str}")
                        return np.nan
            
            # Applying safe conversion
            x["FastestLap"] = x["FastestLap"].apply(safe_convert_to_seconds)
        
        # Doing so similarly using the predefined functions
        try:
            x["AverageTime"] = pd.to_timedelta(x["AverageTime"]).dt.total_seconds()
        except Exception as e:
            print(f"Warning: Error converting AverageTime: {str(e)}")
            print("Using alternative conversion method...")
            x["AverageTime"] = x["AverageTime"].apply(safe_convert_to_seconds)
        
        # Handling missing data by filling penalty times.
        maxtime = x["FastestLap"].max(skipna=True)
        max_avg = x["AverageTime"].max(skipna=True)
        
        if pd.isna(maxtime) or pd.isna(max_avg):
            print("Warning: All lap times are missing. Using default penalty.")
            penalty = 120.0  # Default 2-minute penalty if all times are missing
        else:
            penalty = max(maxtime, max_avg) + 100
        
        print(f"Using penalty time of {penalty} seconds for DNFs")
        x["AverageTime"] = x["AverageTime"].fillna(penalty)
        x["FastestLap"] = x["FastestLap"].fillna(penalty)
        
        # Ensure Grid positions are numeric
        x["Grid"] = pd.to_numeric(x["Grid"], errors='coerce').fillna(20)  # Default to back of grid
        
        # Ensure Finishpos is numeric
        x["Finishpos"] = pd.to_numeric(x["Finishpos"], errors='coerce').fillna(20)
        
        # Ensure pitstops is numeric
        if "pitstops" in x.columns:
            x["pitstops"] = pd.to_numeric(x["pitstops"], errors='coerce').fillna(0)
        else:
            print("Warning: pitstops column not found. Adding default values.")
            x["pitstops"] = 1  # Default to 1 pitstop
        
        # Encoding categorical variables
        print("Encoding categorical variables...")
        
        # Ensure we have string types for categorical columns before encoding
        for col in ["Driver", "Team", "Circuit"]:
            x[col] = x[col].astype(str)
        
        # Driver encoding
        drivernames = x["Driver"].unique()
        driver_enum = {name: idx for idx, name in enumerate(drivernames)}
        
        # Team encoding
        teamnames = x["Team"].unique()
        team_enum = {name: idx for idx, name in enumerate(teamnames)}
        
        # Track encoding
        tracknames = x["Circuit"].unique()
        track_enum = {name: idx for idx, name in enumerate(tracknames)}
        
        # Apply encodings
        x["Driver_enc"] = x["Driver"].map(driver_enum)
        x["Team_enc"] = x["Team"].map(team_enum)
        x["Track_enc"] = x["Circuit"].map(track_enum)
        
        # Create target variable - flag for race win
        x["Win"] = (x["Finishpos"] == 1).astype(int)
        
        # Save the cleaned dataset
        print("Saving cleaned dataset...")
        x.to_csv("F1DatasetCleaned.csv", index=False)
        
        # Save the encoding mappings for later use
        encoding_data = {
            "driver_mapping": driver_enum,
            "team_mapping": team_enum,
            "track_mapping": track_enum
        }
        
        import json
        with open("encoding_mappings.json", "w") as f:
            json.dump(encoding_data, f)
        
        # Display summary stats
        print("\nCleaning summary:")
        print(f"Total races: {len(x)}")
        print(f"Total drivers: {len(drivernames)}")
        print(f"Total teams: {len(teamnames)}")
        print(f"Total tracks: {len(tracknames)}")
        print(f"DNF rate: {x['DNF'].mean():.1%}")
        
        print("\nData cleaning complete!")
        return x
    
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# If the script is run directly, execute the cleaning
if __name__ == "__main__":
    getcleanedData()