import fastf1 as f1
import datacollect as d1
import pandas as pd
import os
import time

# This code creates a comprehensive F1 dataset with track information from 2018-2024
os.makedirs('data', exist_ok=True)
totalListofData = []
success_count = 0
failure_count = 0
print("Starting data collection process...")
for year in range(2018, 2025):
    print(f"\nProcessing season: {year}")
    try:
        event_schedule = f1.get_event_schedule(year, include_testing=False)
        race_events = event_schedule[event_schedule["EventFormat"] != "Testing"]
        print(f"Found {len(race_events)} race events for {year}")
    except Exception as e:
        print(f"Error getting schedule for {year}: {str(e)}")
        continue
    
    for idx, (_, race) in enumerate(race_events.iterrows()):
        round_number = race["RoundNumber"]
        
        if year == 2024 and round_number > 12:  #Skipping the 2025 Season
            print(f"Skipping future races from {year}")
            break
            
        race_name = race.get("EventName", f"Round {round_number}")
        print(f"Processing {year} - {race_name} (Round {round_number})...")
        
        try:
            time.sleep(0.5)  # API Cooldown time
            currentdata = d1.Datacollect(year, round_number, 'R')
            
            if currentdata is not None and not currentdata.empty:
                print(f"Successfully loaded data: {len(currentdata)} rows")
                # Saving every single race in the data folder
                race_filename = f"data/race_{year}_{round_number}.csv"
                currentdata.to_csv(race_filename, index=False)
                totalListofData.append(currentdata)
                success_count += 1
            else:
                print(f"Warning: Empty dataset returned for {year} Round {round_number}")
                failure_count += 1
        except Exception as e:
            print(f"Failed {year} Round {round_number} due to error: {str(e)}")
            failure_count += 1
print(f"\n--- Data Collection Summary ---")
print(f"Successful races: {success_count}")
print(f"Failed races: {failure_count}")
if len(totalListofData) == 0:
    print("No data collected. Cannot create dataset.")
    exit(1)
print(f"\nCombining {len(totalListofData)} race datasets...")
FinalDf = pd.concat(totalListofData, ignore_index=True)
print(f"Combined dataset shape: {FinalDf.shape}")
# Create summary statistics
track_stats = FinalDf.groupby("Circuit").size().reset_index(name='RaceCount')
track_stats = track_stats.sort_values('RaceCount', ascending=False)
print(f"\nData collected from {len(track_stats)} different circuits:")
print(track_stats.head(10))  # Show top 10 tracks by race count
driver_stats = FinalDf.groupby("Driver").size().reset_index(name='RaceCount')
driver_stats = driver_stats.sort_values('RaceCount', ascending=False)
print(f"\nData collected from {len(driver_stats)} different drivers:")
print(driver_stats.head(10))  # Show top 10 drivers by race count
# Save the complete dataset
FinalDf.to_csv("F1Dataset.csv", index=False)
print("\nComplete dataset saved as 'F1Dataset.csv'")
# Save the track statistics
track_stats.to_csv("data/TrackStats.csv", index=False)
print("Track statistics saved as 'data/TrackStats.csv'")