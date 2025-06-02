import fastf1 as f1
import datacollect as d1
import pandas as pd
#This code is used to create a CSV datafile which inlcudes all data from the year 2018-2024 from the FastF1 api

totalListofData = []

for year in range(2018, 2025):
    print(f"Processing season: {year}")
    event_schedule = f1.get_event_schedule(year, include_testing=False)
    race_events = event_schedule[event_schedule["EventFormat"] != "Testing"]

    for _, race in race_events.iterrows():
        round_number = race["RoundNumber"]

        if year == 2025 and round_number > 8:
            continue

        print(f"Loading data for {year} - Round {round_number}...")

        try:
            currentdata = d1.Datacollect(year, round_number, 'R')
            totalListofData.append(currentdata)
        except Exception as e:
            print(f"Skipped {year} Round {round_number} [Session: R] due to error: {e}")

print("All available races processed. Combining data...")

FinalDf = pd.concat(totalListofData, ignore_index=True)
FinalDf.to_csv("F1Dataset.csv", index=False)
print("Dataset saved as 'F1Dataset.csv'")
