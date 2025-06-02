import pandas as pd
import fastf1 as f1

def Datacollect(year, race, sessiontype):
    mydata = pd.DataFrame()
    session = f1.get_session(year, race, sessiontype)
    session.load(laps=True, telemetry=False)
    laps = session.laps
    results = session.results
    data = session.session_info
    
    # Basic driver and race data
    mydata["Driver"] = results.get("Abbreviation")
    mydata["Team"] = results.get("TeamName")
    mydata["Grid"] = results["GridPosition"]
    mydata["Finishpos"] = results["Position"]
    
    # Add circuit information
    try:
        # Different versions of FastF1 might store this differently
        if "CircuitName" in session.event:
            mydata["Circuit"] = session.event["CircuitName"]
        elif "Circuit" in session.event:
            mydata["Circuit"] = session.event["Circuit"]
        else:
            # Fallback: Use event name as circuit
            mydata["Circuit"] = session.event.get("EventName", f"Circuit_{year}_{race}")
    except Exception as e:
        # Add a placeholder value
        mydata["Circuit"] = f"Circuit_{year}_{race}"
    
    # Add country information if available
    try:
        if "Country" in session.event:
            mydata["Country"] = session.event["Country"]
        else:
            mydata["Country"] = "Unknown"
    except:
        mydata["Country"] = "Unknown"
    
    # Add season year
    mydata["Season"] = year
    
    # Fastest lap calculation
    fastlaps = []
    for driver_code in results["Abbreviation"]:
        fastestlap = laps[laps["Driver"] == driver_code].pick_quicklaps()
        x = fastestlap["LapTime"].min()
        fastlaps.append(x)
    mydata["FastestLap"] = fastlaps 
    
    # Average lap time calculation
    avgtime = []
    for driver_code in results["Abbreviation"]:
        fastestlap = laps[laps["Driver"] == driver_code].pick_quicklaps()
        x = fastestlap["LapTime"].mean()
        avgtime.append(x)
    mydata["AverageTime"] = avgtime
    
    # Pitstop calculation
    stints = laps[['Driver', 'Stint']].drop_duplicates()
    stint_counts = stints.groupby('Driver').size()
    pit_counts = stint_counts - 1
    mydata['pitstops'] = mydata['Driver'].map(pit_counts).fillna(0).astype(int)
    
    return mydata