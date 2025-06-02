import pandas as pd
import fastf1 as f1
#This code is used in the DataFarmer Module for fetching data from the FastF1 api


#Gets The race details in the args, curates the required data and returns the dataframe
def Datacollect(year,race,sessiontype):
    mydata=pd.DataFrame()
    session=f1.get_session(year,race,sessiontype)
    session.load(laps=True,telemetry=False)
    laps=session.laps
    results=session.results
    data=session.session_info
    mydata["Driver"]=results.get("Abbreviation")
    mydata["Team"]=results.get("TeamName")
    mydata["Grid"]=results["GridPosition"]
    mydata["Finishpos"]=results["Position"]
    #fastestLap
    fastlaps=[]
    for driver_code in results["Abbreviation"]:
        fastestlap=laps[laps["Driver"] == driver_code].pick_quicklaps()
        x=fastestlap["LapTime"].min()
        fastlaps.append(x)
    mydata["FastestLap"]=fastlaps 
    #AvgTime
    avgtime=[]
    for driver_code in results["Abbreviation"]:
        fastestlap=laps[laps["Driver"] == driver_code].pick_quicklaps()
        x=fastestlap["LapTime"].mean()
        avgtime.append(x)
    mydata["AverageTime"]=avgtime
    stints = laps[['Driver', 'Stint']].drop_duplicates()
    stint_counts = stints.groupby('Driver').size()
    pit_counts = stint_counts - 1
    mydata['pitstops'] = mydata['Driver'].map(pit_counts).fillna(0).astype(int)
    return mydata