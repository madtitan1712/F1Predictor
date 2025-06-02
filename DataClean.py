import pandas as pd
#This Module will be used to get the data Cleaned and Preproccessed so that the main model can use it.

def getcleanedData():
    x=pd.read_csv("NewSet.csv")
    x["DNF"]=((x["FastestLap"].isna() )| (x["AverageTime"].isna())).astype(int)
    x["FastestLap"]=pd.to_timedelta(x["FastestLap"]).dt.total_seconds()
    x["AverageTime"]=pd.to_timedelta(x["AverageTime"]).dt.total_seconds()
    maxtime=x["FastestLap"].max(skipna=True)
    max_avg=x["AverageTime"].max(skipna=True)
    penalty=max(maxtime,max_avg)+100
    print(x.dtypes)
    x["AverageTime"]=x["AverageTime"].fillna(penalty)
    x["FastestLap"]=x["FastestLap"].fillna(penalty)
    #Encoding driver names and team names 
    drivernames=x["Driver"].unique()
    driver_enum={name:idx for idx,name in enumerate(drivernames)}
    teamnames=x["Team"].unique()
    team_enum={name:idx for idx,name in enumerate(teamnames)}
    print(driver_enum)
    print(team_enum)
    x["Driver_enc"]=x["Driver"].map(driver_enum)
    x["Team_enc"]=x["Team"].map(team_enum)
    x["Win"] = (x["Finishpos"] == 1).astype(int)
    x.to_csv("NewSetborn.csv",mode="w",index=False)
    return x