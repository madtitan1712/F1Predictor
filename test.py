import fastf1 as f1
import pandas as pd
session=f1.get_session(2021,'Monaco','R')
session.load(laps=True,telemetry=False)
data=session.session_info
print(data)