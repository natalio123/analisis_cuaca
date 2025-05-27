import pandas as pd
from dashboard import create_dashboard

df = pd.read_parquet("data/weather_data.parquet")
df['date'] = pd.to_datetime(df['date'])

create_dashboard(df)