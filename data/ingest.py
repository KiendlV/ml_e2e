import argparse
import os
from datetime import date, timedelta

import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 48.1374,
        "longitude": 11.5755,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "surface_pressure"],
        "models": "dwd_icon_seamless",
        "start_date": start_date,
        "end_date": end_date,
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "rain": hourly.Variables(2).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(3).ValuesAsNumpy(),
    }
    return pd.DataFrame(data=hourly_data)


def save_partitioned_by_year(df: pd.DataFrame, base_dir: str = "data/raw"):
    """Splits df by year (in case a pull spans a year boundary) and
    appends/dedupes into data/raw/historical_training_data_<year>.parquet"""
    df["year"] = pd.to_datetime(df["date"]).dt.year

    for year, group in df.groupby("year"):
        group = group.drop(columns="year")
        output_path = os.path.join(base_dir, f"historical_training_data_{year}.parquet")

        if os.path.exists(output_path):
            existing = pd.read_parquet(output_path)
            group = pd.concat([existing, group]).drop_duplicates(subset="date").sort_values("date")

        os.makedirs(base_dir, exist_ok=True)
        group.to_parquet(output_path, index=False)
        print(f"Saved {len(group)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD, defaults to yesterday")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD, defaults to yesterday")
    parser.add_argument("--output-dir", default="data/raw")
    args = parser.parse_args()

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    start_date = args.start_date or yesterday
    end_date = args.end_date or yesterday

    df = fetch_weather(start_date, end_date)
    save_partitioned_by_year(df, args.output_dir)


if __name__ == "__main__":
    main()