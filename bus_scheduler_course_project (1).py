import pandas as pd
import math
import os
import numpy as np
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpInteger
import googlemaps
from datetime import datetime
import time
import schedule
import logging
from statsmodels.formula.api import ols
from sklearn.metrics import mean_absolute_error, r2_score
import googlemaps.exceptions

# Set API key directly for Jupyter Notebook
os.environ['GOOGLE_MAPS_API_KEY'] = 'AIzaSyC9igNw56QlMLyejzlLao4lFexiHgAONP8'
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
print("API Key:", API_KEY)

# Configuration
SCHEDULE_INTERVAL_MINUTES = 5
OUTPUT_FILE = "improved_bus_schedule.csv"
PREVIOUS_OUTPUT_FILE = "previous_bus_schedule.csv"
HISTORICAL_DATA_FILE = "historical_demand.csv"
OFFICIAL_MAIN_SCHEDULE_FILE = "main_bus_schedule.csv"
OFFICIAL_SPECIAL_SCHEDULE_FILE = "special_trips.csv"
LOG_FILE = "bus_schedule_log.txt"
VALIDATION_LOG_FILE = "validation_log.txt"
CHART_FILE = "demand_chart.json"
ORIGIN = "Hajee Mohammad Danesh Science and Technology University, Dinajpur, Bangladesh"
DESTINATION = "Gor-e-Shahid Boro Math, Dinajpur, Bangladesh"
BUS_CAPACITY = 55
MAX_TRAVEL_TIME_MINUTES = 30
ROUND_TRIP_TIME_MINUTES = 60

# Create official schedules if not present
def create_official_schedules():
    main_schedule = [
        {"Trip Name": "1st Trip", "Departure Time (University)": "8:50 AM", "Bus No.": "17", "Departure Time (City)": "7:20 AM", "Bus No.1": "17"},
        {"Trip Name": "2nd Trip", "Departure Time (University)": "8:10 AM", "Bus No.": "4, 20", "Departure Time (City)": "7:20 AM", "Bus No.1": "4, 20"},
        {"Trip Name": "3rd Trip", "Departure Time (University)": "9:00 AM", "Bus No.": "10, 19, 21", "Departure Time (City)": "7:30 AM", "Bus No.1": "10, 19, 21"},
        {"Trip Name": "4th Trip", "Departure Time (University)": "9:30 AM", "Bus No.": "2, 5, 11", "Departure Time (City)": "7:50 AM", "Bus No.1": "2, 5, 11"},
        {"Trip Name": "5th Trip", "Departure Time (University)": "9:35 AM", "Bus No.": "3, 6, 7", "Departure Time (City)": "8:00 AM", "Bus No.1": "3, 6, 7"},
        {"Trip Name": "6th Trip", "Departure Time (University)": "10:00 AM", "Bus No.": "1, 8, 13", "Departure Time (City)": "8:10 AM", "Bus No.1": "1, 8, 13"},
        {"Trip Name": "7th Trip", "Departure Time (University)": "10:30 AM", "Bus No.": "12, 18", "Departure Time (City)": "8:30 AM", "Bus No.1": "12, 18"},
        {"Trip Name": "8th Trip", "Departure Time (University)": "11:00 AM", "Bus No.": "9, 15", "Departure Time (City)": "9:00 AM", "Bus No.1": "9, 15"},
        {"Trip Name": "9th Trip", "Departure Time (University)": "11:30 AM", "Bus No.": "14, 16", "Departure Time (City)": "9:15 AM", "Bus No.1": "14, 16"},
        {"Trip Name": "10th Trip", "Departure Time (University)": "12:00 PM", "Bus No.": "22", "Departure Time (City)": "9:30 AM", "Bus No.1": "22"},
        {"Trip Name": "11th Trip", "Departure Time (University)": "12:30 PM", "Bus No.": "23", "Departure Time (City)": "9:40 AM", "Bus No.1": "23"},
        {"Trip Name": "12th Trip", "Departure Time (University)": "1:00 PM", "Bus No.": "24", "Departure Time (City)": "9:50 AM", "Bus No.1": "24"},
        {"Trip Name": "13th Trip", "Departure Time (University)": "1:30 PM", "Bus No.": "25", "Departure Time (City)": "10:00 AM", "Bus No.1": "25"},
        {"Trip Name": "14th Trip", "Departure Time (University)": "2:00 PM", "Bus No.": "26", "Departure Time (City)": "10:10 AM", "Bus No.1": "26"},
        {"Trip Name": "15th Trip", "Departure Time (University)": "2:30 PM", "Bus No.": "27", "Departure Time (City)": "10:20 AM", "Bus No.1": "27"},
        {"Trip Name": "16th Trip", "Departure Time (University)": "3:00 PM", "Bus No.": "28", "Departure Time (City)": "10:30 AM", "Bus No.1": "28"},
        {"Trip Name": "17th Trip", "Departure Time (University)": "3:30 PM", "Bus No.": "29", "Departure Time (City)": "10:40 AM", "Bus No.1": "29"},
        {"Trip Name": "18th Trip", "Departure Time (University)": "4:00 PM", "Bus No.": "30", "Departure Time (City)": "10:50 AM", "Bus No.1": "30"},
        {"Trip Name": "19th Trip", "Departure Time (University)": "4:30 PM", "Bus No.": "31", "Departure Time (City)": "11:00 AM", "Bus No.1": "31"},
        {"Trip Name": "20th Trip", "Departure Time (University)": "5:00 PM", "Bus No.": "32", "Departure Time (City)": "11:10 AM", "Bus No.1": "32"},
        {"Trip Name": "21st Trip", "Departure Time (University)": "5:30 PM", "Bus No.": "33", "Departure Time (City)": "11:20 AM", "Bus No.1": "33"},
        {"Trip Name": "22nd Trip", "Departure Time (University)": "6:00 PM", "Bus No.": "34", "Departure Time (City)": "11:30 AM", "Bus No.1": "34"},
        {"Trip Name": "23rd Trip", "Departure Time (University)": "6:30 PM", "Bus No.": "35", "Departure Time (City)": "11:40 AM", "Bus No.1": "35"},
        {"Trip Name": "24th Trip", "Departure Time (University)": "7:00 PM", "Bus No.": "36", "Departure Time (City)": "11:50 AM", "Bus No.1": "36"},
        {"Trip Name": "25th Trip", "Departure Time (University)": "7:30 PM", "Bus No.": "37", "Departure Time (City)": "12:00 PM", "Bus No.1": "37"},
        {"Trip Name": "26th Trip", "Departure Time (University)": "8:00 PM", "Bus No.": "38", "Departure Time (City)": "12:10 PM", "Bus No.1": "38"},
        {"Trip Name": "27th Trip", "Departure Time (University)": "8:30 PM", "Bus No.": "39", "Departure Time (City)": "12:20 PM", "Bus No.1": "39"},
        {"Trip Name": "28th Trip", "Departure Time (University)": "9:00 PM", "Bus No.": "40", "Departure Time (City)": "12:30 PM", "Bus No.1": "40"},
        {"Trip Name": "29th Trip", "Departure Time (University)": "9:30 PM", "Bus No.": "41", "Departure Time (City)": "12:40 PM", "Bus No.1": "41"},
        {"Trip Name": "30th Trip", "Departure Time (University)": "10:00 PM", "Bus No.": "42", "Departure Time (City)": "12:50 PM", "Bus No.1": "42"},
        {"Trip Name": "31st Trip", "Departure Time (University)": "10:30 PM", "Bus No.": "43", "Departure Time (City)": "1:00 PM", "Bus No.1": "43"},
        {"Trip Name": "32nd Trip", "Departure Time (University)": "11:00 PM", "Bus No.": "44", "Departure Time (City)": "1:10 PM", "Bus No.1": "44"}
    ]
    special_schedule = [
        {"Occasion": "Friday Prayers", "University Departure": "1:00 PM", "Buses": 5, "City Departure": "1:30 PM", "Buses.1": 5},
        {"Occasion": "Evening Prayer", "University Departure": "3:00 PM", "Buses": 12, "City Departure": "3:30 PM", "Buses.1": 12},
        {"Occasion": "Special Event Trips", "University Departure": "4:00 PM", "Buses": 15, "City Departure": "4:30 PM", "Buses.1": 15},
        {"Occasion": "Weekend Service", "University Departure": "5:00 PM", "Buses": 20, "City Departure": "5:30 PM", "Buses.1": 20}
    ]
    pd.DataFrame(main_schedule).to_csv(OFFICIAL_MAIN_SCHEDULE_FILE, index=False)
    pd.DataFrame(special_schedule).to_csv(OFFICIAL_SPECIAL_SCHEDULE_FILE, index=False)
    return pd.DataFrame(main_schedule), pd.DataFrame(special_schedule)

# Load or create official schedules
def load_official_schedules():
    if not os.path.exists(OFFICIAL_MAIN_SCHEDULE_FILE) or not os.path.exists(OFFICIAL_SPECIAL_SCHEDULE_FILE):
        return create_official_schedules()
    return pd.read_csv(OFFICIAL_MAIN_SCHEDULE_FILE), pd.read_csv(OFFICIAL_SPECIAL_SCHEDULE_FILE)

# Generate or fix historical data
def ensure_historical_data():
    expected_columns = ['hour', 'day_of_week', 'traffic_time', 'demand']
    if not os.path.exists(HISTORICAL_DATA_FILE):
        print(f"Generating initial synthetic historical data for {HISTORICAL_DATA_FILE}...")
        num_days = 100
        hours = np.repeat(range(6, 24), num_days)
        days = np.tile(range(7), len(hours) // 7 + 1)[:len(hours)]
        traffic_times = np.random.uniform(20, MAX_TRAVEL_TIME_MINUTES, len(hours))
        is_peak = np.isin(hours, [7, 8, 9, 12, 13, 16, 17, 18])
        base_demand = np.where(is_peak, 300, 100)  # Reduced variance
        day_effect = np.where(days >= 5, 30, 0)  # Smaller weekend boost
        demands = base_demand + 3 * (traffic_times - 25) + day_effect + np.random.normal(0, 15, len(hours))
        demands = np.clip(demands, 80, 450)  # Tighter range for consistency
        historical_df = pd.DataFrame({
            'hour': hours,
            'day_of_week': days,
            'traffic_time': traffic_times,
            'demand': demands
        })
        historical_df.to_csv(HISTORICAL_DATA_FILE, index=False)
        print(f"Initial synthetic data saved to {HISTORICAL_DATA_FILE}")
    else:
        historical_df = pd.read_csv(HISTORICAL_DATA_FILE)
        missing_cols = [col for col in expected_columns if col not in historical_df.columns]
        if missing_cols:
            print(f"Missing columns {missing_cols} in {HISTORICAL_DATA_FILE}. Regenerating...")
            os.remove(HISTORICAL_DATA_FILE)
            num_days = 100
            hours = np.repeat(range(6, 24), num_days)
            days = np.tile(range(7), len(hours) // 7 + 1)[:len(hours)]
            traffic_times = np.random.uniform(20, MAX_TRAVEL_TIME_MINUTES, len(hours))
            is_peak = np.isin(hours, [7, 8, 9, 12, 13, 16, 17, 18])
            base_demand = np.where(is_peak, 300, 100)
            day_effect = np.where(days >= 5, 30, 0)
            demands = base_demand + 3 * (traffic_times - 25) + day_effect + np.random.normal(0, 15, len(hours))
            demands = np.clip(demands, 80, 450)
            historical_df = pd.DataFrame({
                'hour': hours,
                'day_of_week': days,
                'traffic_time': traffic_times,
                'demand': demands
            })
            historical_df.to_csv(HISTORICAL_DATA_FILE, index=False)
            print(f"Regenerated historical data saved to {HISTORICAL_DATA_FILE}")
    historical_df = pd.read_csv(HISTORICAL_DATA_FILE)
    print(f"Historical data columns: {list(historical_df.columns)}")
    print(f"Historical data sample:\n{historical_df.head()}")
    return historical_df

# Set up logging
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
validation_logger = logging.getLogger('validation')
validation_handler = logging.FileHandler(VALIDATION_LOG_FILE)
validation_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
validation_logger.addHandler(validation_handler)
validation_logger.setLevel(logging.INFO)

# Initial departure times from official schedule
def get_initial_departures(main_schedule):
    uni_times = pd.to_datetime(main_schedule['Departure Time (University)'], format='%I:%M %p').dt.hour + pd.to_datetime(main_schedule['Departure Time (University)'], format='%I:%M %p').dt.minute / 60
    city_times = pd.to_datetime(main_schedule['Departure Time (City)'], format='%I:%M %p').dt.hour + pd.to_datetime(main_schedule['Departure Time (City)'], format='%I:%M %p').dt.minute / 60
    return list(uni_times) + list(city_times)

def get_traffic_data_with_retry(max_retries=3):
    """Fetch real-time travel time from Google Maps, capped at 30 minutes."""
    if not API_KEY:
        logging.error("No API key provided. Set GOOGLE_MAPS_API_KEY environment variable.")
        print("Error: No API key provided. Set GOOGLE_MAPS_API_KEY environment variable.")
        return MAX_TRAVEL_TIME_MINUTES
    gmaps = googlemaps.Client(key=API_KEY)
    for attempt in range(max_retries):
        try:
            directions = gmaps.directions(ORIGIN, DESTINATION, mode="driving", departure_time=datetime.now())
            duration_in_traffic = min(directions[0]['legs'][0]['duration_in_traffic']['value'] / 60, MAX_TRAVEL_TIME_MINUTES)
            logging.info(f"Fetched travel time: {duration_in_traffic} minutes")
            print(f"Fetched travel time: {duration_in_traffic} minutes")
            return duration_in_traffic
        except googlemaps.exceptions.ApiError as e:
            logging.warning(f"API error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    logging.error("Max retries exceeded, using max travel time.")
    return MAX_TRAVEL_TIME_MINUTES

def predict_demand(travel_time):
    """Predict demand using ML model with traffic data and evaluate."""
    historical_df = ensure_historical_data()
    train_size = int(0.8 * len(historical_df))
    train_df = historical_df[:train_size]
    test_df = historical_df[train_size:]
    
    formula = 'demand ~ hour + day_of_week + traffic_time'
    if 'day_of_week' not in historical_df.columns:
        logging.warning("day_of_week missing, using simpler model.")
        print("Warning: day_of_week missing, using simpler model.")
        formula = 'demand ~ hour + traffic_time'
    
    try:
        model = ols(formula, data=train_df).fit()
        hours = range(6, 24)
        current_day = datetime.now().weekday()
        pred_df = pd.DataFrame({
            'hour': hours,
            'day_of_week': [current_day] * len(hours),
            'traffic_time': [travel_time] * len(hours)
        })
        predicted_demands = model.predict(pred_df)
        demand_dict = {h: max(0, p) for h, p in zip(hours, predicted_demands)}
        
        test_predictions = model.predict(test_df)
        mae = mean_absolute_error(test_df['demand'], test_predictions)
        r2 = r2_score(test_df['demand'], test_predictions)
        logging.info(f"Model evaluation: MAE = {mae:.2f}, R² = {r2:.2f}")
        print(f"Model evaluation: MAE = {mae:.2f}, R² = {r2:.2f}")
        return demand_dict
    except Exception as e:
        logging.error(f"Error in model fitting: {e}")
        print(f"Error in model fitting: {e}")
        return {h: 200 for h in range(6, 24)}  # Fallback demand

def optimize_schedule(demand, official_main, official_special):
    """Optimize bus schedule based on predicted demand, using official as baseline and 1-hour round trip."""
    official_main['University_Hour'] = pd.to_datetime(official_main['Departure Time (University)'], format='%I:%M %p').dt.hour
    official_main['City_Hour'] = pd.to_datetime(official_main['Departure Time (City)'], format='%I:%M %p').dt.hour
    uni_frequency = official_main.groupby('University_Hour')['Bus No.'].apply(lambda x: sum(len(str(b).split(',')) for b in x)).reindex(range(6, 24), fill_value=0)
    city_frequency = official_main.groupby('City_Hour')['Bus No.1'].apply(lambda x: sum(len(str(b).split(',')) for b in x)).reindex(range(6, 24), fill_value=0)
    
    needed = {h: max(1, math.ceil(demand[h] / BUS_CAPACITY)) for h in demand}
    
    improved_uni = {h: max(needed.get(h, 0), uni_frequency.get(h, 0)) for h in range(6, 24)}
    improved_city = {h: max(needed.get(h, 0), city_frequency.get(h, 0)) for h in range(6, 24)}
    
    slots = list(range(6, 24))
    prob = LpProblem("Bus_Scheduling", LpMinimize)
    buses = LpVariable.dicts("Buses", slots, lowBound=0, cat=LpInteger)
    prob += lpSum([buses[i] for i in slots])
    for i in slots:
        prob += buses[i] * BUS_CAPACITY >= demand[i]
    
    current_day = datetime.now().strftime('%A')
    special_buses = {}
    if current_day == 'Friday':
        for _, row in official_special.iterrows():
            hour = pd.to_datetime(row['University Departure'], format='%I:%M %p').hour
            special_demand = demand.get(hour, 200)
            special_buses[hour] = min(row['Buses'], max(math.ceil(special_demand / BUS_CAPACITY), uni_frequency.get(hour, 0)))
            prob += buses[hour] >= special_buses[hour]
    
    prob.solve()
    
    schedule_df = pd.DataFrame({
        'Hour': slots,
        'Predicted_Demand': [demand[h] for h in slots],
        'Official_Uni_Buses': [uni_frequency.get(h, 0) for h in slots],
        'Official_City_Buses': [city_frequency.get(h, 0) for h in slots],
        'Suggested_Uni_Buses': [min(improved_uni[h] + special_buses.get(h, 0), math.ceil(demand[h] / BUS_CAPACITY) + 1) for h in slots],
        'Suggested_City_Buses': [min(improved_city[h] + special_buses.get(h, 0), math.ceil(demand[h] / BUS_CAPACITY) + 1) for h in slots],
        'Optimal_Buses_AI': [int(buses[h].value()) for h in slots]
    })
    return schedule_df

def generate_demand_chart(demand):
    """Generate Chart.js line plot for predicted demand."""
    chart_config = {
        "type": "line",
        "data": {
            "labels": [str(h) for h in range(6, 24)],
            "datasets": [{
                "label": "Predicted Passenger Demand",
                "data": [demand[h] for h in range(6, 24)],
                "borderColor": "#1f77b4",
                "backgroundColor": "rgba(31, 119, 180, 0.2)",
                "fill": True
            }]
        },
        "options": {
            "responsive": True,
            "title": {
                "display": True,
                "text": "Hourly Passenger Demand Prediction"
            },
            "scales": {
                "x": {"title": {"display": True, "text": "Hour of Day"}},
                "y": {"title": {"display": True, "text": "Predicted Demand (Passengers)"}, "beginAtZero": True}
            }
        }
    }
    with open(CHART_FILE, 'w') as f:
        import json
        json.dump(chart_config, f, indent=2)
    logging.info(f"Demand chart saved to {CHART_FILE}")
    print(f"Demand chart saved to {CHART_FILE}")

def update_historical_data(travel_time, demand):
    """Append new data to historical CSV, log for validation."""
    historical_df = ensure_historical_data()
    current_time = datetime.now()
    current_hour = current_time.hour
    current_day = current_time.weekday()
    predicted_demand = demand.get(current_hour, historical_df['demand'].mean())
    new_row = pd.DataFrame({
        'hour': [current_hour],
        'day_of_week': [current_day],
        'traffic_time': [travel_time],
        'demand': [predicted_demand]
    })
    historical_df = pd.concat([historical_df, new_row], ignore_index=True)
    historical_df.to_csv(HISTORICAL_DATA_FILE, index=False)
    logging.info(f"Updated historical data with hour {current_hour}, day {current_day}, travel_time {travel_time}, demand {new_row['demand'].iloc[0]}")
    print(f"Updated historical data with hour {current_hour}, day {current_day}, travel_time {travel_time}, demand {new_row['demand'].iloc[0]}")
    validation_log = f"Hour: {current_hour}, Travel_Time: {travel_time}, Predicted_Demand: {predicted_demand}, Actual_Demand: Not provided, Manual_Busyness: Not provided"
    validation_logger.info(validation_log)
    print(f"Validation Log: {validation_log}")

def save_schedule(schedule_df, travel_time):
    """Save improved schedule with traffic and time data."""
    current_time_str = datetime.now().isoformat()
    schedule_df['Current_Time'] = current_time_str
    schedule_df['Travel_Time'] = travel_time
    schedule_df['Round_Trip_Time'] = ROUND_TRIP_TIME_MINUTES
    schedule_df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Improved schedule saved to {OUTPUT_FILE}")
    print(f"Improved schedule saved to {OUTPUT_FILE}")

def job():
    """Main job to run every 5 minutes."""
    print("Running update at", datetime.now())
    logging.info("Starting update")
    official_main, official_special = load_official_schedules()
    previous_df = None
    if os.path.exists(OUTPUT_FILE):
        previous_df = pd.read_csv(OUTPUT_FILE)
        previous_df.to_csv(PREVIOUS_OUTPUT_FILE, index=False)
    travel_time = get_traffic_data_with_retry()
    demand = predict_demand(travel_time)
    schedule_df = optimize_schedule(demand, official_main, official_special)
    if previous_df is not None and 'Optimal_Buses_AI' in previous_df.columns:
        schedule_df['Previous_Optimal_Buses'] = previous_df['Optimal_Buses_AI']
    else:
        schedule_df['Previous_Optimal_Buses'] = [0] * len(schedule_df)
    update_historical_data(travel_time, demand)
    save_schedule(schedule_df, travel_time)
    generate_demand_chart(demand)
    print("\nImproved Schedule (AI Suggestions vs Official):")
    print(schedule_df)
    logging.info("Update completed")

# Schedule the job to run every 5 minutes
schedule.every(SCHEDULE_INTERVAL_MINUTES).minutes.do(job)

# Run indefinitely
print("Starting infinite scheduling loop. Press Ctrl+C to stop.")
try:
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
except KeyboardInterrupt:
    print("Scheduling stopped by user.")
    logging.info("Scheduling stopped by user.")