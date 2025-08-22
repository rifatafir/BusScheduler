# AI-Enhanced Bus Scheduling Project
## Overview
This project optimizes bus schedules for Hajee Mohammad Danesh Science and Technology University to Gor-e-Shahid Boro Math, Dinajpur, using AI. It runs every 5 minutes, predicting passenger demand with linear regression, integrating Google Maps traffic data (30-min one-way, 1-hr round trip), and optimizing bus allocation with PuLP. The dataset (`historical_demand.csv`) grows with each run, improving accuracy (R² ~0.85). See `project_report.txt` for details.

## Files
- `bus_scheduler_course_project.py`: Main script for scheduling and predictions.
- `project_report.txt`: Detailed project description and methodology.
- `improved_bus_schedule.csv`: Optimized bus schedule output.
- `demand_chart.json`: Chart.js configuration for demand visualization.
- `bus_schedule_log.txt`: Execution logs.
- `validation_log.txt`: Validation logs for demand predictions.
- `main_bus_schedule.csv`: Official university-to-city schedule.
- `special_trips.csv`: Official special trips schedule (e.g., Friday Prayers).
- `historical_demand.csv`: Growing dataset for demand predictions.
- `README.md`: This file.

## How to Run
1. Install dependencies: `pip install pandas numpy pulp googlemaps statsmodels scikit-learn schedule`
2. Run the script: `python bus_scheduler_course_project.py`
3. Outputs are generated in the same folder.
4. Stop with Ctrl+C.

## Notes
- Requires a Google Maps API key (set in script).
- Synthetic data used; replace with real passenger data for production.
- Submitted for Data Science with AI course, August 2025.