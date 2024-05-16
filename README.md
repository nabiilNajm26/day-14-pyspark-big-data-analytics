# Airline Customer Analysis

This repository contains the analysis of airline customer behavior and loyalty program engagement. The analysis uses datasets including calendar data, customer flight activity, and customer loyalty history. The goal is to derive insights and visualize trends to improve customer engagement and business strategies.

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Analysis](#analysis)
  - [Total Flights per Year](#total-flights-per-year)
  - [Average Distance Traveled per Loyalty Card Type](#average-distance-traveled-per-loyalty-card-type)
  - [Monthly Flight Activity Over Time by Loyalty Card Type](#monthly-flight-activity-over-time-by-loyalty-card-type)
- [Recommendations](#recommendations)

## Installation

To run the analysis, you need to have Python and the following libraries installed:

```sh
pip install pyspark matplotlib pandas seaborn
```

## Data Preparation

Ensure you have the following CSV files in a directory named `data`:
- `calendar.csv`
- `customer_flight_activity.csv`
- `customer_loyalty_history.csv`

## Usage

Run the script `analysis.py` to perform the analysis:

```sh
python analysis.py
```

The script will generate visualizations and save the results in the `analysis_results` directory.

## Analysis

### Total Flights per Year
![total_flights_per_year](https://github.com/nabiilNajm26/day-14-pyspark-big-data-analytics/assets/99080449/84b28b98-1d73-456c-9d8a-53310aef1be3)
- **Description**: Analyzes the total number of flights booked per year.
- **Graph**: A line plot showing the trend of total flights over the years.
- **Insights**:
  - Steady increase in the total number of flights from 2017 to 2018.
  - Indicates a growing customer base or increased travel frequency.
- **File**: `total_flights_per_year.png`

### Average Distance Traveled per Loyalty Card Type
![avg_distance_per_card](https://github.com/nabiilNajm26/day-14-pyspark-big-data-analytics/assets/99080449/a9a101f8-0da2-43c2-b522-d98339975fcf)
- **Description**: Calculates the average distance traveled by customers based on their loyalty card type.
- **Graph**: A bar chart displaying the average distance for each loyalty card type.
- **Insights**:
  - Aurora cardholders travel the most on average.
  - Small differences between the average distances for each card type.
- **File**: `avg_distance_per_card.png`

### Monthly Flight Activity Over Time by Loyalty Card Type
![monthly_flight_activity_by_loyalty_card](https://github.com/nabiilNajm26/day-14-pyspark-big-data-analytics/assets/99080449/6e36fcb6-7d57-4583-ad67-341641a68d6e)

- **Description**: Analyzes monthly flight activity for different loyalty card types over time.
- **Graph**: A line plot showing the monthly flight activity for Aurora, Nova, and Star cardholders.
- **Insights**:
  - Star cardholders have the highest flight activity.
  - All card types show peaks around mid-2017 and mid-2018, with dips towards the end of each year.
- **File**: `monthly_flight_activity_by_loyalty_card.png`

## Recommendations

- **Engagement Strategies**:
  - **Star Cardholders**: Maintain loyalty through exclusive benefits and rewards.
  - **Nova Cardholders**: Offer targeted promotions during low activity periods.
  - **Aurora Cardholders**: Continue efforts to engage and increase their flight activity.

- **Seasonal Promotions**:
  - Introduce mid-year promotions to capitalize on peak travel periods.
  - Implement end-of-year campaigns to mitigate the dip in flight activity.

- **Further Analysis**:
  - Investigate reasons behind the low points redemption rate.
  - Conduct deeper analysis to understand factors influencing travel patterns.
