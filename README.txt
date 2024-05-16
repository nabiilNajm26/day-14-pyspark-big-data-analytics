# Airline Customer Analysis

This analysis examines the flight activity and loyalty program engagement of airline customers. We use three main datasets: calendar, customer flight activity, and customer loyalty history. The analysis includes visualizations and insights derived from the data.

## How to Run the Script

1. Install Required Libraries:
    Ensure you have PySpark and other necessary libraries installed:
    
    pip install pyspark matplotlib pandas seaborn
    

2. Prepare the Data:
    Place the provided CSV files (`calendar.csv`, `customer_flight_activity.csv`, `customer_loyalty_history.csv`) in a directory named `data`.

3. Run the Script:
    Execute the script `analysis.py`:
    
    python analysis.py

4. Output:
    The analysis results, including CSV files and visualizations, will be saved in the `analysis_results` directory.

## Analysis Explanation

### 1. Total Flights per Year
- Description: Analyzes the total number of flights booked per year.
- Graph: A line plot showing the trend of total flights over the years.
- Insights:
  - Steady increase in the total number of flights from 2017 to 2018.
  - Indicates a growing customer base or increased travel frequency.
- File: `total_flights_per_year.png`

### 2. Average Distance Traveled per Loyalty Card Type
- Description: Calculates the average distance traveled by customers based on their loyalty card type.
- Graph: A bar chart displaying the average distance for each loyalty card type.
- Insights:
  - Aurora cardholders travel the most on average.
  - Small differences between the average distances for each card type.
- File: `avg_distance_per_card.png`

### 3. Monthly Flight Activity Over Time by Loyalty Card Type
- Description: Analyzes monthly flight activity for different loyalty card types over time.
- Graph: A line plot showing the monthly flight activity for Aurora, Nova, and Star cardholders.
- Insights:
  - Star cardholders have the highest flight activity.
  - All card types show peaks around mid-2017 and mid-2018, with dips towards the end of each year.
- File: `monthly_flight_activity_by_loyalty_card.png`

## Recommendations
- Engagement Strategies:
  - Star Cardholders: Maintain loyalty through exclusive benefits and rewards.
  - Nova Cardholders: Offer targeted promotions during low activity periods.
  - Aurora Cardholders: Continue efforts to engage and increase their flight activity.

- Seasonal Promotions:
  - Introduce mid-year promotions to capitalize on peak travel periods.
  - Implement end-of-year campaigns to mitigate the dip in flight activity.

- Further Analysis:
  - Investigate reasons behind the low points redemption rate.
  - Conduct deeper analysis to understand factors influencing travel patterns.

## Conclusion

This analysis provides valuable insights into customer behavior and loyalty program engagement. By leveraging these insights, the airline can enhance its loyalty programs, design effective marketing strategies, and increase customer satisfaction and engagement.