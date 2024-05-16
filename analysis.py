from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as _sum, avg as _avg, round as _round, to_date, date_format, lit
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Initialize Spark session
spark = SparkSession.builder.appName('dibimbing').master('local').getOrCreate()

# Load datasets
calendar_df = spark.read.csv("data/calendar.csv", header=True, inferSchema=True)
customer_flight_activity_df = spark.read.csv("data/customer_flight_activity.csv", header=True, inferSchema=True)
customer_loyalty_history_df = spark.read.csv("data/customer_loyalty_history.csv", header=True, inferSchema=True)

# Inspect data
calendar_df.show(5)
customer_flight_activity_df.show(5)
customer_loyalty_history_df.show(5)

# Data Cleaning & Transformation
customer_flight_activity_df = customer_flight_activity_df.na.fill(0)
customer_loyalty_history_df = customer_loyalty_history_df.na.fill({'Salary': 0, 'customer_lifetime_value': 0})

# Join datasets if necessary for analysis
data_df = customer_flight_activity_df.join(customer_loyalty_history_df,
                                           customer_flight_activity_df["loyalty_number"] == customer_loyalty_history_df["loyalty_number"],
                                           how="inner")

# Inspect the joined dataset
data_df.printSchema()
data_df.show(5)

# Analysis 1: Total Flights per Year
total_flights_per_year = data_df.groupBy("year").agg(_sum("total_flights").alias("total_flights")).orderBy("year")
total_flights_per_year.show()

# Convert to Pandas for plotting
total_flights_per_year_df = total_flights_per_year.toPandas()
plt.figure(figsize=(10, 6))
plt.plot(total_flights_per_year_df['year'], total_flights_per_year_df['total_flights'], marker='o')
plt.title('Total Flights per Year')
plt.xlabel('Year')
plt.ylabel('Total Flights')
plt.grid(True)
plt.savefig("total_flights_per_year.png")
plt.show()

# Analysis 2: Average Distance Traveled per Loyalty Card Type
avg_distance_per_card = data_df.groupBy("loyalty_card").agg(_avg("distance").alias("avg_distance")).orderBy("loyalty_card")
avg_distance_per_card.show()

# Convert to Pandas for plotting
avg_distance_per_card_df = avg_distance_per_card.toPandas()
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_distance_per_card_df['loyalty_card'], avg_distance_per_card_df['avg_distance'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Average Distance Traveled per Loyalty Card Type')
plt.xlabel('Loyalty Card Type')
plt.ylabel('Average Distance')
plt.ylim(1920, 1980)  # Adjust y-axis limits to spread out the values
plt.grid(True)

# Add data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 2), ha='center', va='bottom')

plt.savefig("avg_distance_per_card.png")
plt.show()

# Analysis 3: Flight Activity Over Time by Loyalty Card Type
monthly_flight_activity = data_df.groupBy("year", "month", "loyalty_card").agg(_sum("total_flights").alias("total_flights")).orderBy("year", "month", "loyalty_card")
monthly_flight_activity.show()

# Convert to Pandas for plotting
monthly_flight_activity_df = monthly_flight_activity.toPandas()

# Pivot Data
pivot_df = monthly_flight_activity_df.pivot(index=['year', 'month'], columns='loyalty_card', values='total_flights').fillna(0)
pivot_df.reset_index(inplace=True)

# Combine year and month into a datetime column for easier plotting
pivot_df['date'] = pd.to_datetime(pivot_df[['year', 'month']].assign(day=1))

# Plot
plt.figure(figsize=(14, 8))
sns.lineplot(data=pivot_df, x='date', y='Aurora', label='Aurora', marker='o')
sns.lineplot(data=pivot_df, x='date', y='Nova', label='Nova', marker='o')
sns.lineplot(data=pivot_df, x='date', y='Star', label='Star', marker='o')
plt.title('Monthly Flight Activity Over Time by Loyalty Card Type')
plt.xlabel('Date')
plt.ylabel('Total Flights')
plt.legend(title='Loyalty Card Type')
plt.grid(True)
plt.savefig("monthly_flight_activity_by_loyalty_card.png")
plt.show()

# Exporting the analysis results to CSV
output_dir = "analysis_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

total_flights_per_year.write.csv(os.path.join(output_dir, "total_flights_per_year.csv"), header=True)
avg_distance_per_card.write.csv(os.path.join(output_dir, "avg_distance_per_card.csv"), header=True)
monthly_flight_activity.write.csv(os.path.join(output_dir, "monthly_flight_activity.csv"), header=True)

print("Analysis and export complete.")
