import pandas as pd

# Read the CSV file
csv_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/filter_twitter_training.csv'  # Replace with the actual file name
df = pd.read_csv(csv_file)
# Group by "PF" and count occurrences of each unique value in "ST"
grouped_data = df.groupby('PF')['ST'].value_counts()

print("Category-wise counts of unique values in ST:")
for (category, st_value), count in grouped_data.items():
    print(f"Category: {category}, ST: {st_value}, Count: {count}")
