import csv
import random

# Read the original CSV file
input_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/twitter_training.csv'
output_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/filter_twitter_training.csv'
# Number of rows to randomly select
limit_rows = 5000

with open(input_file, 'r', newline='', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    header = next(reader)  # Read the header
    header.append('Sentiment')  # Add a new column header
    header.append('Category')  # Add another new column header for category

    # Create a CSV writer with the updated header
    writer = csv.writer(outfile)
    writer.writerow(header)

    # Read all lines into a list
    all_rows = list(reader)

    # Randomly select 5000 rows
    selected_rows = random.sample(all_rows, min(limit_rows, len(all_rows)))

    # Iterate through the selected rows and add random sentiment and category
    for row in selected_rows:

        writer.writerow(row)

print(f"New CSV file with random sentiments and categories created successfully. Randomly selected {min(limit_rows, len(all_rows))} rows.")
