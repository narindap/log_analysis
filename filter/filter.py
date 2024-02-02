import csv
import random

# Read the original CSV file
input_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/twitter_training.csv'
output_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/filter_twitter_training.csv'
pf_mapping_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/pf_mapping.csv'

# Number of rows to randomly select
limit_rows = 10000

with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile, \
        open(pf_mapping_file, 'w', newline='', encoding='utf-8') as mapping_file:

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

    # Identify unique values in the "PF" column
    unique_pf_values = set(row[1] for row in all_rows)

    # Create a mapping for cleaning
    # pf_mapping = {value: f'p{i}' for i, value in enumerate(unique_pf_values)}

    # Write PF Mapping to a new CSV file
    # mapping_writer = csv.writer(mapping_file)
    # mapping_writer.writerow(['Original_Value', 'Cleaned_Value'])
    # for original_value, cleaned_value in pf_mapping.items():
    #     mapping_writer.writerow([original_value, cleaned_value])

    # Print the pf_mapping
    # print("PF Mapping:")
    # for original_value, cleaned_value in pf_mapping.items():
    #     print(f"{original_value} -> {cleaned_value}")

    # Iterate through the selected rows and add random sentiment, category, and clean "PF" column
    for row in selected_rows:
        # Clean the data in the "PT" column
        row[2] = 'N' if row[2] == 'Negative' else row[2]
        row[2] = 'P' if row[2] == 'Positive' else row[2]
        row[2] = 'I' if row[2] == 'Irrelevant' else row[2]
        row[2] = 'O' if row[2] == 'Neutral' else row[2]
        # Clean the data in the "PF" column based on the mapping
        # row[1] = pf_mapping.get(row[1], row[1])

        writer.writerow(row)

print(f"New CSV file with random sentiments and categories created successfully. Randomly selected {min(limit_rows, len(all_rows))} rows.")
