import csv
from collections import Counter

# Read the CSV file
input_file = '/Users/nakarin.rue/Documents/log_analyze/log_analysis/data/filter_twitter_training.csv'

# Initialize counters for PF and ST
pf_counter = Counter()
st_counter = Counter()

with open(input_file, 'r', newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    
    # Iterate through the rows and update counters
    for row in reader:
        pf_counter[row['PF']] += 1
        st_counter[row['ST']] += 1

# Print unique values and counts for PF
print("Unique categories in PF:")
for category, count in pf_counter.items():
    print(f"{category}: {count} occurrences")

# Print unique values and counts for ST
print("\nUnique categories in ST:")
for category, count in st_counter.items():
    print(f"{category}: {count} occurrences")
