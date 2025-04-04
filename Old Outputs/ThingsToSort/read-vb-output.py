import csv

# Input and output file paths
input_file = "EPSC_Simulation"  # Replace with your file path
output_file = "formatted_output.csv"  # Replace with desired output file name

# Read and process the file
headers = []
data = {}

with open(input_file, 'r') as file:
    for line in file:
        # Split the line to separate the header from the values
        if '"' in line:
            parts = line.split('"')
            if len(parts) >= 3:
                header = parts[1]
                values = parts[2].strip(', \n').split(',')

                # Add header if not already in the list
                if header not in headers:
                    headers.append(header)

                # Store values under the corresponding header
                if header not in data:
                    data[header] = []
                data[header].extend(values)

# Ensure all columns have equal length by padding with empty strings
max_length = max(len(values) for values in data.values())
for header in headers:
    data[header].extend([""] * (max_length - len(data[header])))

# Write the data to a CSV file
with open(output_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(headers)  # Write the headers
    for i in range(max_length):
        row = [data[header][i] for header in headers]
        csvwriter.writerow(row)  # Write each row

print(f"File formatted successfully and saved as: {output_file}")
