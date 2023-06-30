import csv


def read_last_n_rows(file_path, n):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        data = list(csv_reader)
        headers = data[0]
        last_n_rows = data[-n:]
        return headers, last_n_rows


def write_to_csv(file_path, headers, data):
    with open(file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(headers)
        csv_writer.writerows(data)
    print(f"Data successfully written to {file_path}.")


# Specify the number of rows to read from the end
n = 1000

# Specify the file paths
input_file = 'input.csv'
output_file = f'last_{n}_rows.csv'

# Read the last n rows from the input file
headers, last_n_rows = read_last_n_rows(input_file, n)

# Write the last n rows to the output file
write_to_csv(output_file, headers, last_n_rows)
