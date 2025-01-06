def calculate_average_sampling_time(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # List to store the calculated sampling times
    sampling_times = []

    for line in lines:
        # Find the `it/s` value in each line
        if 'it/s' in line:
            try:
                # Extract the `it/s` rate, it's the number before `it/s`
                it_per_s = float(line.split('it/s')[0].split()[-1])
                # Calculate the sampling time
                sampling_time = 1.0 / it_per_s
                # Add to list
                sampling_times.append(sampling_time)
            except (IndexError, ValueError):
                # Handle any parsing errors gracefully
                continue

    # Calculate the average sampling time
    if sampling_times:
        average_sampling_time = sum(sampling_times) / len(sampling_times)
        return average_sampling_time
    else:
        return None

# Specify the path to your data file
file_path = '/home/zhangtonghe/dppo/data.txt'

# Calculate and print the average sampling time
average_sampling_time = calculate_average_sampling_time(file_path)
if average_sampling_time is not None:
    print(f"The average sampling time is {average_sampling_time:.4f} seconds, with average samping speed {1/average_sampling_time:.4f}")
else:
    print("No valid `it/s` data found in the file.")
