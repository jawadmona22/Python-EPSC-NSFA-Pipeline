import numpy as np



#Checkpoint 1: Peak Alignment
def check_peak_alignment(unprocessed_EPSCs, peak_index):
    # Iterate through columns
    processed_EPSCs = unprocessed_EPSCs.copy()
    cols_to_delete = []
    count = 0
    for col_idx, col in enumerate(processed_EPSCs.T):
        # Find the index of the maximum value in the column
        max_index = np.argmax(col)

        # Check if the index matches the specified peak_index
        if max_index != peak_index:
            # Store the column index to delete later
            cols_to_delete.append(col_idx)
            count+=1

    # Delete columns in reverse order to prevent index shifting
    for col_idx in reversed(cols_to_delete):
        processed_EPSCs = np.delete(processed_EPSCs, col_idx, axis=1)
    return processed_EPSCs,count



#Checkpoint 2: Minimum Peak Amplitude (absolute)

def check_minimum_peak_amplitude(unprocessed_EPSCs, min_pA,peak_index): #pA is the unit
    processed_EPSCs = unprocessed_EPSCs.copy()
    count = 0
    cols_to_delete = []

    for col_idx, col in enumerate(processed_EPSCs.T):
        # Find the index of the maximum value in the column
        peak_value = col[peak_index]

        # Check if the peak index is at least as large as your threshold
        if peak_value < min_pA:
            # Drop the column if the index doesn't match
            cols_to_delete.append(col_idx)
            count +=1

        # Delete columns in reverse order to prevent index shifting
    for col_idx in reversed(cols_to_delete):
        processed_EPSCs = np.delete(processed_EPSCs, col_idx, axis=1)

    return processed_EPSCs,count

def calculate_rise_time(trace, peak_index, duration_ms):
    # Find the peak value
    peak_value = trace[peak_index]

    # Calculate 10% and 90% of the peak value
    lower_threshold = 0.1 * peak_value
    upper_threshold = 0.9 * peak_value

    # Find the index where the trace crosses the lower threshold
    lower_index = np.where(trace >= lower_threshold)[0][0]

    # Find the index where the trace crosses the upper threshold after the peak
    upper_index = np.where(trace[peak_index:] <= upper_threshold)[0][0] + peak_index

    # Calculate the duration
    duration = upper_index - lower_index

    # Calculate the duration in milliseconds
    duration_ms = duration * (duration_ms / len(trace))

    return duration_ms

def check_rise_time(unprocessed_EPSCs,max_rise_time, duration_ms,peak_index):
    processed_EPSCs = unprocessed_EPSCs.copy()
    count = 0
    cols_to_delete = []
    for col_idx, col in enumerate(processed_EPSCs.T):
        rise_time = calculate_rise_time(col,peak_index,duration_ms)
        # Check if the peak index is at least as large as your threshold
        if rise_time > max_rise_time:
            # Drop the column if the index doesn't match
            count += 1
            cols_to_delete.append(col_idx)

    # Delete columns in reverse order to prevent index shifting
    for col_idx in reversed(cols_to_delete):
        processed_EPSCs = np.delete(processed_EPSCs, col_idx, axis=1)

    print(f"Dropped {count}  EPSC traces with large rise times!")
    return processed_EPSCs,count


def calculate_baseline_mean(unprocessed_EPSCs, peak_index, duration_ms):
    # Calculate the sampling rate from the duration and the length of the recording
    num_samples = np.shape(unprocessed_EPSCs)[0]
    sampling_rate = num_samples/ duration_ms
    # Calculate the index 2ms before the peak
    index_2ms_before_peak = peak_index - int(2 * sampling_rate)

    # Calculate the index 1ms before the peak
    index_1ms_before_peak = peak_index - int(sampling_rate)

    # Ensure the indices are within the bounds of the recording
    index_2ms_before_peak = max(index_2ms_before_peak, 0)
    index_1ms_before_peak = max(index_1ms_before_peak, 0)

    # Extract the segment of the recording from 2ms before the peak to 1ms before the peak
    print("Ind before peak:", index_2ms_before_peak)
    print("Ind after peak:", index_1ms_before_peak)

    segment = unprocessed_EPSCs[index_2ms_before_peak:index_1ms_before_peak]

    # Calculate the baseline mean
    baseline_mean = np.mean(segment)
    print("Baseline Mean:", baseline_mean)
    return baseline_mean


def check_return_to_base(unprocessed_EPSCs, baseline_mean, duration_ms, pA_threshold, time_threshold,peak_index):
    # Calculate the sampling rate from the duration and the length of the recording
    sampling_rate = np.shape(unprocessed_EPSCs)[0]/ duration_ms

    # Calculate the index a set threshold after the peak
    index_5ms_after_peak = int(time_threshold * sampling_rate) + peak_index

    # Extract the segment of the recording 5ms after the peak
    segment = unprocessed_EPSCs[index_5ms_after_peak:]

    # Find columns where the values don't return to within 10 of the baseline_mean
    too_high_columns = np.where(np.any(segment > baseline_mean + pA_threshold, axis=0))[0]

    # Remove outlier columns
    processed_EPSCs = np.delete(unprocessed_EPSCs, too_high_columns, axis=1)

    total_removed_columns = len(too_high_columns)

    return processed_EPSCs, total_removed_columns
