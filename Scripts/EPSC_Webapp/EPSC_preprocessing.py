import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


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

#Actually align peaks
def align_peaks(unprocessed_EPSCs):
    # Convert to float to avoid integer truncation issues
    unprocessed_EPSCs = unprocessed_EPSCs.astype(float)

    # Create a zero-filled array of the same shape
    processed_EPSCs = np.zeros_like(unprocessed_EPSCs)
    num_rows, num_cols = processed_EPSCs.shape
    target_idx = 15

    print(f"Aligning by peaks. EPSC Peak Index = {target_idx}")

    for col_idx in range(num_cols):
        col = unprocessed_EPSCs[:, col_idx]
        max_index = np.argmax(col)  # Find peak index

        shift = target_idx - max_index  # Compute shift amount

        # print(f"\nColumn {col_idx}: Original peak index = {max_index}, Shift required = {shift}")

        # Apply np.roll() for shifting
        shifted_col = np.roll(col, shift)

        # Prevent wrapping effects
        if shift < 0:
            shifted_col[shift:] = 0  # Zero out wrapped values
        elif shift > 0:
            shifted_col[:shift] = 0  # Zero out introduced values

        # Debugging print: Before & after shift
        print(f"Original col: {col[:20]}")  # Print first 20 values
        print(f"Shifted col: {shifted_col[:20]}")  # Print first 20 values

        # Ensure assignment works correctly
        processed_EPSCs[:, col_idx] = shifted_col.copy()

        # Check if alignment is correct
        new_peak_idx = np.argmax(processed_EPSCs[:, col_idx])
        if new_peak_idx != target_idx:
            print(f"❌ Warning: Peak in col {col_idx} expected at {target_idx}, but found at {new_peak_idx}")
        else:
            print(f"✅ Success: Peak in col {col_idx} correctly aligned at {target_idx}")

    return processed_EPSCs, target_idx

def align_dv_dt(unprocessed_EPSCs,debug=False):
    # Create a copy to avoid modifying the original data
    processed_EPSCs = np.zeros_like(unprocessed_EPSCs)
    alignment_point = 15
    time = np.linspace(0,processed_EPSCs.shape[0] * .02,processed_EPSCs.shape[0])
    for col_idx, col in enumerate(unprocessed_EPSCs.T):
        # Find highest dv/dt for template
        peak_value = np.max(col)
        peak_index = np.argmax(col)
        # Calculate 10% and 90% of the peak value
        lower_threshold = 0.1 * peak_value
        upper_threshold = 0.9 * peak_value

        # Find the first index where the trace crosses the lower threshold
        lower_index = np.where(col >= lower_threshold)[0][0]

        # Find the first index where the trace crosses the upper threshold after the peak
        upper_index = np.where(col >= upper_threshold)[0][0]

        ninety_ten_region = col[lower_index:upper_index]

        biggest_difference_index = np.argmax(np.diff(ninety_ten_region)) + lower_index #Relative to 0 (full EPSC)
        #
        if debug:
            biggest_difference_index_time = biggest_difference_index * .02
            # print(f"Max DV/DT Point: {biggest_difference_index}")
            # plt.figure()
            # plt.plot(time,col) #Plot the EPSC
            # plt.xlabel("Time (ms)")
            # plt.ylabel("Current (pA)")
            # plt.axvline(x=lower_index*.02, color='r',label="10%",linestyle="--",linewidth=1)
            # plt.axvline(x=upper_index*.02, color='g',label="90%",linestyle="--",linewidth=1)
            # plt.annotate("", xytext=(biggest_difference_index_time, ninety_ten_region[int(biggest_difference_index_time-(lower_index*.02))]), xy=((biggest_difference_index_time+.02, ninety_ten_region[int(biggest_difference_index_time-(lower_index*.02))+1])),
            #             arrowprops=dict(arrowstyle="->"))
            # plt.legend()
            # plt.show()
        # Calculate the shift required to align the peak
        shift = alignment_point - biggest_difference_index
        # Apply np.roll() for shifting
        shifted_col = np.roll(col, shift)

        # Prevent wrapping effects
        if shift < 0:
            shifted_col[shift:] = 0  # Zero out wrapped values
        elif shift > 0:
            shifted_col[:shift] = 0  # Zero out introduced values

        # Debugging print: Before & after shift


        # Ensure assignment works correctly
        processed_EPSCs[:, col_idx] = shifted_col.copy()

    return processed_EPSCs

def align_midpoint(unprocessed_EPSCs):
    # Create a copy to avoid modifying the original data
    processed_EPSCs = np.zeros_like(unprocessed_EPSCs)
    alignment_point = 15
    # Iterate through columns
    for col_idx, col in enumerate(unprocessed_EPSCs.T):
        ###First neeed to find the 10-90% rise time indexes
        # Find the peak value
        peak_value = np.max(col)
        peak_index = np.argmax(col)
        # Calculate 10% and 90% of the peak value
        lower_threshold = 0.1 * peak_value
        upper_threshold = 0.9 * peak_value

        # Find the index where the trace crosses the lower threshold
        lower_index = np.where(col >= lower_threshold)[0][0]

        # Find the index where the trace crosses the upper threshold after the peak
        upper_index = np.where(col[peak_index:] >= upper_threshold)[0][0] + peak_index

        individual_midpoint = int((lower_index + upper_index)/2)

        # Calculate the shift required to align the dv/dt region
        shift = alignment_point - individual_midpoint

        shifted_col = np.roll(col, shift)

        # Prevent wrapping effects
        if shift < 0:
            shifted_col[shift:] = 0  # Zero out wrapped values
        elif shift > 0:
            shifted_col[:shift] = 0  # Zero out introduced values


        # Ensure assignment works correctly
        processed_EPSCs[:, col_idx] = shifted_col.copy()

    return processed_EPSCs

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
