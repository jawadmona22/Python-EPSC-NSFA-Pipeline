from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
import numpy as np
import Scripts.NSFA_Tools.EPSC_Graphic_Utilities as egg
import matplotlib.pyplot as plt
#Initialize some values
num_decay_segments = 20



def mean_calculation(raw_sorted, start_index, endPoint, segment_indices,analysis_type, pool_indices = None): #Peak index can also indicate start index for analysis type
    if len(raw_sorted.shape) == 1:
        decay_data = raw_sorted[start_index + 1:endPoint]
    else:
        decay_data = raw_sorted[start_index + 1:endPoint,:]  # Sorted data within the range of the decay period (peak-endPoint)
    means = []
    i = 0
    j = 0
    if analysis_type == 0:
        for i in range(len(segment_indices) - 1):
            row_lower = segment_indices[i]
            row_upper = segment_indices[i + 1]
            for j in range(len(pool_indices) - 1):
                column_lower = pool_indices[j]
                column_upper = pool_indices[j + 1]
                # Check for edge case: Single trace in a pool
                if column_lower == column_upper:
                    decay_block = decay_data[row_lower:row_upper, column_lower]
                else:
                    decay_block = decay_data[row_lower:row_upper, column_lower:column_upper]
                current_mean = np.mean(decay_block)
                means.append(current_mean)

    if analysis_type == 1:  # Individual trace means
        for i in range(num_traces):
            for j in range(len(segment_indices) - 1):
                row_lower = segment_indices[j]
                row_upper = segment_indices[j + 1]
                decay_block = decay_data[row_lower:row_upper, i]
                current_mean = np.mean(decay_block)
                means.append(current_mean)
    if analysis_type == 3:  # Indiviual pool analysis OR MA-P1-B20(pA) analysis
        for i in range(len(segment_indices) - 1):
            row_lower = segment_indices[i]
            row_upper = segment_indices[i + 1]
            if len(raw_sorted.shape) == 1:
                decay_block = decay_data[row_lower:row_upper]
            else:
                decay_block = decay_data[row_lower:row_upper, :]
            current_mean = np.mean(decay_block)
            if np.isnan(current_mean).any():
                print("NAN detected")
            means.append(current_mean)
    if analysis_type == 4:  # Individual trace means, but for pooling (Method II pool)
        means = np.zeros((len(segment_indices) - 1, num_traces))
        for i in range(num_traces):
            for j in range(len(segment_indices) - 1):
                row_lower = segment_indices[j]
                row_upper = segment_indices[j + 1]
                decay_block = decay_data[row_lower:row_upper, i]
                current_mean = np.mean(decay_block)
                means[j, i] = current_mean
    #                 means.append(current_mean)

    return means


def var_calculation(analysis_start_point, residuals_array, segment_indices, pool_indices, endPoint, analysis_type):
    decay_begin = analysis_start_point + 1  # Starts one time point after the analysis start point...
    if len(residuals_array.shape) == 1:
        print("Calculating variance of 1D data")
        decay_data = residuals_array[decay_begin:endPoint]  # total decay data
    else:
        decay_data = residuals_array[decay_begin:endPoint, :]  # total decay data

    variances = []
    # plt.close('all')
    # mean_decay_data = np.mean(decay_data,axis=0)
    # plt.plot(mean_decay_data)
    # plt.show()

    if analysis_type == 0:
        print("Analysis Type 0 Chosen")
        for i in range(len(segment_indices) - 1):
            row_lower = segment_indices[i]
            row_upper = segment_indices[i + 1]
            for j in range(len(pool_indices) - 1):
                column_lower = pool_indices[j]
                column_upper = pool_indices[j + 1]
                # Check for edge case: Single trace in a pool
                if column_lower == column_upper or len(residuals_array.shape) == 1:
                    #                     print(f"Columns {column_lower} to {column_upper}")
                    print("Edge case detected!")
                    decay_block = decay_data[row_lower:row_upper, column_lower]
                else:
                    decay_block = decay_data[row_lower:row_upper, column_lower:column_upper]
                sum_block = np.sum(decay_block)
                n = np.prod(np.shape(decay_block))
                # edge case check
                if n == 1:
                    variances.append(0)
                else:
                    variance = sum_block / (n - 1)
                    variances.append(variance)

    if analysis_type == 1:
        print("Analysis Type 1 Chosen")
        for i in range(num_traces):
            for j in range(len(segment_indices) - 1):
                row_lower = segment_indices[j]
                row_upper = segment_indices[j + 1]
                decay_block = decay_data[row_lower:row_upper, i]
                sum_block = np.sum(decay_block)
                n = np.prod(np.shape(decay_block))
                if n == 1:
                    variances.append(0)
                else:
                    variance = sum_block / (n - 1)
                    variances.append(variance)
    if analysis_type == 3:  # Indiviual pool analysis
        print("Analysis Type 3 Chosen")
        for i in range(len(segment_indices) - 1):
            row_lower = segment_indices[i]
            row_upper = segment_indices[i + 1]
            if len(residuals_array.shape) == 1:
                print("Edge Case Detected: Single trace in pool.")
                decay_block = decay_data[row_lower:row_upper]
            #                 print("Decay Block", decay_block)
            else:
                decay_block = decay_data[row_lower:row_upper, :]
            sum_block = np.sum(decay_block)
            n = np.prod(np.shape(decay_block))
            if n == 1:
                print("N=1 detected")
                variances.append(0)
            else:
                variance = sum_block / (n - 1)  #Sum of our RESIDUALS ARRAY/(n-1)
                variances.append(variance)

    if analysis_type == 4:
        print("Analysis Type: Individual Traces (Pooling) Chosen")
        variances = np.zeros((len(segment_indices) - 1, num_traces))
        for i in range(num_traces):
            for j in range(len(segment_indices) - 1):
                row_lower = segment_indices[j]
                row_upper = segment_indices[j + 1]
                decay_block = decay_data[row_lower:row_upper, i]
                sum_block = np.sum(decay_block)
                n = np.prod(np.shape(decay_block))
                if n == 1:
                    variances[j, i] = 0
                else:
                    variance = sum_block / (n - 1)
                    variances[j, i] = variance

    return variances


def create_template(data, time_duration,num_samples):
    # Generate timepoints
    timepoints = np.linspace(0, time_duration, num_samples)
    # Extract EPSC values
    EPSCs = data[:, :]  # Shape is (sample_size,number_traces)
    # Find the average EPSC from this
    print(np.isnan(EPSCs).any())
    print(np.isnan(EPSCs).sum())
    template = np.mean(EPSCs, axis=1)
    print(np.isinf(EPSCs).any())
    print(EPSCs.dtype)
    print(np.max(EPSCs), np.min(EPSCs))
    print(EPSCs.shape)
    print(f"Template shape: {template.shape}")
    nan_positions = np.where(np.isnan(EPSCs))
    print(f"Row indices with NaN: {nan_positions[0]}")
    print(f"Column indices with NaN: {nan_positions[1]}")

    # plt.plot(timepoints,template,color='black')
    # plt.plot(timepoints,EPSCs)
    # plt.title("EPSC Template debugging")
    return timepoints, template


def visualize_size_pools(EPSCs,num_pools):
    ##VISUALIZATION
    # Find the maximum values for each EPSC
    max_values = np.max(EPSCs, axis=0)

    # Calculate bin boundaries based on the maximum values of each trace
    bin_width = (np.max(max_values) - np.min(max_values)) / num_pools
    bin_boundaries = np.linspace(np.min(max_values), np.max(max_values) + bin_width, num_pools + 1)

    # Assign each trace to a bin
    trace_bins = np.digitize(max_values, bin_boundaries, right=True)
    return trace_bins

def sort_EPSCs_by_size(EPSCs,peak_index):
    raw_sorted = EPSCs[:, EPSCs[peak_index, :].argsort()]
    return raw_sorted

def sort_EPSCs_by_risetimes(EPSCs,rise_times):
    sorted_risetimes = rise_times.sort_values(by=rise_times.columns[0], ascending=True)

    # Extract the sorted index
    sorted_index = sorted_risetimes.index

    # Reorder the columns of the (500,500) dataframe
    sorted_EPSCs = EPSCs.loc[:, sorted_index]

    #Check that we have truly sorted the risetimes
    egg.rise_times_histogram_creator(EPSCs, folder_name='Scripts/', plt_show=False)

    return sorted_EPSCs
def create_pool_indices(EPSCs, peak_index,num_pools=1): #For creating pools for amplitude plots
    # Sort the raw traces by their peaks from smallest to largest
    print("Peak Index",peak_index)
    raw_sorted = EPSCs[:, EPSCs[peak_index, :].argsort()]
    mini = np.min(raw_sorted[peak_index, :])
    maxi = np.max(raw_sorted[peak_index, :])
    bin_cutoffs = np.linspace(maxi, mini, num_pools + 1)

    bin_cutoffs = np.flip(bin_cutoffs)  # Order it from smallest to largest
    # print("The pool cutoffs are: ", bin_cutoffs)  # Confirmed the same as with excel doc
    # print(raw_sorted.shape)
    bins = []
    pool_indices = []
    for cutpoint in bin_cutoffs:
        index = 0
        for value in raw_sorted[peak_index, :]:
            if value >= cutpoint:
                pool_indices.append(index)
                break
            index += 1

    print("Pool Indices:", pool_indices)  # this is the index of where the bins begin in terms of the peak row
    return pool_indices

def create_risetime_pool_indices(EPSCs,num_pools=1): #For creating pools for risetime plots
    # Sort the raw traces by their peaks from smallest to largest
    print("Peak Index",peak_index)
    raw_sorted = EPSCs[:, EPSCs[peak_index, :].argsort()]
    mini = np.min(raw_sorted[peak_index, :])
    maxi = np.max(raw_sorted[peak_index, :])
    bin_cutoffs = np.linspace(maxi, mini, num_pools + 1)

    bin_cutoffs = np.flip(bin_cutoffs)  # Order it from smallest to largest
    print("The pool cutoffs are: ", bin_cutoffs)  # Confirmed the same as with excel doc
    print(raw_sorted.shape)
    bins = []
    pool_indices = []
    for cutpoint in bin_cutoffs:
        index = 0
        for value in raw_sorted[peak_index, :]:
            if value >= cutpoint:
                pool_indices.append(index)
                break
            index += 1

    print("Pool Indices:", pool_indices)  # this is the index of where the bins begin in terms of the peak row
    return pool_indices

def objective(scale_factor, raw, average):
    max_raw = np.max(raw)
    max_average = np.max(average)
    term1 = (raw - average * scale_factor * (max_raw / max_average)) ** 2 ##Same equation for MSE as used in Jim's excel

    return np.sum(term1)


# Another option for scaling of the template, this time to make the peaks match
def peak_scaling(template, raw_trace):
    max_raw = np.max(raw_trace)
    max_template = np.max(template)
    scale_factor = max_raw / max_template
    return scale_factor


#A third option (used for ARO posted 2025) where
# the template is scaled to the value of the individual response at the time of the template peak

def scale_peak_to_individual(template,raw_trace,peak_to_peak=False):
    template_peak_index = np.argmax(template)
    template_peak_value = np.max(template)

    if peak_to_peak:
        trace_peak = np.max(raw_trace)
        scale_factor = trace_peak/template_peak_value
    else:
        trace_value_at_template_peak = raw_trace[template_peak_index]
        scale_factor = trace_value_at_template_peak/template_peak_value
    return scale_factor

def create_minimize_error_residuals(scale_factor, raw, average):  #Scaling template to value of individual response here.
    max_raw = np.max(raw)
    max_average = np.max(average)
    term1 = (raw - average * scale_factor * (max_raw / max_average)) ** 2
    return term1

def create_residual_array_template_scaled_to_individual(individual,template,peak_to_peak=False):
    scale_factor = scale_peak_to_individual(template=template,raw_trace=individual,peak_to_peak=peak_to_peak)
    error = ((template * scale_factor) - individual)**2
    return error

def create_residuals(num_traces,EPSCs_sorted,template, error_minimize=True, peak_to_peak =False,debug=False): #If peak-to-peak is true then we scale the template peak to the individual peak, not to the individual at the time of template peak
    optimized_scale_factors = []
    sum_residuals = []
    for i in range(num_traces):
        raw_data = EPSCs_sorted[:, i]
        # Define bounds for the scale factor
        scale_factor_bounds = (0.85, 1.17)
        # Find the optimal scale factor
        result = minimize_scalar(objective, bounds=scale_factor_bounds, args=(raw_data, template))
        optimal_scale_factor = result.x
        optimized_scale_factors.append(optimal_scale_factor)
        sum_residuals.append(result.fun)

    # sum_residuals = np.array(sum_residuals)  ## Equal to row 9632 of sheet 1, part 1
    optimized_scale_factors = np.array(optimized_scale_factors)
    residuals_array = np.copy(EPSCs_sorted)
    for i in range(num_traces):
        if error_minimize:
            current_scale = optimized_scale_factors[i]
            residuals_array[:, i] = create_minimize_error_residuals(current_scale, EPSCs_sorted[:, i], template)
            ##Example for debug
            if debug:
                if i == 1:
                    total_time = np.shape(EPSCs_sorted)[0] * .02
                    num_samples = np.shape(EPSCs_sorted)[0]
                    print(f"Time: {total_time} and samples: {num_samples}")
                    time = np.linspace(0,total_time,num_samples)
                    pre_MSE = np.sum((EPSCs_sorted[:, i] - template * (np.max(EPSCs_sorted[:, i]) / np.max(template))) ** 2)
                    post_MSE = np.sum((EPSCs_sorted[:, i] - template * current_scale * (np.max(EPSCs_sorted[:, i]) / np.max(template))) ** 2)

                    fig, axs = plt.subplots(1, 2)
                    axs[0].plot(time,EPSCs_sorted[:, i], color = 'red', label=f'Example EPSC')
                    axs[0].plot(time,template, color='blue',linestyle = '--', label=f'Template')
                    axs[0].text(400, 100, f'Pre-scaling MSE: {pre_MSE}', horizontalalignment='center', verticalalignment='center')
                    axs[1].text(400, 100, f'Post-Scaling MSE: {post_MSE}', horizontalalignment='center', verticalalignment='center')
                    axs[0].set_xlabel('Time (ms)', fontsize=13)
                    axs[0].set_ylabel('Current (pA)', fontsize=13)
                    axs[0].set_title('Before Scaling')
                    axs[1].plot(time,EPSCs_sorted[:, i], color='red', label=f'Example EPSC')
                    axs[1].plot(time,template * current_scale * (np.max(EPSCs_sorted[:, i]) / np.max(template)), color='blue', linestyle='--', label=f'Template')
                    axs[1].set_xlabel('Time (ms)', fontsize=13)
                    axs[1].set_ylabel('Current (pA)', fontsize=13)
                    axs[1].set_title('After Scaling')
                    axs[0].set_ylim([0,250])
                    axs[1].set_ylim([0,250])
                    plt.legend()

                    plt.savefig(f'Scaling_Validation.png')
                    plt.show()

        else:
            residuals_array[:, i] = create_residual_array_template_scaled_to_individual(EPSCs_sorted[:, i], template,peak_to_peak)
    return residuals_array
def create_segment_indices(template,start_option):

    ###Actual segments
    peak_index = np.argmax(template)

    print("The start index of the analysis is: ", start_option)
    print("The peak index of the analysis is", peak_index)
    endPoint = template.shape[0] - 1
    template_decay_range = template[peak_index:endPoint]

    peak = np.max(template_decay_range)
    minimum = np.min(template_decay_range)
    thresholds = np.linspace(peak, minimum, num_decay_segments + 1).tolist()

    # for i in range (0,20):
    #     temp_thresh -= decay_interval_width
    #     if temp_thresh < 0:
    #         temp_thresh = 0
    #     thresholds.append(temp_thresh)


    segment_indices = []
    iter_segment_index = 0
    for i in range(0,len(template_decay_range)):

        if template_decay_range[i] <= thresholds[iter_segment_index]:

            segment_indices.append(i)
            iter_segment_index+=1




        if len(segment_indices) == 21:
            break






    print("The indices of the decay segment (w.rt starting point) are: ", segment_indices)
    return segment_indices


def fitting_parabola(means,variances,force_linear=False):
    # print(f"Means: {means}")
    coefficients_2 = np.polyfit(means, variances, 2)
    coefficients_2[2] = 0 #Force zero intercept
    #Force linear if not concave
    if coefficients_2[0] > 0 or force_linear==True:
        coefficients_2 = np.polyfit(np.sort(means)[0:2],np.sort(variances)[0:2],1)
        coefficients_2[1] = 0
    fit_parabola = np.poly1d(coefficients_2)
    roots = np.roots(coefficients_2)
    # Define the derivative of the linear line
    derivative_coefficients = np.polyder(coefficients_2)

    # Create a function representing the derivative
    derivative_function = np.poly1d(derivative_coefficients)

    # Calculate the initial slope at a specific point (e.g., x=1)
    initial_slope = derivative_function(0)
    return fit_parabola,roots, initial_slope


def one_pool_analysis(EPSCs,peak_index,num_pools,endPoint,template,analysis_start_point,peak_to_peak=False):
    pool_indices = create_pool_indices(EPSCs,peak_index,num_pools)
    num_traces = EPSCs.shape[1]
    raw_sorted = sort_EPSCs_by_size(EPSCs,peak_index)
    segment_indices,residuals_array = create_segment_indices(num_traces,raw_sorted,template,analysis_start_point=analysis_start_point,peak_to_peak=peak_to_peak,)
    means = mean_calculation(raw_sorted, analysis_start_point, endPoint, segment_indices, analysis_type=3)
    vars = var_calculation(analysis_start_point, residuals_array, segment_indices, pool_indices, endPoint, 3)
    return means,vars




def quadratic_function(x, a, b):
    return a * x ** 2 + b * x


def regularized_quadratic_function(x, a, b, alpha):
    # Add L2 regularization penalty term to the objective function for the 'a' coefficient only
    regularization_term = alpha * a ** 2
    return a * x ** 2 + b * x + regularization_term


def quadratic_function(x, a, b):
    return a * x ** 2 + b * x


# Set the bounds for optimization
bounds = ([-np.inf, -np.inf], [-.0001, np.inf])

# Perform k-fold cross-validation using cross_val_score
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the range of alpha values to test
alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0]


def mse_for_alpha(alpha, means, variances):
    # Define a function to calculate mean squared error for a given alpha
    scores = []

    for train_index, test_index in kf.split(means):
        X_train, X_test = means[train_index], means[test_index]
        y_train, y_test = variances[train_index], variances[test_index]

        # Fit the model with regularization
        v_regularized = curve_fit(lambda x, a, b: regularized_quadratic_function(x, a, b, alpha), X_train, y_train,
                                  bounds=bounds)
        #         v_regularized = curve_fit(lambda x, a, b: quadratic_function(x, a, b), X_train, y_train, bounds=bounds)

        co_regularized = np.append(v_regularized[0], 0)
        fit_parab_regularized = np.poly1d(co_regularized)

        # Compute the mean squared error on the test set
        mse = np.mean((fit_parab_regularized(X_test) - y_test) ** 2)
        scores.append(mse)

    return np.mean(scores)

def multi_pool_analysis(num_pools,pool_indices,raw_sorted,residuals_array,peak_index,endPoint,segment_indices):
    multi_means_list = []
    multi_var_list = []

    # Means calculation

    for index in range(len(pool_indices) - 1):
        column_lower = pool_indices[index]
        column_upper = pool_indices[index + 1]
        if column_lower == column_upper:  # Check for edge case
            pool_segment = raw_sorted[:, column_lower]
        else:
            pool_segment = raw_sorted[:, column_lower:column_upper]
        pool_mean = mean_calculation(raw_sorted=pool_segment, start_index=peak_index, endPoint=endPoint, segment_indices=segment_indices, analysis_type=3)
        multi_means_list.append(pool_mean)

    # Variance calculation
    for index in range(len(pool_indices) - 1):
        column_lower = pool_indices[index]
        column_upper = pool_indices[index + 1]
        if column_lower == column_upper:  # Check for edge case
            pool_segment = residuals_array[:, column_lower]
        #         print("Pool segment shape:", pool_segment.shape)
        #         print("edge case detected")
        else:
            pool_segment = residuals_array[:, column_lower:column_upper]
        pool_var = var_calculation(peak_index, pool_segment, segment_indices, pool_indices, endPoint, 3)
        multi_var_list.append(pool_var)

    return multi_means_list, multi_var_list










