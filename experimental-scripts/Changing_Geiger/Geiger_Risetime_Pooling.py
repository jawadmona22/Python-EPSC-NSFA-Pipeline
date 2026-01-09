import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def rise_times_histogram_creator(df,folder_name,plt_show = False,sampling_rate = .02,export_name="risetimes"):
    max_amplitudes = df.abs().max(axis=0)
    rise_times = []
    for col in df.columns:
        trace = df[col].abs()
        max_amp = trace.max()
        threshold_10 = 0.1 * max_amp
        threshold_90 = 0.9 * max_amp

        # Find where the signal first crosses the 10% threshold
        above_10 = trace >= threshold_10
        above_90 = trace >= threshold_90

        try:
            idx_10 = above_10.idxmax()  # First index where >= threshold_10
            idx_90 = above_90.idxmax()

            if idx_10 == 0:
                time = idx_90 * sampling_rate
                rise_times.append(time)
                continue  # Can't interpolate at the start of trace

            # Interpolate time_10
            prev_idx_10 = idx_10 - 1
            x0, y0 = prev_idx_10, trace[prev_idx_10]
            x1, y1 = idx_10, trace[idx_10]
            time_10_interp = x0 + (threshold_10 - y0) / (y1 - y0)

            # Interpolate time_90
            prev_idx_90 = idx_90 - 1
            x0, y0 = prev_idx_90, trace[prev_idx_90]
            x1, y1 = idx_90, trace[idx_90]
            time_90_interp = x0 + (threshold_90 - y0) / (y1 - y0)

            # Convert to seconds
            rise_time = (time_90_interp - time_10_interp) * sampling_rate
            rise_times.append(rise_time)

        except Exception as e:
            print(f"Interpolation failed for column {col}: {e}")

    print(f"Rise Time shape: {len(rise_times)}")
    print(f"Trace shape: {df.shape}")

    rise_times_df = pd.DataFrame({
        'Trace':df.columns,
        'Rise Time (10% to 90%)': rise_times
    })

    rise_times_df.to_excel(f'{export_name}.xlsx',index=False)
    if plt_show == True:
        plt.figure()
        plt.hist(rise_times, bins=10, edgecolor='black')
        plt.xlabel('Rise Times (ms)')
        plt.title(f'Rise Times Histogram')
        plt.ylabel('Frequency')
        plt.show()
        plt.savefig(f'rise_times_histogram.png')
    return rise_times

#
file_name = r'C:\Users\j.mona\Documents\GitHub\Python-EPSC-NSFA-Pipeline\data-files\uEPSCS_forNSFA_WG24726.xlsx'

df = pd.read_excel(file_name,sheet_name=0)
print(df.shape)
rise_times = rise_times_histogram_creator(df,"/",sampling_rate=.01,plt_show=True)
# print(rise_times)
# rise_times = pd.Series(rise_times, index=df.columns)
#
# #First we will make  bins for our risetimes
# num_bins = 10
# bins = pd.qcut(rise_times,q=num_bins)
# bin_categories = bins.dtype.categories
#
# ###Only for qcut####
# bin_counts = bins.value_counts().sort_index()  # sort_index to order bins properly
#
# # Plot histogram
# plt.figure(figsize=(10,5))
# plt.bar(range(len(bin_counts)), bin_counts.values, tick_label=[f"{str(interval)}" for interval in bin_counts.index])
# plt.xticks(rotation=45, ha='right')
# plt.ylabel("Number of traces")
# plt.xlabel("Rise time quantile bin")
# plt.title("Histogram of traces per quantile bin")
# plt.tight_layout()
# plt.show()
#
#
#
# print(bin_categories)
# #then we'll create an excel file with the binned columns as a tab for each
# with pd.ExcelWriter("binned_geiger_traces_quantiles.xlsx", engine="openpyxl") as writer:
#     for i, category in enumerate(bin_categories,start=1):
#         print("category:",category)
#         cols_in_bin = rise_times.index[bins == category]
#         subset = df.loc[:, cols_in_bin]
#         # risetimes_subset = rise_times_histogram_creator(df=subset,folder_name="/",plt_show=True,export_name="temp_risetimes")
#         subset.to_excel(writer, sheet_name=f"Bin_{i}", index=False,header=False)
#
#
# print("Excel file created...")


#then an excel file removing the outliers...only include rise times from
#
# with pd.ExcelWriter("binned_geiger_traces_quantiles.xlsx",engine="openpyxl") as writer:
#     cleaned_cols = rise_times.index[rise_times < .4]
#     subset = df.loc[:,cleaned_cols]
#     subset.to_excel(writer,index=False,header=False)
#
# print("Outlier removal file created...")
# subset_risetimes = rise_times_histogram_creator(subset,"/",sampling_rate=.01,plt_show=True)


#
# rise_times = pd.read_excel("cdf.xlsx")
# print(rise_times.head())
# normal = rise_times["Normal"]
# fixed = rise_times["Fixed"]
# plt.figure(figsize=(8,5))
# normal.hist(bins=10, alpha=0.5, label='Normal', color='skyblue')
# fixed.hist(bins=10, alpha=0.5, label='Fixed', color='red')
# plt.grid(False)
# plt.legend()
# plt.xlabel('Rise Times (ms)')
# plt.ylabel('Frequency')
# plt.title('Rise Times Comparison')
# plt.show()


# Load data
# df = pd.read_excel("EPSCs_test.xlsx")
# print(df.shape)
#
# # Compute rise times
# rise_times = rise_times_histogram_creator(df, "/", sampling_rate=0.01, plt_show=True)
# print(rise_times)
#
# # Convert to Series with column names as index
# rise_times = pd.Series(rise_times, index=df.columns)
#
# # Define bin width and edges
# bin_width = 0.02
# bins = np.arange(rise_times.min(), rise_times.max() + bin_width, bin_width)
#
# # Bin the rise times
# binned = pd.cut(rise_times, bins=bins, include_lowest=True)
#
# # Extract bin categories
# bin_categories = binned.cat.categories
# print("Bin categories:", bin_categories)
# bin_counts = binned.value_counts().sort_index()  # sort_index ensures bins are in order
#
# # Plot as a bar chart
# plt.figure(figsize=(10,6))
# bin_counts.plot(kind='bar', color='skyblue', edgecolor='black')
#
# plt.xlabel("Rise Time Bins (s)")
# plt.ylabel("Number of Traces")
# plt.title("Distribution of Rise Times by Bin")
# plt.xticks(rotation=45, ha='right')  # rotate bin labels
# plt.tight_layout()
# plt.show()
# # Create Excel file with a tab per bin
# with pd.ExcelWriter("binned_equal_width_test.xlsx", engine="openpyxl") as writer:
#     for i, category in enumerate(bin_categories, start=1):
#         # Get columns that fall into this bin
#         cols_in_bin = rise_times.index[binned == category]
#         if len(cols_in_bin) > 100:
#             subset = df.loc[:, cols_in_bin]
#
#
#             # Write subset to Excel
#             subset.to_excel(writer, sheet_name=f"Bin_{i}", index=False, header=False)
#
# print("Excel file created...")



import seaborn as sns

##
binned_nsfa = pd.read_excel(r"C:\Users\j.mona\Documents\GitHub\Python-EPSC-NSFA-Pipeline\Scripts\Experiments\Creating_NSFA_Debugger\testing-cell-ratio.xlsx")
print(binned_nsfa.head)

df = binned_nsfa.copy()

# Make scaling a categorical with a stable order (optional)
categories = df['scaling'].unique()            # or specify order: ['Linear','Log','Power']
df['scaling'] = pd.Categorical(df['scaling'], categories=categories, ordered=True)

# map categories to x positions
x_codes = df['scaling'].cat.codes               # 0,1,2,...

# reproducible jitter
rng = np.random.default_rng(42)
jitter_strength = 0.12                          # adjust to spread points horizontally
x_jitter = x_codes + rng.normal(0, jitter_strength, size=len(df))

# color palette (one color per category)
palette = sns.color_palette("Set2", n_colors=len(categories))
colors = [palette[c] for c in x_codes]

plt.figure(figsize=(7,5))
plt.scatter(x_jitter, df['linear_i'], c=colors, s=60, alpha=0.8)

# add text labels (Bin)
for x, y, lbl in zip(x_jitter, df['linear_i'], df['bin']):   # ensure column name is 'bin'
    plt.text(x, y + 0.01, str(lbl), ha='center', va='bottom', fontsize=8)

plt.axhline(y=1.275, color='green', linestyle='--', linewidth=2, label='Expected Current')

# cosmetics
plt.xticks(range(len(categories)), categories)
plt.xlabel("Scaling Type")
plt.ylabel("linear_i")
plt.title("linear_i by Scaling Type (Labeled by Bin)")
plt.show()
