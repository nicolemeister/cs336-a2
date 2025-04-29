import os
import pandas as pd
import glob

def combine_and_save_latex(num_warmups, use_mixed_precision):

    # Get all CSV files with the specified pattern
    csv_files = glob.glob(f"*_num_warmups_{num_warmups}_mixed_precision_{use_mixed_precision}.csv")
    
    # Initialize an empty list to store dataframes
    dfs = []
    
    # Read each CSV file and append to the list
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        # change the column names to the following
        combined_df.columns = ['Model', 'Warmup Steps', 'Forward (mean)', 'Backward (mean)', 'Forward (std)', 'Backward (std)']

        # change the model row order to small medium large xl 2.7B
        combined_df['Model'] = pd.Categorical(combined_df['Model'], categories=['small', 'medium', 'large', 'xl', '2.7B'], ordered=True)
        combined_df = combined_df.sort_values('Model')

        # Save the combined dataframe as LaTeX
        combined_df.to_latex(f"warmup_{num_warmups}_mixed_precision_{use_mixed_precision}_table.txt", index=False)
        print(f"Combined {len(dfs)} files for num_warmups={num_warmups} and saved as LaTeX")
    else:
        print(f"No files found for num_warmups={num_warmups}")

# Process files for num_warmups=0 and num_warmups=5
# combine_and_save_latex(0)
# combine_and_save_latex(5)
# combine_and_save_latex(2)

combine_and_save_latex(5, "True")
