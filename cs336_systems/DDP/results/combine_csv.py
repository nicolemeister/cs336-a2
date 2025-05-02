import os
import pandas as pd
import glob

def combine_and_save_latex():

    # Get all CSV files with the specified pattern
    csv_files = glob.glob(f"distributed_communication_single_node_data_size_*.csv")
    
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
        # reorder the columns to the following
        combined_df.columns = ['Backend', 'Data Size', 'Processes', 'Time (s)']

        # Convert D Model to categorical for proper ordering
        # Sort first by D Model, then by sequence length
        combined_df = combined_df.sort_values(['Data Size', 'Processes'])

        # Save the combined dataframe as LaTeX
        combined_df.to_latex(f"table.txt", index=False)
        print(f"Combined {len(dfs)} files and saved as LaTeX")


        combined_df.to_latex(f"table_single_node_GPU.txt", index=False)
    else:
        print(f"No files found")

# Process files for num_warmups=0 and num_warmups=5
# combine_and_save_latex(0)
# combine_and_save_latex(5)
# combine_and_save_latex(2)

combine_and_save_latex()
