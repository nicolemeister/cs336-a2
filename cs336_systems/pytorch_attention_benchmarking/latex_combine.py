import os
import pandas as pd
import glob

def combine_and_save_latex():

    # Get all CSV files with the specified pattern
    csv_files = glob.glob(f"*.csv")
    
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
        combined_df = combined_df.drop(columns=['end_to_end_times_std'])
        combined_df = combined_df.drop(columns=['forward_times_std'])
        combined_df = combined_df.drop(columns=['backward_times_std'])

        combined_df.columns = ['D Model', 'Seq Len', 'Compile Attention', 'Forward Time', 'Backward Time', 'End-to-End Time']
        
        # Convert D Model to categorical for proper ordering
        combined_df['D Model'] = pd.Categorical(combined_df['D Model'], ordered=True)
        # Sort first by D Model, then by sequence length
        combined_df = combined_df.sort_values(['D Model', 'Seq Len', "Compile Attention"])

        # Save the combined dataframe as LaTeX
        combined_df.to_latex(f"table.txt", index=False)
        print(f"Combined {len(dfs)} files and saved as LaTeX")

        # save a version of the dataframe where the compile model and compile attention are 0
        combined_df = combined_df[combined_df['Compile Attention'] == 0]
        
        # combined_df_0.to_csv(f"table_nocompile.csv", index=False)
        combined_df.to_latex(f"table_nocompile.txt", index=False)
    else:
        print(f"No files found")

# Process files for num_warmups=0 and num_warmups=5
# combine_and_save_latex(0)
# combine_and_save_latex(5)
# combine_and_save_latex(2)

combine_and_save_latex()
