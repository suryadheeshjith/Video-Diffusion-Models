import yaml
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set the Seaborn style for better aesthetics
sns.set(style="whitegrid")

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def process_directory_for_csv(directory, keys_to_keep):
    data = []
    for path in Path(directory).rglob('vid_metrics.yml'):
        file_data = read_yaml(path)
        filtered_data = {key: round(file_data[key], 4) for key in keys_to_keep if key in file_data}

        # Extract the last four segments of the file path
        file_path_segments = str(path).split('/')
        short_file_path = file_path_segments[-4]
        filtered_data['file_path'] = short_file_path

        data.append(filtered_data)

    df = pd.DataFrame(data)

    # Move file_path to the first column
    cols = ['file_path'] + [col for col in df if col != 'file_path']
    df = df[cols].sort_values(by='file_path')

    return df

def create_csv(df, output_directory):
    output_path = Path(output_directory) / 'aggregated_data.csv'
    df.to_csv(output_path, index=False)
    print(f"CSV file created at: {output_path}")




if __name__ == "__main__":
    
    keys = [
        "gen_fvd", "interp_fvd", "interp_lpips", "interp_mse", 
        "interp_psnr", "interp_ssim", "pred_fvd", "pred_lpips", 
        "pred_mse", "pred_psnr", "pred_ssim"
    ]

    # Specify the root directory path
    directory_path = '/scratch/pt2310/llvm-project/DiffusionModels/main2_convert2'

    # Process the directory and get data in DataFrame
    df = process_directory_for_csv(directory_path, keys)

    # Create and save the CSV file
    create_csv(df, directory_path)

    # Create a figure and axis to plot the table
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.4)) # Adjust figure size as needed
    ax.axis('tight')
    ax.axis('off')

    # Create the table and adjust layout
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc = 'center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # Adjust font size as needed
    table.auto_set_column_width(col=list(range(len(df.columns)))) # Adjust to fit content

    # Save the figure
    plt.savefig(Path(directory_path) / 'table_image.png', dpi=300, bbox_inches='tight')  # Adjust dpi for resolution