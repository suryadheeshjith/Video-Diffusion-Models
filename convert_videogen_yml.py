import yaml
from pathlib import Path

def filter_yaml(input_file, output_file, keys_to_keep):
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)

    filtered_data = {key: round(data[key],2) for key in keys_to_keep if key in data}

    with open(output_file, 'w') as file:
        yaml.dump(filtered_data, file)

def process_directory(directory, keys_to_keep):
    for path in Path(directory).rglob('vid_metrics.yml'):
        output_path = path.with_name('vid_metrics_converted.yml')
        filter_yaml(path, output_path, keys_to_keep)

if __name__ == "__main__":
    keys = [
        "gen_fvd", "pred_fvd", "pred_lpips", "pred_mse", "pred_psnr", "pred_ssim", 
        "interp_fvd", "interp_lpips", "interp_mse", "interp_psnr", "interp_ssim",
    ]

    # Specify the root directory path
    directory_path = '/scratch/pt2310/llvm-project/DiffusionModels/main2_convert2'
    process_directory(directory_path, keys)
