import argparse
import pandas as pd

from .run_inference2 import segment_and_save_raw


def main():
    parser = argparse.ArgumentParser(
        description="Run DWI/ADC segmentation with preprocessing and UNet inference."
    )
    parser.add_argument(
        '--cases-csv', type=str, required=True,
        help="CSV listing the raw MR images and the output paths. Expected columns are 'dwi','adc','output'."
    )
    parser.add_argument(
        '--configs-weights-csv' , type=str, required=True,
        help='CSV listing the configuration and weights paths of each model. Two columns: config_path, weights_path.'
    )
    parser.add_argument(
        '--no-gpu', action='store_true', help='Forces CPU inference even if GPU is available',
    )
    args = parser.parse_args()
    
    df = pd.read_csv(args.cases_csv)
    dwi_paths = df.dwi.tolist()
    adc_paths = df.adc.tolist()
    output_paths = df.output.tolist()
    
    df = pd.read_csv(args.configs_weights_csv)
    config_paths = df.config_path.tolist()
    weights_paths = df.weights_path.tolist()
    
    segment_and_save_raw(
        dwi_paths=dwi_paths,
        adc_paths=adc_paths,
        output_paths=output_paths,
        config_paths=config_paths,
        weights_paths=weights_paths,
        use_gpu = not args.no_gpu,
    )
    
if __name__ == '__main__': main()