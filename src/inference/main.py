import argparse
import pandas as pd

from .run_inference import segment_and_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cases-csv', type=str, required=True,
        help="CSV listing the preprocessed MR images (reoriented, skull-stripped and bias-field corrected for DWI). Expected columns are 'dwi','adc','brain_mask','output_path'."
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
    brainMask_paths = df.brain_mask.tolist()
    output_paths = df.output_path.tolist()
        
    df = pd.read_csv(args.configs_weights_csv)
    config_paths = df.config_path.tolist()
    weights_paths = df.weights_path.tolist()
    
    segment_and_save(
        dwi_paths=dwi_paths,
        adc_paths=adc_paths,
        brainMask_paths=brainMask_paths,
        output_paths = output_paths,
        config_paths = config_paths,
        weights_paths = weights_paths,
        use_gpu = not args.no_gpu
    )     
    
if __name__ == '__main__': main()