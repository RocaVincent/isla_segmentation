import argparse
import pandas as pd

from .run_inference import segment_and_save

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cases-csv', type=str, required=True,
        help='CSV listing the preprocessed MR images, the reference paths, the padding/cropping values, and the output paths. See README for exact required columns.'
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
    model_inputs = []
    for _,row in df.iterrows():
        d = {v:row[v] for v in ('dwi_path_shape1','adc_path_shape1','dwi_path_shape2','adc_path_shape2','reference_path')}
        d['padcrops1'] = {s:row[f'padcrop_{s}1'] for s in ('r','l','p','a','i','s')}
        d['padcrops2'] = {s:row[f'padcrop_{s}2'] for s in ('r','l','p','a','i','s')}
        model_inputs.append(d)
    output_paths = df.output_path.tolist()
        
    df = pd.read_csv(args.configs_weights_csv)
    config_paths = df.config_path.tolist()
    weights_paths = df.weights_path.tolist()
    
    segment_and_save(
        model_inputs = model_inputs,
        output_paths = output_paths,
        config_paths = config_paths,
        weights_paths = weights_paths,
        use_gpu = not args.no_gpu
    )     
    
if __name__ == '__main__': main()