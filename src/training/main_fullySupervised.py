import argparse
import os, json

def main():
    parser = argparse.ArgumentParser(
        "Runs fully-supervised training of a segmentation model (ISLA sub-model)."
    )
    parser.add_argument(
        '--image-shape', type=str, required=True,
        help="MRI input shape. '160,192,80' or '256,256,48'. Has to match with the --tfrecord-paths argument."
    )
    parser.add_argument(
        '--tfrecord-paths', type=str, required=True,
        help="Path of a file containing comma-separated GZIP-compressed tfrecord paths for training. Each tfrecord entry has two keys : 'mris', 'lesion'."
    )
    parser.add_argument(
        '--unet-type', type=str, required=True,
        help="'StdUNet' or 'ResUNet'"
    )
    parser.add_argument(
        '--loss', type=str, required=True,
        help="'DiceBCE' or 'UFL'"
    )
    parser.add_argument(
        '--output-dir', type=str, required=True
    )
    parser.add_argument(
        '--n-samples', type=int, required=True,
        help="User has to provide the number of entries in --tfrecords-path."
    )
    parser.add_argument(
        '--n-epochs', type=int, default=200,
        help="(default : %(default)s)"
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help="(default : %(default)s)"
    )
    parser.add_argument(
        '--verbose', type=int, default=2,
        help="0: no log ; 1: log at each epoch; 2: log at each step (default = %(default)s)"
    )
    parser.add_argument(
        '--no-std-output', action='store_true',
        help="Prevents logs on standard output (activated by default)."
    )
    args = parser.parse_args()

    with open(args.tfrecord_paths, 'r') as f:
        tfrecord_paths = [p.strip() for p in f.read().split(',') if p.strip()]

    try:
      shape = [int(x.strip()) for x in args.image_shape.split(',')]
    except ValueError:
      parser.error(f"--image-shape must be comma-separated integers, got: {args.image_shape!r}")
    os.environ['IMAGE_SHAPE'] = json.dumps(shape)

    from .run_training_fullySupervised import train_fullySupervised
    train_fullySupervised(
        tfrecord_paths=tfrecord_paths,
        unet_type=args.unet_type,
        loss_str=args.loss,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        steps_per_epoch = args.n_samples // args.batch_size,
        verbose=args.verbose,
        std_output = not args.no_std_output
    )

if __name__ == '__main__': main()