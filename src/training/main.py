import argparse
import os, json

def main():
    parser = argparse.ArgumentParser(
        "Runs training of a segmentation model."
    )
    parser.add_argument(
        '--image-shape', type=str, required=True,
        help="MRI input shape. '160,192,80' or '256,256,48'. Has to match with the --tfrecord-paths argument."
    )
    parser.add_argument(
        '--tfrecord-paths', type=str, required=True,
        help="Path of a file containing comma-separated GZIP-compressed tfrecord paths for training. Each tfrecord entry has two keys : 'mri', 'lesion'."
    )
    parser.add_argument(
        '--batch-size', type=int, required=True
    )
    parser.add_argument(
        '--steps-per-epoch', type=int, required=True,
        help="User has to compute the integer division of the number of entries in --tfrecords-paths divided by --batch-size."
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
        '--n-epochs', type=int, default=200,
        help="(default : %(default)s)"
    )
    parser.add_argument(
        '--verbose', type=int, default=2,
        help="0: no log ; 1: log at each epoch; 2: log at each step %(default)s"
    )
    parser.add_argument(
        '--no-std-output', action='store_true',
        help="Prevents log on standard output."
    )
    args = parser.parse_args()

    with open(args.tfrecord_paths, 'r') as f:
        tfrecord_paths = [p.strip() for p in f.read().split(',') if p.strip()]

    try:
      shape = [int(x.strip()) for x in args.image_shape.split(',')]
    except ValueError:
      parser.error(f"--image-shape must be comma-separated integers, got: {args.image_shape!r}")
    os.environ['IMAGE_SHAPE'] = json.dumps(shape)
    
    from .run_training import train_supervised
    train_supervised(
        tfrecord_paths=tfrecord_paths,
        unet_type=args.unet_type,
        loss_str=args.loss,
        output_dir=args.output_dir,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        verbose=args.verbose,
        std_output = not args.no_std_output
    )

if __name__ == '__main__': main()