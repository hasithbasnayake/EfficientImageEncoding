from train_model import train_model
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train',
        description='Script that trains, decodes, and plots with given flags'
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-s", "--start", type=Path, metavar="CONFIG",
                        help="Path to config.json to start a new run")
    mode.add_argument("-r", "--resume", type=Path, metavar="RUN_DIR",
                        help="Path to model_directory to continue a run")
    mode.add_argument("-ex", "--extend", type=Path, metavar="RUN_DIR",
                        help="Path to model_directory to extend a run")
    mode.add_argument("-ev", "--eval", type=Path, metavar="RUN_DIR",
                        help="Path to model_directory/model to evaluate")

    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--epoch", type=int)

    args = parser.parse_args()

    if args.extend is None and args.epochs is not None:
        parser.error("--epochs only applies to --extend")

    if args.start is None and args.name is not None:
        parser.error("--name only applies to --start")

    if args.eval is None and args.epochs is not None:
        parser.error("--epochs only applies to --eval")
