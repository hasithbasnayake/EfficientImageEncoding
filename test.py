from train_model import train_model
import argparse
from pathlib import Path


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

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
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.extend is None and args.epochs is not None:
        parser.error("--epochs only applies to --extend")

    if args.start is None and args.name is not None:
        parser.error("--name only applies to --start")

    if args.eval is None and args.epoch is not None:
        parser.error("--epoch only applies to --eval")

    if args.extend is not None and args.epochs is None:
        parser.error("--extend requires --epochs")

    if args.force and args.eval is None:
        parser.error("--force only applies to --eval")

    return args

def orchestrator(args):
    match args:
        case {'start': config_path, 'resume': None, 'extend': None, 'eval': None, 'name': run_name, 'epochs': None, 'epoch': None}:
            return config_path, run_name
        case {'start': None, 'resume': run_dir, 'extend': None, 'eval': None, 'name': None, 'epochs': None, 'epoch': None}:
            return run_dir
        case {'start': None, 'resume': None, 'extend': run_dir, 'eval': None, 'name': None, 'epochs': epochs, 'epoch': None}:
            return run_dir, epochs
        case {'start': None, 'resume': None, 'extend': None, 'eval': run_dir, 'name': None, 'epochs': epochs, 'epoch': epoch, 'force': force}:
            return run_dir, epoch, force
        case _:
            return f"Invalid arguments"


if __name__ == '__main__':

    args = parse_cli_arguments()

    print(args)
    print(vars(args))
    print(type(vars(args)))

    print(orchestrator(vars(args)))

    print(args.start)

    # train_model(args.start)
