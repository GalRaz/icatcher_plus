import argparse
from pathlib import Path


def parse_arguments():
    """
    parse command line arguments
    TODO: add option to parse from configuration file
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", help="The name of the experiment.")
    parser.add_argument("dataset_folder", help="The path to the folder containing the data")
    parser.add_argument("--number_of_classes", type=int, default=3, help="number of classes to predict")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size to train with")
    parser.add_argument("--image_size", type=int, default=100, help="All images will be resized to this size")
    parser.add_argument("--frames_per_datapoint", type=int, default=10, help="Number of frames in each datapoint")
    parser.add_argument("--frames_stride_size", type=int, default=2, help="Stride between frames")
    parser.add_argument("--eliminate_transitions", action="store_true",
                        help="If true, does not use frames where transitions occur (train only!)")
    parser.add_argument("--lr", type=int, default=1e-5, help="Initial learning rate")
    parser.add_argument('--lr_policy', type=str, choices=["lambda", "plateau"],
                        default='plateau',
                        help='learning rate scheduler policy')
    parser.add_argument("--lr_decay_rate", type=int, default=0.98, help="Decay rate for lamda lr policy.")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from latest iteration")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to train with")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--tensorboard",
                        help="If present, writes training stats to this path (readable with tensorboard)")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    args.dataset_folder = Path(args.dataset_folder)
    if args.tensorboard:
        args.tensorboard = Path(args.tensorboard)

    # add some useful arguments for the rest of the code
    args.experiment_path = Path(args.dataset_folder, args.experiment_name)
    if args.gpu_id == -1:
        args.device = "cpu"
    else:
        args.device = "cuda:{}".format(args.gpu_id)
    return args