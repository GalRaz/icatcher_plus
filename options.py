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
    parser.add_argument("--architecture", type=str, choices=["fc", "icatcher_vanilla", "icatcher+", "rnn"],
                        default="icatcher+",
                        help="Selects architecture to use")
    parser.add_argument("--loss", type=str, choices=["cat_cross_entropy"], default="cat_cross_entropy",
                        help="Selects loss function to optimize")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument('--lr_policy', type=str, choices=["lambda", "plateau", "cyclic"],
                        default='plateau',
                        help='learning rate scheduler policy')
    parser.add_argument("--lr_decay_rate", type=int, default=0.98, help="Decay rate for lamda lr policy.")
    parser.add_argument("--continue_train", action="store_true", help="Continue training from latest iteration")
    parser.add_argument("--number_of_epochs", type=int, default=100, help="Total number of epochs to train model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to train with")
    parser.add_argument("--gpu_id", type=int, default=-1, help="Which GPU to use (or -1 for cpu)")
    parser.add_argument("--num_threads", type=int, default=0, help="How many threads for dataloader")
    parser.add_argument("--tensorboard", action="store_true", help="Activates tensorboard logging")
    parser.add_argument("--log", action="store_true", help="Logs into a file instead of stdout")
    parser.add_argument("-v", "--verbosity", type=str, choices=["debug", "info", "warning"], default="info",
                        help="Selects verbosity level")
    args = parser.parse_args()
    args.dataset_folder = Path(args.dataset_folder)
    # add some useful arguments for the rest of the code
    args.experiment_path = Path("runs", args.experiment_name)
    args.experiment_path.mkdir(exist_ok=True, parents=True)
    if args.gpu_id == -1:
        args.device = "cpu"
    else:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        args.device = "cuda:{}".format(0)
    return args