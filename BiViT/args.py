import argparse
import sys


# args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch project training')

    parser.add_argument(
        "--dataset",
        help="choose dataset",
        type=str,
        default="SpokenArabicDigits"
    )

    parser.add_argument(
        "--multigpu",
        help="use one or more gpu(s)",
        type=int,
        nargs='+',
        default=[2]
    )

    parser.add_argument(
        "--batch_size",
        help="mini-batch size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lr",
        help="learning rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--epoch",
        help="training epoch",
        type=int,
        default=200
    )

    parser.add_argument(
        "--dropout",
        help="dropout rate",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--maxL2",
        help="max L2 norm allowed (None for not employ)",
        type=float,
        default=4.0
    )

    parser.add_argument(
        "--wd",
        help="weight decay",
        type=float,
        default=0.0
    )

    parse = parser.parse_args()

    return parse


args = parse_arguments()
