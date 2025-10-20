                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts import train, validate, test, inference, benchmark


def main():
    parser = argparse.ArgumentParser(description="CamoXpert - Camouflaged Object Detection")
    parser.add_argument("mode", choices=["train", "validate", "test", "inference", "benchmark"],
                        help="Mode to run the program in")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to configuration file")

    args, remaining_args = parser.parse_known_args()

    if args.mode == "train":
        sys.argv = ["train.py"] + remaining_args
        train.main(train.parser.parse_args())
    elif args.mode == "validate":
        sys.argv = ["validate.py"] + remaining_args
        validate.main(validate.parser.parse_args())
    elif args.mode == "test":
        sys.argv = ["test.py"] + remaining_args
        test.main(test.parser.parse_args())
    elif args.mode == "inference":
        sys.argv = ["inference.py"] + remaining_args
        inference.main(inference.parser.parse_args())
    elif args.mode == "benchmark":
        sys.argv = ["benchmark.py"] + remaining_args
        benchmark.main(benchmark.parser.parse_args())
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()