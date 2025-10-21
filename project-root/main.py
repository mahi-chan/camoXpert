import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="CamoXpert - Camouflaged Object Detection")
    parser.add_argument("mode", choices=["train", "validate", "test", "inference", "benchmark"],
                        help="Mode to run the program in")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to configuration file")

    args, remaining_args = parser.parse_known_args()

    if args.mode == "train":
        from scripts.train import parser as train_parser, main as train_main
        sys.argv = ["train.py"] + remaining_args
        train_main(train_parser.parse_args())
    elif args.mode == "validate":
        from scripts.validate import parser as val_parser, main as val_main
        sys.argv = ["validate.py"] + remaining_args
        val_main(val_parser.parse_args())
    elif args.mode == "test":
        from scripts.test import parser as test_parser, main as test_main
        sys.argv = ["test.py"] + remaining_args
        test_main(test_parser.parse_args())
    elif args.mode == "inference":
        from scripts.inference import parser as inf_parser, main as inf_main
        sys.argv = ["inference.py"] + remaining_args
        inf_main(inf_parser.parse_args())
    elif args.mode == "benchmark":
        from scripts.benchmark import parser as bench_parser, main as bench_main
        sys.argv = ["benchmark.py"] + remaining_args
        bench_main(bench_parser.parse_args())
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()