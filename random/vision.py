import argparse
from main__vision import Vision
import win32gui


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="continuous",
        choices=["one", "continuous"],
        help="capture mode",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="window title or class name to capture",
    )
    parser.add_argument(
        "--show-image",
        type=str,
    )
    parser.add_argument(
        "--color-mode",
        type=str,
    )
    args = parser.parse_args()

    print(f"args: {args}")
    if args.target is not None:
        Vision(win32gui.FindWindow(None, args.target)).capture(
            mode=args.mode, width=1280, height=720, color_mode=args.color_mode
        )
    elif args.show_image is not None:
        Vision.show_local_image(args.show_image, color_mode=args.color_mode)
