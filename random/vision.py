import argparse
from main__vision import Vision
import win32gui

# Example usage:
# python vision.py --target "Untitled - Paint" --offset-top 274 --offset-left 46 --width 806 --height 486

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
        "--width",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )
    parser.add_argument(
        "--offset-top",
        type=int,
        default=38,
    )
    parser.add_argument(
        "--offset-left",
        type=int,
        default=8,
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
        window = win32gui.FindWindow(None, args.target)

        # Fetch the actual window size
        rect = win32gui.GetWindowRect(window)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        print(f"Actual target window size: {width} x {height}")

        Vision(window).capture(
            mode=args.mode,
            width=args.width,
            height=args.height,
            color_mode=args.color_mode,
            offset_top=args.offset_top,
            offset_left=args.offset_left,
        )
    elif args.show_image is not None:
        Vision.show_local_image(args.show_image, color_mode=args.color_mode)
