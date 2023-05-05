import cv2 as cv
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import sys
import matplotlib.pyplot as plt

# pip install opencv-python pyautogui pywin32


def get_screen_dimensions():
    return win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(
        win32con.SM_CYSCREEN
    )


def rotate_image(image, angle=0.0, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    rotMat = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(image, rotMat, (w, h))


class Vision:
    def __init__(self, hwnd):  # TODO: handle exceptions
        self.hwnd = hwnd
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def __del__(self):
        self.dcObj.DeleteDC()
        self.cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.wDC)

    def _window_capture(self, w=1280, h=720):
        border_offset_top = 38
        border_offset_left = 8

        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(self.dcObj, w, h)
        self.cDC.SelectObject(dataBitMap)

        self.cDC.BitBlt(
            (0, 0),
            (w, h),
            self.dcObj,
            (border_offset_left, border_offset_top),
            win32con.SRCCOPY,
        )

        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img: np.ndarray[np.float64] = np.frombuffer(signedIntsArray, dtype=np.uint8)
        img.shape = (h, w, 4)

        win32gui.DeleteObject(dataBitMap.GetHandle())

        return img

    def capture(self, mode="continuous", width=1920, height=1080, color_mode="rgb"):
        cv.namedWindow("Window Capture", cv.WINDOW_NORMAL)

        if mode == "one":
            screenshot = self._window_capture(w=width, h=height)
            screenshot = cv.cvtColor(screenshot, cv.COLOR_RGBA2RGB)
            if color_mode == "gray":
                screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2GRAY)
            cv.imshow("Window Capture", screenshot)
            cv.waitKey(0)

        elif mode == "continuous":
            loop_time = cv.getTickCount()

            while True:
                screenshot = self._window_capture(w=width, h=height)
                screenshot = cv.cvtColor(screenshot, cv.COLOR_RGBA2RGB)
                if color_mode == "gray":
                    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2GRAY)

                loop_time = self._display_fps(screenshot, loop_time)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    print("\n")
                    break

        cv.destroyAllWindows()
        print("Done.")

    @staticmethod
    def _log_fps(refresh_interval=30, loop_time=0):
        """Log the FPS to the console.

        Args:
            `refresh_interval` (int, optional): _description_. Defaults to 30.

            `loop_time` (int, optional): _description_. Defaults to 0.

        """
        fps = cv.getTickFrequency() / (cv.getTickCount() - loop_time)

        # Refresh the line @ refresh_interval 30 per second
        if int(time.time() * 1000) % (2 * refresh_interval) < refresh_interval:
            sys.stdout.write("\r[FPS: {:>3}]".format(int(fps)))
            sys.stdout.flush()

        fps_cap = 60
        sleep_time = max(0, (1 / fps_cap) - (1 / fps))
        time.sleep(sleep_time)

        loop_time = cv.getTickCount()
        return loop_time

    @staticmethod
    def _display_fps(screenshot, loop_time):
        fps = int(cv.getTickFrequency() / (cv.getTickCount() - loop_time))
        fps_text = "FPS: {}".format(fps)
        font = cv.FONT_HERSHEY_SIMPLEX
        org = (10, 30)  # top-left corner
        font_scale = 0.75
        color = (0, 255, 0)  # green
        thickness = 2

        cv.putText(screenshot, fps_text, org, font, font_scale, color, thickness)
        cv.imshow("Window Capture", screenshot)

        loop_time = cv.getTickCount()
        return loop_time

    @staticmethod
    def show_local_image(image_path, color_mode="rgb"):
        img = cv.imread(image_path)
        if img is None:
            print("Image not found or unable to read.")
            return

        screen_width, screen_height = get_screen_dimensions()

        img_height, img_width = img.shape[:2]
        scale_ratio = min(screen_width / img_width, screen_height / img_height)
        new_width, new_height = int(img_width * scale_ratio), int(
            img_height * scale_ratio
        )

        resized_img = cv.resize(
            img, (new_width, new_height), interpolation=cv.INTER_AREA
        )

        if color_mode == "gray":
            # Converting an image to grayscale takes the 'brightness' of each pixel but ignores the color information. In the case of detecting colored objects, this is not ideal.
            resized_img_gray = cv.cvtColor(resized_img, cv.COLOR_RGB2GRAY)
            cv.namedWindow("Gray Scale", cv.WINDOW_NORMAL)
            cv.imshow("Gray Scale", resized_img_gray)

        colors = ("b", "g", "r")
        for i, col in enumerate(colors):
            histogram = cv.calcHist([resized_img], [i], None, [256], [0, 256])
            plt.figure("Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            plt.plot(histogram, color=col)
            plt.xlim([0, 256])
        plt.show()

        #! Performance Blurring
        # blur = cv.medianBlur(resized_img_gray, 3)
        blur = cv.bilateralFilter(resized_img, 5, 75, 75)
        blur = cv.GaussianBlur(resized_img_gray, (3, 3), cv.BORDER_DEFAULT)

        #! Thresholding
        # Regular thresholding
        ret, thresh = cv.threshold(blur, 45, 128, cv.THRESH_BINARY)

        # Adaptive thresholding
        adaptive_thresh = cv.adaptiveThreshold(
            resized_img_gray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            11,
            7,
        )
        cv.namedWindow("Adaptive Threshold", cv.WINDOW_NORMAL)
        cv.imshow("Adaptive Threshold", adaptive_thresh)

        #! Find contours of the thresholded image
        contours, hierarchy = cv.findContours(
            adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        print(f"Number of contours: {len(contours)}")

        # cv.namedWindow("Threshold", cv.WINDOW_NORMAL)
        # cv.imshow("Threshold", thresh)

        cv.drawContours(resized_img, contours, -1, (0, 255, 0), 2)

        color_inverted_img = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
        plt.figure(num="Contours Drawn")
        plt.imshow(color_inverted_img)
        plt.show()

        cv.waitKey(0)
        cv.destroyAllWindows()
