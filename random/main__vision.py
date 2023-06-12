import cv2 as cv
import numpy as np
import win32gui, win32ui, win32con, win32api
import time
import sys
import matplotlib.pyplot as plt
import threading

# pip install opencv-python pyautogui pywin32


def get_screen_dimensions():
    return win32api.GetSystemMetrics(win32con.SM_CXSCREEN), win32api.GetSystemMetrics(
        win32con.SM_CYSCREEN
    )


def smooth_move(x2, y2, speed=0.001, step_size=70):
    """
    Utility function to move the mouse smoothly to a target position

    Default:
        speed = 0.001
        step_size = 70
    """
    x1, y1 = win32api.GetCursorPos()  # get current position
    mouse_down()
    dx = x2 - x1  # calculate horizontal distance to cover
    dy = y2 - y1  # calculate vertical distance to cover
    steps = (
        max(abs(dx), abs(dy)) // step_size
    )  # calculate number of steps, considering the step size

    if steps == 0:  # if the cursor is already at the target position
        return

    for i in range(steps):
        x = x1 + (dx * i) // steps  # calculate intermediate x
        y = y1 + (dy * i) // steps  # calculate intermediate y
        win32api.SetCursorPos((x, y))  # move the cursor to the intermediate position
        time.sleep(speed)  # wait a small amount of time before next step

    win32api.SetCursorPos(
        (x2, y2)
    )  # ensure cursor ends up exactly at the target position
    mouse_up()


def move_mouse_to_center_screen():
    """
    Utility function to move the mouse to the center of the screen
    """
    while True:
        print(f"Mouse position: {win32api.GetCursorPos()}")
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

        center_x = screen_width // 2
        center_y = screen_height // 2

        win32api.SetCursorPos((center_x, center_y))

        time.sleep(3)


def rotate_image(image, angle=0.0, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    rotMat = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(image, rotMat, (w, h))


def mouse_down():
    """
    Utility function to press the left mouse button down
    """
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)  # press left button down


def mouse_up():
    """
    Utility function to release the left mouse button
    """
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)  # release left button


class Vision:
    """
    TODO: A way to set the regions of interest
    """

    def __init__(self, hwnd):  # TODO: handle exceptions
        self.hwnd = hwnd
        self.wDC = win32gui.GetWindowDC(self.hwnd)
        self.dcObj = win32ui.CreateDCFromHandle(self.wDC)
        self.cDC = self.dcObj.CreateCompatibleDC()

    def __del__(self):
        self.dcObj.DeleteDC()
        self.cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.wDC)

    def get_current_window_position(self):
        """
        Utility function to get the current position of the window
        """
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        return left, top, right, bottom

    def move_mouse_to_center_window(self):
        """
        Utility function to move the mouse to the center of the window

        TODO: Add paddings for the window title bar and the window border
        TODO: Implement a region inside the window to bound the mouse movement, so that the mouse move to the center of the region instead
        """
        while True:
            left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
            center_x = (left + right) // 2
            center_y = (top + bottom) // 2

            smooth_move(center_x, center_y)
            time.sleep(3)

    def _window_capture(self, w=1280, h=720, offset_top=38, offset_left=8):
        border_offset_top = offset_top
        border_offset_left = offset_left

        left, top, right, bottom = self.get_current_window_position()
        print(f"Target window position: {left}, {top}, {right}, {bottom}")

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

    def capture(
        self, mode="continuous", width=1920, height=1080, color_mode="rgb", **kwargs
    ):
        cv.namedWindow("Window Capture", cv.WINDOW_NORMAL)

        if mode == "one":
            screenshot = self._window_capture(
                w=width,
                h=height,
                offset_top=kwargs["offset_top"],
                offset_left=kwargs["offset_left"],
            )
            screenshot = cv.cvtColor(screenshot, cv.COLOR_RGBA2RGB)
            if color_mode == "gray":
                screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2GRAY)
            cv.imshow("Window Capture", screenshot)
            cv.waitKey(0)

        elif mode == "continuous":
            loop_time = cv.getTickCount()

            # Starting a new thread for the mouse movement function
            # so that it doesn't block the main thread
            move_mouse_thread = threading.Thread(
                target=self.move_mouse_to_center_window
            )
            move_mouse_thread.daemon = True
            move_mouse_thread.start()
            # Also, it's a daemon thread so that it stops when the main thread stops

            while True:
                screenshot = self._window_capture(
                    w=width,
                    h=height,
                    offset_top=kwargs["offset_top"],
                    offset_left=kwargs["offset_left"],
                )
                screenshot = cv.cvtColor(screenshot, cv.COLOR_RGBA2RGB)
                if color_mode == "gray":
                    screenshot = cv.cvtColor(screenshot, cv.COLOR_RGB2GRAY)

                loop_time = self._display_fps(screenshot, loop_time)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    """
                    Press q to quit the program
                    TODO: This doesn't work if the view window is not in focus
                    """
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
