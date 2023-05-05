import win32gui


def enumerate_windows():
    def callback(hwnd, titles):
        title = win32gui.GetWindowText(hwnd)
        if title:
            print(f"{hwnd}: {title}")
        return True

    win32gui.EnumWindows(callback, None)


if __name__ == "__main__":
    enumerate_windows()
