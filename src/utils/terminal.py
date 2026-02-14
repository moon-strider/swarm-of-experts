import os
import platform


def clear_screen() -> None:
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def enable_windows_ansi() -> None:
    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass