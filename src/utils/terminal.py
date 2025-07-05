import os
import sys
import platform
from typing import Tuple


def clear_screen() -> None:
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def get_terminal_size() -> Tuple[int, int]:
    try:
        return os.get_terminal_size()
    except:
        return (80, 24)


def supports_color() -> bool:
    if platform.system() == 'Windows':
        return True
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def enable_windows_ansi() -> None:
    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass