import sys
from enum import Enum
from typing import Optional


class Color(Enum):
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class Theme:
    USER_PREFIX = f"{Color.BOLD.value}{Color.BLUE.value}You:{Color.RESET.value}"
    ASSISTANT_PREFIX = f"{Color.BOLD.value}{Color.GREEN.value}Assistant:{Color.RESET.value}"
    USER_COLOR = Color.BLUE.value
    ASSISTANT_COLOR = f"{Color.DIM.value}"
    ERROR_COLOR = Color.RED.value
    WARNING_COLOR = Color.YELLOW.value
    INFO_COLOR = Color.CYAN.value
    SUCCESS_COLOR = Color.GREEN.value
    
    @property
    def input_prompt(self) -> str:
        return f"{Color.BOLD.value}{Color.BLUE.value}>{Color.RESET.value}"
        
    def primary(self, text: str) -> str:
        return f"{Color.CYAN.value}{text}{Color.RESET.value}"
        
    def assistant_text(self, text: str) -> str:
        return f"{self.ASSISTANT_COLOR}{text}{Color.RESET.value}"
    

def format_user_message(message: str) -> str:
    return f"\n{Theme.USER_PREFIX} {message}"


def format_assistant_message(message: str) -> str:
    return f"\n{Theme.ASSISTANT_PREFIX}\n{Theme.ASSISTANT_COLOR}{message}{Color.RESET.value}"


def print_error(message: str) -> None:
    print(f"{Theme.ERROR_COLOR}Error: {message}{Color.RESET.value}", file=sys.stderr)


def print_warning(message: str) -> None:
    print(f"{Theme.WARNING_COLOR}Warning: {message}{Color.RESET.value}")


def print_info(message: str) -> None:
    print(f"{Theme.INFO_COLOR}{message}{Color.RESET.value}")


def print_success(message: str) -> None:
    print(f"{Theme.SUCCESS_COLOR}{message}{Color.RESET.value}")


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    print(f"{Color.BOLD.value}{Color.CYAN.value}{title}{Color.RESET.value}")
    if subtitle:
        print(f"{Color.DIM.value}{subtitle}{Color.RESET.value}")
    print("-" * 50)


def get_input_prompt() -> str:
    return f"\n{Color.BOLD.value}{Color.BLUE.value}>{Color.RESET.value} "


class UI:
    def __init__(self):
        self.theme = Theme()
        
    def print_header(self):
        print(self.theme.primary("=" * 60))
        print(self.theme.primary("🤖 Swarm of Experts - Multi-Model Chat Interface"))
        print(self.theme.primary("=" * 60))
        
    def format_user_message(self, message: str) -> str:
        return format_user_message(message)
        
    def format_assistant_message(self) -> str:
        return f"\n{Theme.ASSISTANT_PREFIX} "
        
    def print_error(self, message: str) -> None:
        print_error(message)
        
    def print_warning(self, message: str) -> None:
        print_warning(message)
        
    def print_info(self, message: str) -> None:
        print_info(message)
        
    def print_success(self, message: str) -> None:
        print_success(message)
        
    def get_input_prompt(self) -> str:
        return get_input_prompt()