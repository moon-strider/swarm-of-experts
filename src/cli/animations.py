import sys
import threading
import time
from typing import List, Optional
from abc import ABC, abstractmethod

from .ui import Color


class Animation(ABC):
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
        
    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        self._clear_line()
        
    @abstractmethod
    def _animate(self) -> None:
        pass
        
    def _clear_line(self) -> None:
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()


class SpinnerAnimation(Animation):
    def __init__(self, text: str = "Thinking", frames: Optional[List[str]] = None):
        super().__init__()
        self.text = text
        self.frames = frames or ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        
    def _animate(self) -> None:
        frame_index = 0
        while self.running:
            frame = self.frames[frame_index % len(self.frames)]
            sys.stdout.write(f'\r{Color.CYAN.value}{frame} {self.text}...{Color.RESET.value}')
            sys.stdout.flush()
            time.sleep(0.1)
            frame_index += 1


class DotsAnimation(Animation):
    def __init__(self, text: str = "Processing"):
        super().__init__()
        self.text = text
        
    def _animate(self) -> None:
        dots_count = 0
        while self.running:
            dots = "." * (dots_count % 4)
            sys.stdout.write(f'\r{Color.CYAN.value}{self.text}{dots:<4}{Color.RESET.value}')
            sys.stdout.flush()
            time.sleep(0.5)
            dots_count += 1


class ProgressAnimation(Animation):
    def __init__(self, text: str = "Loading", width: int = 20):
        super().__init__()
        self.text = text
        self.width = width
        
    def _animate(self) -> None:
        position = 0
        direction = 1
        while self.running:
            bar = [' '] * self.width
            bar[position] = '█'
            bar_str = ''.join(bar)
            sys.stdout.write(f'\r{Color.CYAN.value}{self.text} [{bar_str}]{Color.RESET.value}')
            sys.stdout.flush()
            
            position += direction
            if position >= self.width - 1 or position <= 0:
                direction *= -1
                
            time.sleep(0.05)