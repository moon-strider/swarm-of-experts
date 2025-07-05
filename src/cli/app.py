import sys
from typing import Optional

from ..config.settings import settings
from ..providers.factory import ProviderFactory
from ..core.chat import ChatSession
from ..utils.terminal import clear_screen, enable_windows_ansi
from .ui import (
    print_header, print_error, print_warning, print_info,
    format_user_message, get_input_prompt, Color
)
from .animations import SpinnerAnimation


class CLIApp:
    def __init__(self):
        enable_windows_ansi()
        self.session: Optional[ChatSession] = None
        self.animation = SpinnerAnimation()
        
    def setup(self) -> bool:
        valid, error_msg = settings.validate()
        if not valid:
            print_error(error_msg)
            return False
            
        try:
            provider_name = ProviderFactory.get_provider_from_model(settings.default_model)
            if not provider_name:
                provider_name = "openai"
                
            api_key = getattr(settings, f"{provider_name}_api_key", None)
            if not api_key:
                print_error(f"No API key found for {provider_name}")
                return False
                
            provider = ProviderFactory.create(
                provider_name=provider_name,
                api_key=api_key,
                model=settings.default_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                stream=settings.stream
            )
            
            self.session = ChatSession(provider)
            return True
            
        except Exception as e:
            print_error(f"Failed to initialize provider: {e}")
            return False
            
    def run(self) -> None:
        clear_screen()
        print_header(
            "CLI Chat Assistant",
            "Type 'exit' or 'quit' to leave, 'clear' to clear screen, '/clear' to clear history"
        )
        
        while True:
            try:
                user_input = input(get_input_prompt())
                
                if not user_input.strip():
                    continue
                    
                if user_input.lower() in ['exit', 'quit']:
                    print_info("\nGoodbye!")
                    break
                    
                if user_input.lower() == 'clear':
                    clear_screen()
                    print_header(
                        "CLI Chat Assistant",
                        "Type 'exit' or 'quit' to leave, 'clear' to clear screen, '/clear' to clear history"
                    )
                    continue
                    
                if user_input.lower() == '/clear':
                    self.session.clear_history()
                    print_info("Conversation history cleared.")
                    continue
                    
                print(format_user_message(user_input))
                
                self.animation.start()
                
                first_chunk = True
                sys.stdout.write(f"\n{Color.BOLD.value}{Color.GREEN.value}Assistant:{Color.RESET.value}\n")
                sys.stdout.write(f"{Color.DIM.value}")
                
                for chunk in self.session.stream_message(user_input):
                    if first_chunk:
                        self.animation.stop()
                        first_chunk = False
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    
                sys.stdout.write(f"{Color.RESET.value}\n")
                
            except KeyboardInterrupt:
                self.animation.stop()
                print_warning("\n\nInterrupted. Type 'exit' to quit.")
                continue
                
            except Exception as e:
                self.animation.stop()
                print_error(f"An error occurred: {e}")
                
    def start(self) -> None:
        if self.setup():
            self.run()
        else:
            sys.exit(1)