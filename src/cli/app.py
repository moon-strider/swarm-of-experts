import asyncio

from ..config.settings import settings
from ..providers.factory import ProviderFactory
from ..core.chat import ChatSession
from ..utils.terminal import clear_screen, enable_windows_ansi
from .ui import UI
from .animations import SpinnerAnimation
from ..config.swarm_configs import SWARM_CONFIGS
from ..utils.logging import setup_logging
import logging

logger = logging.getLogger(__name__)


class CLIApp:
    def __init__(self):
        enable_windows_ansi()
        self.provider = None
        self.chat_session = None
        self.ui = UI()
        self.is_running = False
        self.swarm_config = None

        
    def setup(self):
        setup_logging()
        logger.info("Starting Swarm of Experts CLI")

        clear_screen()
        self.ui.print_header()

        is_valid, error_msg = settings.validate()
        if not is_valid:
            self.ui.print_error(error_msg or "No API keys found in environment variables")
            self.ui.print_info("Please set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, GROQ_API_KEY")
            return False

        self.ui.print_info("\nAvailable configurations:")
        for name, config in SWARM_CONFIGS.items():
            if config.has_merger:
                self.ui.print_info(f"  - {name}: {len(config.generators)} models in parallel → merged response")
            else:
                self.ui.print_info(f"  - {name}: Single model mode")

        while True:
            config_name = input(f"\n{self.ui.theme.input_prompt} Select configuration (default: basic): ").strip()
            if not config_name:
                config_name = "basic"

            if config_name in SWARM_CONFIGS:
                self.swarm_config = SWARM_CONFIGS[config_name]
                settings.swarm_config_name = config_name
                break
            else:
                self.ui.print_error(f"Invalid configuration: {config_name}")

        logger.info(f"Selected swarm configuration: {config_name}")

        try:
            factory = ProviderFactory()

            if self.swarm_config.has_merger:
                self.ui.print_success(f"\n✓ Initialized swarm mode with {len(self.swarm_config.generators)} models")
            else:
                gen = self.swarm_config.generators[0]
                api_key = settings.get_api_key_for_provider(gen.provider)
                factory.create(
                    gen.provider,
                    api_key=api_key,
                    model=gen.model,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens
                )
                self.ui.print_success(f"\n✓ Using {gen.model}")

            self.chat_session = ChatSession(swarm_config=self.swarm_config)

            self.ui.print_info("\nCommands:")
            self.ui.print_info("  - Type 'exit' or 'quit' to end the conversation")
            self.ui.print_info("  - Type '/clear' to clear the screen and conversation history")
            self.ui.print_info("  - Press Ctrl+C to interrupt streaming responses\n")

            return True

        except Exception as e:
            self.ui.print_error(f"Failed to initialize: {str(e)}")
            logger.exception("Setup failed")
            return False
            
    def run(self):
        self.is_running = True

        while self.is_running:
            try:
                user_input = input(self.ui.get_input_prompt()).strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit']:
                    self.ui.print_info("\nGoodbye! 👋")
                    self.is_running = False
                    break

                elif user_input.lower() == '/clear':
                    clear_screen()
                    self.ui.print_header()
                    self.chat_session.clear_history()
                    self.ui.print_info("Screen cleared and conversation history cleared.")
                    continue

                print(self.ui.format_user_message(user_input))

                if self.swarm_config.has_merger:
                    animation = SpinnerAnimation(
                        f"Querying {len(self.swarm_config.generators)} models in parallel"
                    )
                else:
                    animation = SpinnerAnimation("Thinking")

                animation.start()

                try:
                    print(self.ui.format_assistant_message(), end='', flush=True)

                    if settings.stream:
                        first_chunk = True

                        async def stream_and_print():
                            nonlocal first_chunk
                            async for chunk in self.chat_session.stream_message(user_input):
                                if first_chunk:
                                    animation.stop()
                                    first_chunk = False
                                print(self.ui.theme.assistant_text(chunk), end='', flush=True)

                        asyncio.run(stream_and_print())
                    else:
                        response = asyncio.run(self.chat_session.send_message(user_input))
                        animation.stop()
                        print(self.ui.theme.assistant_text(response))

                    print()

                except KeyboardInterrupt:
                    animation.stop()
                    self.ui.print_warning("\n\nResponse interrupted by user")
                    logger.info("User interrupted response")
                    continue

                except ValueError as ve:
                    animation.stop()
                    if "Task decomposition failed" in str(ve):
                        self.ui.print_error(f"\n\n{str(ve)}")
                        logger.error(f"Task decomposition error: {ve}")
                    else:
                        self.ui.print_error(f"\n\nError: {str(ve)}")
                        logger.exception("Validation error during message handling")
                except Exception as e:
                    animation.stop()
                    self.ui.print_error(f"\n\nError: {str(e)}")
                    logger.exception("Error during message handling")

            except KeyboardInterrupt:
                self.ui.print_warning("\n\nUse 'exit' or 'quit' to leave the chat")
                continue

            except EOFError:
                self.ui.print_info("\nGoodbye! 👋")
                self.is_running = False
                break

            except Exception as e:
                self.ui.print_error(f"Unexpected error: {str(e)}")
                logger.exception("Unexpected error in main loop")
                
    def start(self):
        try:
            if self.setup():
                self.run()
        finally:
            if self.chat_session:
                self.chat_session.cleanup()
            logger.info("Application shutdown")