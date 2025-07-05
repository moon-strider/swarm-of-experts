#!/usr/bin/env python3

from src.cli.app import CLIApp
from src.utils.logging import setup_logging


def main():
    setup_logging()
    
    app = CLIApp()
    app.start()


if __name__ == "__main__":
    main()