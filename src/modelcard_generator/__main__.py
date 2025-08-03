"""Main entry point for the model card generator package."""

import sys
from .cli.main import main

if __name__ == "__main__":
    sys.exit(main())