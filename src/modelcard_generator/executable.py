"""Entry point shim — use `mcg` CLI or `python -m modelcard_generator`."""

from .cli.main import main

if __name__ == "__main__":
    main()
