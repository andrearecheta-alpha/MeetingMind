"""
__main__.py
-----------
Enables the package to be invoked directly with:

    python -m meetingmind <command> [options]

All logic lives in cli.py; this file is a thin entry-point shim.
"""

from meetingmind.cli import main

if __name__ == "__main__":
    main()
