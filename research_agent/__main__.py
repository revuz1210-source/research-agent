"""Package entry-point — enables `python -m research_agent`."""

from research_agent.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
