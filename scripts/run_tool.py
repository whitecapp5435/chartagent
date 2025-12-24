import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chartagent_tools.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
