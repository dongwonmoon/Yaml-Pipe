"""
Configuration file for pytest.

This file adds the project's root directory to the Python path so that
pytest can find the 'vectorflow' module without needing to install it.
"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
