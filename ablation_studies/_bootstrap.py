"""Read-only access to radar data and network modules."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXTERNAL_ROOT = HERE / 'external' if (HERE / 'external').is_dir() else HERE.parent
sys.path.insert(1, str(EXTERNAL_ROOT))
