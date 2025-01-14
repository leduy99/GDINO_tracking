__version__ = '10.0.10'

from pathlib import Path

from boxmot.strongsort.strong_sort import StrongSORT
from boxmot.ocsort.ocsort import OCSort as OCSORT
from boxmot.bytetrack.byte_tracker import BYTETracker
from boxmot.botsort.bot_sort import BoTSORT
from boxmot.deepocsort.ocsort import OCSort as DeepOCSORT
from boxmot.macsortv2.ocsort import OCSort as MACSORTv2
from boxmot.deep.reid_multibackend import ReIDDetectMultiBackend

from boxmot.tracker_zoo import create_tracker, get_tracker_config


FILE = Path(__file__).resolve()
ROOT = FILE.parent  # root directory
EXAMPLES = ROOT / 'examples'
WEIGHTS = ROOT / 'weights'


__all__ = '__version__', 'StrongSORT', 'OCSORT', 'BYTETracker', 'BoTSORT',\
          'DeepOCSORT', 'MACSORTv2'  # allow simpler import
