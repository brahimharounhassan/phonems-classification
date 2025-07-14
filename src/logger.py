import setup
import logging
from configs.config import LOG_PATH
import sys
import warnings
warnings.filterwarnings("ignore")


script_name = sys.argv[0]
fname = (
    script_name.split("/")[-1].split(".py")[0]
    if "/" in script_name
    else script_name.split("\\")[-1].split(".py")[0]
)

logger = logging.getLogger()
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

log_path = LOG_PATH / f"{fname}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

file_ = logging.FileHandler(log_path, mode="a")
file_.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
file_.setFormatter(formatter)

logger.addHandler(console)
logger.addHandler(file_)
logger.info("====" * 20)
