from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class GlobalConfig:
    sfreq: float
    n_channels: int
    n_times: int
    tmin: float
    tmax: float
    file_ext: str
    rs: int

    _times = []

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, times):
        self._times = times

@dataclass
class PathsConfig:
    clean: str
    raw: str
    model: str
    output: str
    log: str


@dataclass
class Config:
    global_params: GlobalConfig
    paths: PathsConfig


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return Config(
        global_params=GlobalConfig(**config["global"]),
        paths=PathsConfig(**config["paths"]),
    )

CONFIG_FILE = "configs/config.yml"

config = load_config(
    path=CONFIG_FILE
)

SFREQ = config.global_params.sfreq
N_CHANNELS = config.global_params.n_channels
N_TIMES = config.global_params.n_times
RANDOM_STATE = config.global_params.rs
FILE_EXT = config.global_params.file_ext

CLEAN_SUBJECTS_PATH = Path(config.paths.clean)
RAW_SUBJECTS_PATH = Path(config.paths.raw)
OUTPUT_PATH = Path(config.paths.output)
LOG_PATH = Path(config.paths.log)
MODEL_PATH = Path(config.paths.model)
