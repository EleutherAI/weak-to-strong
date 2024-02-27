import json
import os
from typing import Optional

import wandb


def append_to_jsonl(path: str, data: dict):
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


class WandbLogger(object):
    CURRENT: Optional["WandbLogger"] = None

    log_path = None

    def __init__(
        self,
        save_path: str,
        wandb_args: dict,
        **kwargs,
    ):
        wandb.init(**wandb_args)

        self.log_path = os.path.join(save_path, "log.jsonl")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._log_dict = {}

    def logkv(self, key, value):
        self._log_dict[key] = value

    def logkvs(self, d):
        self._log_dict.update(d)

    def dumpkvs(self):
        wandb.log(self._log_dict)
        if self.log_path is not None:
            append_to_jsonl(self.log_path, self._log_dict)
        self._log_dict = {}

    def shutdown(self):
        wandb.finish()


def is_configured():
    return WandbLogger.CURRENT is not None


def get_current():
    assert is_configured(), "WandbLogger is not configured"
    return WandbLogger.CURRENT


def configure(**kwargs):
    if is_configured():
        WandbLogger.CURRENT.shutdown()  # type: ignore
    WandbLogger.CURRENT = WandbLogger(**kwargs)
    return WandbLogger.CURRENT


def logkv(key, value):
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.logkv(key, value)  # type: ignore


def logkvs(d):
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.logkvs(d)  # type: ignore


def dumpkvs():
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.dumpkvs()  # type: ignore


def shutdown():
    assert is_configured(), "WandbLogger is not configured"
    WandbLogger.CURRENT.shutdown()  # type: ignore
    WandbLogger.CURRENT = None
