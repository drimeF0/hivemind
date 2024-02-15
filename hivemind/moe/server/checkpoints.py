import os
import threading
from datetime import datetime
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Dict

import torch

from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils.logging import get_logger
from hivemind.moe.server.hf_loader import load_weights_from_hf, load_safetensor, save_safetensor

logger = get_logger(__name__)


def is_directory(directory: Path):
    assert directory is not None
    assert directory.exists()
    assert directory.is_dir()
    return True

def is_file(directory: Path):
    assert directory is not None
    assert directory.exists()
    assert directory.is_file()
    return True


def copy_tree(src: str, dst: str):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        src_entry = os.path.join(src, item)
        dst_entry = os.path.join(dst, item)
        if os.path.isdir(src_entry):
            copy_tree(src_entry, dst_entry)
        else:
            copy2(src_entry, dst_entry)


class CheckpointSaver(threading.Thread):
    def __init__(self, module_backends: Dict[str, ModuleBackend], checkpoint_dir: Path, update_period: float):
        super().__init__()
        assert is_directory(checkpoint_dir)
        self.module_backends = module_backends
        self.update_period = update_period
        self.checkpoint_dir = checkpoint_dir
        self.stop = threading.Event()

        # create expert directories to ensure that the directory is writable and checkpoints can be loaded
        store_experts(self.module_backends, self.checkpoint_dir)

    def run(self) -> None:
        while not self.stop.wait(self.update_period):
            store_experts(self.module_backends, self.checkpoint_dir)


def store_experts(experts: Dict[str, ModuleBackend], checkpoint_dir: Path):
    logger.debug(f"Storing experts at {checkpoint_dir.absolute()}")
    assert is_directory(checkpoint_dir)
    timestamp = datetime.now().isoformat(sep="_")
    with TemporaryDirectory() as tmpdirname:
        for expert_name, expert_backend in experts.items():
            expert_dir = Path(tmpdirname) / expert_name
            expert_dir.mkdir()
            checkpoint_name = expert_dir / f"checkpoint_{timestamp}.safetensors"
            save_safetensor(expert_backend.state_dict(), checkpoint_name)
            os.symlink(checkpoint_name, expert_dir / "checkpoint_last.safetensors")
        copy_tree(tmpdirname, str(checkpoint_dir))


def load_experts(experts: Dict[str, ModuleBackend], checkpoint_dir: Path, load_in_4bit: bool):
    assert is_directory(checkpoint_dir)
    for expert_name, expert in experts.items():
        checkpoints_folder = checkpoint_dir / expert_name
        latest_checkpoint = checkpoints_folder / "checkpoint_last.safetensors"
        if latest_checkpoint.exists():
            expert.load_state_dict(load_safetensor(latest_checkpoint))
        else:
            logger.warning(f"Failed to load checkpoint for expert {expert_name}")




def load_experts_from_hf(experts: Dict[str, ModuleBackend], repo_id: str):
    for _,expert in experts.items():
        state_dict = load_weights_from_hf(expert,repo_id)
        expert.load_state_dict(state_dict)