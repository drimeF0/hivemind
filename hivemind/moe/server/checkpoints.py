import os
import threading
from datetime import datetime
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Dict

import torch
from torch import nn

from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils.logging import get_logger
from hivemind.moe.server.hf_loader import load_safetensor, save_safetensor, _load_weights_from_hf

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
            module_state_dict, backend_state_dict = expert_backend.state_dict()
            module_checkpoint_name = expert_dir / f"module_{timestamp}.safetensors"
            backed_checkpoint_name = expert_dir / f"checkpoint_{timestamp}.safetensors"
            save_safetensor(module_state_dict, module_checkpoint_name)
            save_safetensor(backend_state_dict, backed_checkpoint_name)
            os.symlink(module_checkpoint_name, expert_dir / "module.safetensors")
            os.symlink(backed_checkpoint_name, expert_dir / "checkpoint_last.safetensors")
        copy_tree(tmpdirname, str(checkpoint_dir))


def _load_expert(expert: nn,Module, expert_name: str, checkpoint_dir: Path):
    checkpoints_folder = checkpoint_dir / expert_name
    latest_checkpoint = checkpoints_folder / "module.safetensors"
    if latest_checkpoint.exists():
        expert.load_state_dict(load_safetensor(latest_checkpoint))
    else:
        logger.warning(f"Failed to load checkpoint for expert {expert_name}")




def load_expert_from_hf(expert: nn.Module, repo_id: str, expert_id: int, layer_id: int):
    _load_weights_from_hf(expert,repo_id,expert_id,layer_id)