from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.nn import Linear

from hivemind import BatchTensorDescriptor, ModuleBackend
from hivemind.moe.server.checkpoints import load_expert, store_experts
from hivemind.moe.server.layers.lr_schedule import get_linear_schedule_with_warmup

EXPERT_WEIGHT_UPDATES = 3
BACKWARD_PASSES_BEFORE_SAVE = 2
BACKWARD_PASSES_AFTER_SAVE = 2
EXPERT_NAME = "test_expert"
PEAK_LR = 1.0


@pytest.fixture
def example_experts():
    expert = Linear(1, 1)
    opt = torch.optim.SGD(expert.parameters(), PEAK_LR)

    args_schema = (BatchTensorDescriptor(1),)
    expert_backend = ModuleBackend(
        name=EXPERT_NAME,
        module=expert,
        optimizer=opt,
        scheduler=get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=BACKWARD_PASSES_BEFORE_SAVE,
            num_training_steps=BACKWARD_PASSES_BEFORE_SAVE + BACKWARD_PASSES_AFTER_SAVE,
        ),
        args_schema=args_schema,
        outputs_schema=BatchTensorDescriptor(1),
        max_batch_size=1,
    )
    experts = {EXPERT_NAME: expert_backend}
    yield experts


@pytest.mark.forked
def test_save_checkpoints(example_experts):
    expert = example_experts[EXPERT_NAME].module

    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        for i in range(1, EXPERT_WEIGHT_UPDATES + 1):
            expert.weight.data[0] = i
            store_experts(example_experts, tmp_path)

        checkpoints_dir = tmp_path / EXPERT_NAME

        assert checkpoints_dir.exists()
        # include checkpoint_last.pt
        assert len(list(checkpoints_dir.iterdir())) == EXPERT_WEIGHT_UPDATES + 1

        expert.weight.data[0] = 0

