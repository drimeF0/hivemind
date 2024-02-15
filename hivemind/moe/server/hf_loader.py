from typing import Dict

from safetensors.torch import load_file, save_file
from huggingface_hub import snapshot_download,hf_hub_download

import json
from pathlib import Path
from typing import Dict, List
import functools

from hivemind.moe.server.module_backend import ModuleBackend


#загрузить чекпоинт с huggingface !
#посмотреть в model.safetensors.index.json !
#для каждого эксперта:
#на основе правил с помощью jsonа внутри model.safetensors.index.json загрузить нужный state_dict
#преобразовать ключи внутри state_dictа в нужные ключи для state_dictа эксперта
#загрузить в эксперта полученный state_dict

#rule for mixtral

def mixtral_rules(layer_id : int,expert_id : int):
    return {
        f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight":"block.w1.weight",
    f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight":"block.w2.weight",
f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight":"block.w3.weight"
}


def load_huggingface_rep(repo_id : str) -> Path:
    return Path(snapshot_download(repo_id=repo_id,allow_patterns=["*.json"]))

def load_safetensor(path : str):
    return load_file(path)

def save_safetensor(tensors: Dict, path: str):
    save_file(tensors, path)
    

def load_weights_map_from_json(path: Path) -> Dict:
    return json.load(open(path))["weight_map"]

def get_state_dict_by_key(key : str, weigths_map : Dict, repo_id : str):
    filename = weigths_map[key]

    normally_path = Path(
        hf_hub_download(repo_id=repo_id,filename=filename)
    )
    return load_safetensor(normally_path)

def get_expert_state_dict(state_dict : Dict, key : str):
    return state_dict[key]

def _load_weights_from_hf(expert : ModuleBackend, repo_id: str):
    repo_path = load_huggingface_rep(repo_id=repo_id)
    weights_map = load_weights_map_from_json(repo_path / "model.safetensors.index.json")
    rule = mixtral_rules(expert.layer_id, expert.expert_id)
    module_state_dict = {}

    for key,expert_key in rule.items():
        state_dict = get_state_dict_by_key(key,weights_map,repo_id)
        value = get_expert_state_dict(state_dict,key)
        module_state_dict[expert_key] = value
    
    return {"module":module_state_dict}

def load_weights_from_hf(experts: List[ModuleBackend], repo_id: str):
    for _,expert in experts.items():
        state_dict = _load_weights_from_hf(expert, repo_id)
        expert.load_state_dict(state_dict)