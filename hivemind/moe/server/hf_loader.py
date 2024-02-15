from typing import Dict
from safetensors.torch import load_file
import json
from pathlib import Path
from typing import Dict

#загрузить чекпоинт с huggingface
#посмотреть в model.safetensors.index.json
#для каждого эксперта:
#на основе правил с помощью jsonа внутри model.safetensors.index.json загрузить нужный state_dict
#преобразовать ключи внутри state_dictа в нужные ключи для state_dictа эксперта
#загрузить в эксперта полученный state_dict

#rule for mixtral

def mixtral_rules(layer_id : int,expert_id : int):
    return {
        f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w1.weight":"w1.weight",
    f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w2.weight":"w2.weight",
f"model.layers.{layer_id}.block_sparse_moe.experts.{expert_id}.w3.weight":"w3.weight"
}


def load_safetensor(path):
    return load_file(path)

def load_weights_map_from_json(path):
    return json.load(open(path))["weight_map"]

def get_state_dict_by_key(key, base_path : Path, weigths_map : Dict):
    return load_safetensor(base_path / weigths_map[key])
