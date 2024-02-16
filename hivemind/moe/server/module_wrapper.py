from torch.nn import Module
from typing import Union, Dict
from torch.optim import Adam, SGD

class ModuleWrapper:
    """
    
    """

    def __init__(
            self,
        module: Module,
        expert_id: int,
        layer_id: int,
        optim_class: Union[Adam,SGD],
        load_in_4bit: bool = False,
        optim_args: list =[],
        optim_kwargs: Dict = {}
    ):
        self.module = module
        self.expert_id = expert_id
        self.layer_id = layer_id
        self.optimizer = optim_class(self.module.parameters(),*optim_args,**optim_kwargs)
    

    def get_state_dict(self):
        return self.module.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        self.module.load_state_dict(state_dict)
