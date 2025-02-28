import torch.nn as nn 
import os
import torch
from config import config
from gpu_utils import map_tensors

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def save_checkpoint(orig_args, pruned_args, prune_start_idx, kwargs, checkpoint_path):

    temp_path = os.path.join(os.path.dirname(checkpoint_path), "temp.pt")
    training_state = {
        'orig_args': orig_args,
        'pruned_args': pruned_args,
        'prune_start_idx': prune_start_idx,
        'kwargs': kwargs,
    }
    torch.save(training_state, temp_path)
    os.replace(temp_path, checkpoint_path)

def load_checkpoint(checkpoint_path, map_location=None):

    prune_state = torch.load(checkpoint_path, map_location=map_location)
    orig_args = prune_state['orig_args']
    pruned_args = prune_state['pruned_args']
    prune_start_idx = prune_state['prune_start_idx']
    kwargs = prune_state['kwargs']
    
    return orig_args, pruned_args, prune_start_idx, kwargs

def get_layer0_inputs(model_adapter, batch):
    # Move embeddings to device.
    for embed_module in model_adapter.get_embeddings():
        embed_module.to(config.device)

    class Catcher(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args, **kwargs):
            self.saved_args = args
            self.saved_kwargs = kwargs
            raise ValueError

    layer0_adapter = model_adapter.get_layers()[0]
    layer0_catcher = Catcher()
    model_adapter.set_raw_layer_at(0, layer0_catcher)

    try:
        batch = map_tensors(batch, device=config.device)
        model_adapter.model(**batch)
    except ValueError:
        pass

    # grab the inputs and caught arguments
    args = layer0_catcher.saved_args
    kwargs = layer0_catcher.saved_kwargs

    # put the caught stuff on cpu
    args = map_tensors(args, device='cpu')
    kwargs = map_tensors(kwargs, device='cpu')

    # put the layer back to normal
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    # Move embeddings back to cpu, and clear GPU cache.
    for embed_module in model_adapter.get_embeddings():
        embed_module.to('cpu')

    return args[layer0_adapter.hidden_states_args_position], args, kwargs


