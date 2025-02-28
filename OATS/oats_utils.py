import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from models.model_adapter import ModelAdapter
from OATS.compressed_linear import CompressedLinear, CompressedQKV
from OATS.pruning_utils import load_checkpoint
from OATS.pruning_utils import find_layers
import time

def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def load_oats(
    model_name: str,
    sparsity: float,
    prune_hyperparams: dict,
    model_path: str | None = None,
    *,
    dtype: torch.dtype = torch.float16,
    eval_model: bool = False,
) -> tuple[ModelAdapter, PreTrainedTokenizerBase]:
    """
    Load the model and the tokenizer from the given path.
    The corresponding model adapter class must be imported before calling this method.
    """
    assert model_path is not None

    print(
        f"Loading %s config %s from %s",
        model_name,
        "and model weights",
        model_path
    )

    # LLAMA VARIANTS
    if model_name == "llama3-8b":
        hf_path = "meta-llama/Meta-Llama-3-8B"
    elif model_name == "llama3-70b":
        hf_path = "meta-llama/Meta-Llama-3-70B"
    # PHI-3 VARIANTS
    elif model_name == "phi-3-mini":
        hf_path = "microsoft/Phi-3-mini-4k-instruct"
    elif model_name == "phi-3-medium":
        hf_path = "microsoft/Phi-3-medium-128k-instruct"
    
    model_adapter = ModelAdapter.from_model(
        model_name,
        model_path=hf_path,
        dtype=dtype,
        eval_model=eval_model,
    )

    model = model_adapter.model
    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=True)
    model_adapter.post_init(tokenizer)

    # Convert model to compressed model
    _, _, prune_start_idx, _ = load_checkpoint(model_path +  "/prune_chkpt.pt")

    rank_ratio = prune_hyperparams['rank_ratio']

    if 'sparsity_type' in prune_hyperparams and prune_hyperparams['sparsity_type'] != "unstructured":
        prune_n, prune_m = map(int, prune_hyperparams['sparsity_type'].split(":"))
    else:
        prune_n = 0
        prune_m = 0
    
    layers = model_adapter.get_layers()
    start_time = time.time()
    for layer_idx, layer_adapter in enumerate(layers):
        if layer_idx <= prune_start_idx:
            
            dense_alloc = 1.0 - sparsity
            
            print(dense_alloc)

            layers_to_replace = find_layers(layer_adapter.layer)
            for layer_name in layers_to_replace.keys():

                d_out = layers_to_replace[layer_name].weight.shape[0]
                d_in = layers_to_replace[layer_name].weight.shape[1]

                if len(layer_adapter.qkv_names) == 1 and layer_name == layer_adapter.qkv_names[0]:
                    q_out = layer_adapter.get_qkv_partition()[0]
                    k_out = layer_adapter.get_qkv_partition()[1] - layer_adapter.get_qkv_partition()[0]
                    v_out = d_out - layer_adapter.get_qkv_partition()[1]

                    if prune_n != 0:
                        unstruct_sparse = prune_n/prune_m
                        dense_alloc = unstruct_sparse/(1.0- rank_ratio)

                    q_rank = int(rank_ratio  * dense_alloc * (q_out*d_in)/(q_out + d_in))
                    k_rank = int(rank_ratio  * dense_alloc * (k_out*d_in)/(k_out + d_in))
                    v_rank = int(rank_ratio  * dense_alloc * (v_out*d_in)/(v_out + d_in))

                    parent, target, target_name = _get_submodules(layer_adapter.layer, layer_name)

                    if eval_model:
                        layer_map = target.weight.device
                    else:
                        layer_map = None

                    new_module = CompressedQKV(d_in, q_rank, q_out, k_rank, k_out, v_rank, v_out, bias=layers_to_replace[layer_name].bias is not None, device=layer_map, dtype=dtype) 
                    setattr(parent, target_name, new_module)

                else:
                    if prune_n != 0:
                        unstruct_sparse = prune_n/prune_m
                        dense_alloc = unstruct_sparse/(1.0- rank_ratio)
                    
                    target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                    parent, target, target_name = _get_submodules(layer_adapter.layer, layer_name)

                    if eval_model:
                        layer_map = target.weight.device
                    else:
                        layer_map = None
                    
                    new_module = CompressedLinear(d_in, target_rank, d_out, bias=layers_to_replace[layer_name].bias is not None, device=layer_map, dtype=dtype)
                    setattr(parent, target_name, new_module)
                
            layer_adapter.layer.load_state_dict(torch.load(model_path + "/oats_chkpt_" + str(layer_idx) + ".pt", map_location=layer_map))
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for loading compressed model: {elapsed_time} seconds", flush=True)
    print(model)
    
    return model_adapter, tokenizer