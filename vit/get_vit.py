import torch
from OATS.compressed_linear import CompressedLinear
from vit.vit_adapters.vit_adapter import ViTModelAdapter
from vit.vit_adapters.dino_adapter import DINOModelAdapter
from OATS.pruning_utils import load_checkpoint
from OATS.pruning_utils import find_layers
from OATS.oats_utils import _get_submodules
from transformers import ViTImageProcessor, AutoImageProcessor

def get_vit(
    model_name: str,
    dtype: torch.dtype = torch.float16
):
    if model_name == "vit-base-patch16-224":
        hf_path = "google/vit-base-patch16-224"
    elif model_name == "dinov2-giant-imagenet1k":
        hf_path = "facebook/dinov2-giant-imagenet1k-1-layer"
    
    if "vit" in model_name:
        model_adapter = ViTModelAdapter(
            model_path=hf_path,
            dtype=dtype,
        )

        image_processor = ViTImageProcessor.from_pretrained(hf_path)

    elif "dinov2" in model_name:
        model_adapter = DINOModelAdapter(
            model_path=hf_path,
            dtype=dtype
        )
        image_processor = AutoImageProcessor.from_pretrained(hf_path)
    
    model = model_adapter.model
    model.eval() 

    return model_adapter, image_processor

def load_oats_vit(
    model_name: str,
    sparsity: float,
    rank_ratio: float,
    model_path: str | None = None,
    dtype: torch.dtype = torch.float16):
    assert model_path is not None

    model_adapter, image_processor = get_vit(model_name, dtype=dtype)

    model = model_adapter.model
    model.eval() 

    dense_alloc = 1.0 - sparsity
    layers = model_adapter.get_layers()
    _, _, prune_start_idx, _ = load_checkpoint(model_path +  "/prune_chkpt.pt")

    for layer_idx, layer_adapter in enumerate(layers):
        if layer_idx <= prune_start_idx:
            layers_to_replace = find_layers(layer_adapter.layer)
            for layer_name in layers_to_replace.keys():

                d_out = layers_to_replace[layer_name].weight.shape[0]
                d_in = layers_to_replace[layer_name].weight.shape[1]

                target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                parent, _, target_name = _get_submodules(layer_adapter.layer, layer_name)
                new_module = CompressedLinear(d_in, target_rank, d_out, bias=layers_to_replace[layer_name].bias is not None, dtype=dtype)
                setattr(parent, target_name, new_module)
    
            layer_adapter.layer.load_state_dict(torch.load(model_path + "/oats_chkpt_" + str(layer_idx) + ".pt"))
    
    return model_adapter, image_processor