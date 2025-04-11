import torch 
from tqdm import tqdm
from config import config
from OATS.prune_oats_compress import replace_linear
from OATS.pruning_utils import find_layers
from OATS.oats_wrapper import WrappedOATS
from OATS.pruning_utils import load_checkpoint, save_checkpoint

import os

@torch.no_grad()
def prune_oats_vit(model_adapter, sparsity, calib_loader, prune_hyperparams, checkpoint_path):

    model_adapter.model.eval()

    rank_ratio = prune_hyperparams['rank_ratio']
    num_iters = prune_hyperparams['num_iters']

    dense_alloc = 1-sparsity

    inps = []
    prune_start_idx = -1

    if os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        _, pruned_inps, prune_start_idx, _ = load_checkpoint(checkpoint_path +  "/prune_chkpt.pt")
        print("Resuming pruning from: " + str(prune_start_idx))
    else:
        for batch in calib_loader:
            if isinstance(batch, dict):
                inp_batch = batch["pixel_values"].to(config.device)
            else:
                inp_batch = batch[0].to(config.device)
            layer_inp_batch = get_layer0_inputs(model_adapter, inp_batch)
            inps.append(layer_inp_batch)
    
        pruned_inps = inps
    
    layers = model_adapter.get_layers()
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Pruning using OATS")):
        if layer_idx > prune_start_idx:
            # ========== Setup hooks and wrap layers ==============================
            subset = find_layers(layer_adapter.layer)
            wrapped_layers = {}

            for name in subset:
                wrapped_layers[name] = WrappedOATS(subset[name])
                wrapped_layers[name].scaler_row = wrapped_layers[name].scaler_row.cpu()
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            # =========== Precompute quantities =================================
            for batch_idx, layer_inps_batch in enumerate(pruned_inps):
                layer_inps_batch = layer_inps_batch.to(config.device)
                out = layer_adapter.layer(layer_inps_batch)
            for h in handles:
                h.remove()
            
            # Perform remaining weights in model
            for name in subset:
                print(f"pruning layer {layer_idx} name {name}")
                diag_approx = wrapped_layers[name].scaler_row.clone().reshape((1,-1)).to(config.device)
                orig_weight = subset[name].weight.data.clone().detach().float().to(config.device)
                d_out = orig_weight.shape[0]
                d_in = orig_weight.shape[1]

                target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc
                
                lrc_V, lrc_U, sparse_comp = altern_ls(orig_weight, diag_approx, num_iters, target_rank, unstruct_sparse)
                
                # Compress Layer
                replace_linear(name, subset, layer_adapter, d_in, target_rank, d_out, lrc_V, lrc_U, sparse_comp)

            # ============== Recalculate outputs with pruned weight ====================
            pruned_outs = []

            for batch_idx, layer_inps_batch in enumerate(pruned_inps):
                layer_inps_batch = layer_inps_batch.to(config.device)
                out = layer_adapter.layer(layer_inps_batch)[0]
                out = out.cpu()
                pruned_outs.append(out)
            
            for batch_idx, pruned_out in enumerate(pruned_outs):
                pruned_inps[batch_idx] = pruned_out
            
            torch.save(layer_adapter.layer.state_dict(), checkpoint_path + "/oats_chkpt_" + str(layer_idx) + ".pt")
            save_checkpoint(None, pruned_inps, layer_idx, None, checkpoint_path + "/prune_chkpt.pt")

def altern_ls(weight, diag_approx, num_iters, target_rank, unstruct_sparse):
    
    scaled_weight = weight * torch.sqrt(diag_approx) # d_out x d_in
    sparse_component = torch.zeros_like(scaled_weight).to(config.device)

    for iter_idx in range(num_iters): 
        U, S, V = torch.linalg.svd(scaled_weight - sparse_component , full_matrices=False)
        S[target_rank:] = 0
        low_rank_component = U @ torch.diag(S) @ V
        sparse_component = scaled_weight - low_rank_component

        # Prune the weight
        W_metric = sparse_component.clone()
        W_mask = (torch.zeros_like(W_metric) == 1)

        sort_res = torch.sort(torch.abs(W_metric), dim=-1, stable=True)
        # unstructured pruning
        indices = sort_res[1][:,:int(W_metric.shape[1]* unstruct_sparse)]
        W_mask.scatter_(1, indices, True)
        sparse_component[W_mask] = 0
    
    low_rank_compressed_V = (V[:target_rank, :]).detach().clone() * (1/torch.sqrt(diag_approx))
    low_rank_compressed_U = (U[:, :target_rank] @ torch.diag(S)[:target_rank, : target_rank]).detach().clone()
    sparse_comp = sparse_component * (1/torch.sqrt(diag_approx))

    return low_rank_compressed_V, low_rank_compressed_U, sparse_comp

def get_layer0_inputs(model_adapter, batch):
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
        model_adapter.model(batch)
    except ValueError:
        pass

    args = layer0_catcher.saved_args
    model_adapter.set_raw_layer_at(0, layer0_adapter.layer)

    return args[layer0_adapter.hidden_states_args_position].cpu()