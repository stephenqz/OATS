from OATS.pruning_utils import find_layers
import torch 
from config import config
from gpu_utils import map_tensors
from OATS.oats_wrapper import WrappedOATS
from OATS.pruning_utils import get_layer0_inputs
from tqdm import tqdm
from OATS.pruning_utils import load_checkpoint, save_checkpoint
import os
import pathlib
import shutil

@torch.no_grad()
def prune_oats(model_adapter, tokenizer, sparsity, calib_loader, prune_hyperparams, checkpoint_path, prune_n=0, prune_m=0):

    model_adapter.model.eval()
    use_cache = model_adapter.model.config.use_cache
    
    model_adapter.model.config.use_cache = False 
    
    rank_ratio = prune_hyperparams['rank_ratio']
    num_iters = prune_hyperparams['num_iters']

    inps, args, kwargs = [],  [], []
    prune_start_idx = -1

    if os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        _, pruned_args, prune_start_idx, kwargs = load_checkpoint(checkpoint_path +  "/prune_chkpt.pt")
        print("Resuming pruning from: " + str(prune_start_idx))
    else:
        for batch in calib_loader:
            inp_batch, args_batch, kwargs_batch = get_layer0_inputs(model_adapter, batch)
            args.append(args_batch)
            kwargs.append(kwargs_batch)
            inps.append(inp_batch)
        pruned_args = args
    
    layers = model_adapter.get_layers()
    for layer_idx, layer_adapter in enumerate(tqdm(layers, unit="layer", desc="Pruning using OATS")):
        if layer_idx > prune_start_idx:
            dense_alloc = 1 - sparsity
            # ========== Setup hooks and wrap layers ==============================
            subset = find_layers(layer_adapter.layer)
            wrapped_layers = {}

            for name in subset:
                wrapped_layers[name] = WrappedOATS(subset[name])
            
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            
            # =========== Precompute quantities =================================
            layer_adapter.layer.to(config.device)
            for batch_idx, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(pruned_args, kwargs)):
                layer_args_batch, layer_kwargs_batch = map_tensors(
                    [layer_args_batch, layer_kwargs_batch], device=config.device
                )
                out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
            for h in handles:
                h.remove()
            # ============== Prune the weights in the layer ==============================
            # Perform QKV separately
            if len(layer_adapter.qkv_names) == 1:
                qkv_name = layer_adapter.qkv_names[0]

                diag_approx = wrapped_layers[qkv_name].scaler_row.clone().reshape((1,-1)).to(config.device)

                qkv_weight = subset[qkv_name].weight.data.clone().detach().float()

                q_proj = qkv_weight[                                    : layer_adapter.get_qkv_partition()[0], : ].clone().detach().float()
                k_proj = qkv_weight[layer_adapter.get_qkv_partition()[0]: layer_adapter.get_qkv_partition()[1], : ].clone().detach().float()
                v_proj = qkv_weight[layer_adapter.get_qkv_partition()[1]: ,                                     : ].clone().detach().float()

                compressed_qkv = []
                for qkv_idx, qkv_mat in enumerate([q_proj, k_proj, v_proj]):
                    d_out = qkv_mat.shape[0]
                    d_in = qkv_mat.shape[1]

                    if prune_n != 0:
                        unstruct_sparse = prune_n/prune_m
                        dense_alloc = unstruct_sparse/(1.0- rank_ratio)
                        target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                        print("Unstructured sparsity for Sparse Term: " + str(unstruct_sparse))
                        print("Target Rank for Low-Rank Term: " + str(target_rank))
                        print("Dense Allocation for Layer when Doing N:M Sparsity: " + str(dense_alloc))
                    else:
                        target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                        unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc

                    compressed_qkv.append(altern_ls(qkv_mat, diag_approx, \
                                            num_iters, target_rank, unstruct_sparse, \
                                            prune_n=prune_n, prune_m=prune_m))

                cat_qkv = torch.cat(compressed_qkv, dim = 0)
                subset[qkv_name].weight.data = (cat_qkv).clone().to(config.dtype).to(config.device)
                del subset[qkv_name]
            
            # Perform remaining weights in model
            for name in subset:
                print(f"pruning layer {layer_idx} name {name}")
                diag_approx = wrapped_layers[name].scaler_row.clone().reshape((1,-1)).to(config.device)
                
                orig_weight = subset[name].weight.data.clone().detach().float().to(config.device)
                d_out = orig_weight.shape[0]
                d_in = orig_weight.shape[1]

                if prune_n != 0 :
                    unstruct_sparse = prune_n/prune_m
                    dense_alloc = unstruct_sparse/(1.0- rank_ratio)
                    target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                    print("Unstructured sparsity for Sparse Term: " + str(unstruct_sparse))
                    print("Target Rank for Low-Rank Term: " + str(target_rank))
                    print("Dense Allocation for Layer when Doing N:M Sparsity: " + str(dense_alloc))
                else:
                    target_rank = int(rank_ratio  * dense_alloc * (d_out*d_in)/(d_out + d_in))
                    unstruct_sparse = 1.0 - (1.0-rank_ratio)*dense_alloc
                
                final_weight = altern_ls(orig_weight, diag_approx,\
                                        num_iters, target_rank, unstruct_sparse,\
                                        prune_n=prune_n, prune_m=prune_m)
                
                subset[name].weight.data = (final_weight).clone().to(config.dtype)
            # ============== Recalculate outputs with pruned weight ====================
            pruned_outs = []

            layer_adapter.layer.to(config.device)
            for batch_idx, (layer_args_batch, layer_kwargs_batch) in enumerate(zip(pruned_args, kwargs)):
                layer_args_batch, layer_kwargs_batch = map_tensors(
                    [layer_args_batch, layer_kwargs_batch], device=config.device
                )
                out = layer_adapter.layer(*layer_args_batch, **layer_kwargs_batch)
                if isinstance(out, tuple):
                    out = out[layer_adapter.hidden_states_output_position]
                out = out.cpu()
                pruned_outs.append(out)
            
            for batch_idx, pruned_out in enumerate(pruned_outs):
                pruned_args[batch_idx] = layer_adapter.get_updated_args(
                    pruned_out.cpu(),
                    pruned_args[batch_idx],
                )
            
            layer_adapter.layer.to('cpu')


            model_adapter.model.config.use_cache = use_cache
            

            tmp_chkpt_path = checkpoint_path + "_tmp"
            pathlib.Path(tmp_chkpt_path).mkdir(parents=True, exist_ok=True)
            model_adapter.model.save_pretrained(tmp_chkpt_path)
            tokenizer.save_pretrained(tmp_chkpt_path)
            save_checkpoint(None, pruned_args, layer_idx, kwargs, tmp_chkpt_path + "/prune_chkpt.pt")

            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            os.replace(tmp_chkpt_path, checkpoint_path)
            
            model_adapter.model.config.use_cache = False

    model_adapter.model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def altern_ls(weight, diag_approx, num_iters, target_rank, unstruct_sparse, prune_n=0, prune_m=0):
    
    if diag_approx.isnan().any():
        print("Outliers have NaN. Exiting!")
        raise ValueError
    
    scaled_weight = weight * torch.sqrt(diag_approx) # d_out x d_in
    sparse_component = torch.zeros_like(scaled_weight).to(config.device)
    for iter_idx in range(num_iters): 
        # Apply PCA
        U, S, V = torch.linalg.svd(scaled_weight - sparse_component , full_matrices=False)
        S[target_rank:] = 0

        low_rank_component = U @ torch.diag(S) @ V
        sparse_component = scaled_weight - low_rank_component

        # Prune the weight
        W_metric = sparse_component.clone()
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        
        if prune_n != 0:
            print("Applying N:M Sparsity")
            W_metric = torch.abs(W_metric)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_m - prune_n,dim=1, largest=False)[1], True)
        else:
            sort_res = torch.sort(torch.abs(W_metric), dim=-1, stable=True)
            indices = sort_res[1][:,:int(W_metric.shape[1]* unstruct_sparse)]
            W_mask.scatter_(1, indices, True)

        sparse_component[W_mask] = 0
    
    final_weight = (low_rank_component + sparse_component) * (1/torch.sqrt(diag_approx))

    return final_weight