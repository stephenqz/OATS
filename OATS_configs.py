### OATS Hyperparams ######################################################################
oats_configs = []
rank_ratio_list = [0.25, 0.3]
num_iters_list = [80]

for rank_idx in range(len(rank_ratio_list)):
    for ni_idx in range(len(num_iters_list)):
        oats_config = {
                        'rank_ratio': rank_ratio_list[rank_idx],
                        'num_iters': num_iters_list[ni_idx], 
                        'compress'   : True,
        }
        oats_configs.append(oats_config)

model_list = ["phi-3-mini", "phi-3-medium", "llama3-8b"] 
prune_list = ["OATS"]

OATS_exper = []
counter = 1
for m_idx in range(len(model_list)):
    for p_idx in range(len(prune_list)):
        if prune_list[p_idx] == "dense":
            sparsity_list = [1.0]
        else:
            sparsity_list = [0.3, 0.4, 0.5, 0.6]
        
        for s_idx in range(len(sparsity_list)):
            prune_hyper_list = []

            if sparsity_list[s_idx] == 0.5:
                sparsity_type_list = ["unstructured", "2:8"]
            else:
                sparsity_type_list = ["unstructured"]
            for oats_idx in range(len(oats_configs)):
                for st_idx in range(len(sparsity_type_list)):
                    oats_configs[oats_idx]["sparsity_type"] = sparsity_type_list[st_idx]
                    prune_hyper_list.append(oats_configs[oats_idx].copy())
            
            for ph_idx in range(len(prune_hyper_list)):
                exper_dict = {
                    "Experiment Number": counter,
                    'model': model_list[m_idx],
                    'prune_type': prune_list[p_idx],
                    'sparsity': sparsity_list[s_idx],
                    'dtype': "bf16",
                    "distribute_model": True,
                    "device": None,
                    "cal_dataset": "c4",
                    "cal_nsamples": 128,
                    "cal_batch_size": 32,
                    "cal_max_seqlen": 2048,
                    "varied_seqlen": False,
                    "seed": 42,
                    "eval_zero_shot": True,
                    "eval_mmlu": True,
                    "eval_ppl": True, 
                }
                exper_dict['prune_hyper'] = prune_hyper_list[ph_idx]
                OATS_exper.append(exper_dict)
                counter += 1