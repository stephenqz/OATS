oats_config = {
                'rank_ratio': 0.2,
                'num_iters': 80, 
}

model_list = ["vit-base-patch16-224", "dinov2-giant-imagenet1k"]

vit_exper = []
counter = 1
for m_idx in range(len(model_list)):
    sparsity_list = [0.3, 0.4, 0.5]
    
    for s_idx in range(len(sparsity_list)):
        exper_dict = {
            'model': model_list[m_idx],
            'sparsity': sparsity_list[s_idx],
            'dtype': "fp32",
            "device": None,
            "cal_dataset": 'imnet_cal',
            "cal_nsamples": 2048,
            "cal_batch_size": 256,
            "seed": 42,
            "eval_dataset": 'imnet_val',
        }
        exper_dict['prune_hyper'] = oats_config
        vit_exper.append(exper_dict)
        counter += 1