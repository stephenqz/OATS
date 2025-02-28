import logging
import os
import pathlib

import sys
import torch

from OATS_configs import OATS_exper
import data_utils
import gpu_utils
import hf_utils
from OATS.prune_oats import prune_oats
from OATS.prune_oats_compress import prune_oats_compress
from OATS.oats_utils import load_oats
from config import config
from ml_collections import ConfigDict

import lm_eval
from lm_eval.models.huggingface import HFLM
from data_utils import calculate_avg_accuracy

def process_pruning_args(args):
    for arg, argv in vars(args).items():
        logging.debug(f'{arg} = {argv}')

    if not 0 <= args.sparsity <= 1:
        raise ValueError

    if args.device:
        config.device = torch.device(args.device)

    if args.dtype == "fp16":
        config.dtype = torch.float16
    elif args.dtype == "bf16":
        config.dtype = torch.bfloat16
    elif args.dtype == "fp32":
        config.dtype = torch.float32
    else:
        raise ValueError


def pruning_main(args, checkpoint_path) -> None:
    
    train_dataset = data_utils.get_dataset(args.cal_dataset, train=True)
    print("Finished Train Set", flush=True)

    # Checkpointing for pruning
    if os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        if args.prune_type == "OATS" and args.prune_hyper['compress']:
            model_adapter, tokenizer = load_oats(
                    args.model,
                    args.sparsity,
                    args.prune_hyper,
                    checkpoint_path,
                    dtype=config.dtype
                )
        else:
            model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
                args.model,
                checkpoint_path,
                dtype=config.dtype
            )
    else:
         # load one of the pre-trained models
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(
            args.model, None, dtype=config.dtype
        )

    model = model_adapter.model

    def reset_model_device() -> None:
        if args.distribute_model:
            gpu_utils.distribute_model(model_adapter)
        else:
            model.to(config.device)

    print("Start loading data", flush=True)

    train_loader = data_utils.prepare_dataloader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_seqlen=args.cal_max_seqlen,
        batch_size=args.cal_batch_size,
        nsamples=args.cal_nsamples,
        varied_seqlen=args.varied_seqlen,
        seed=args.seed,
    )

    print("Finished Calibration Loader", flush=True)
    
    original_param_count = sum(p.numel() for p in model.parameters())
    print(f'Original model parameters: {original_param_count:,d}')

    # ========================= Pruning Code ========================================
    prune_n, prune_m = 0, 0
    if args.prune_hyper['sparsity_type'] != "unstructured":
        assert args.sparsity == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.prune_hyper['sparsity_type'].split(":"))
    
    if args.prune_hyper['compress']:
        prune_oats_compress(model_adapter, args.sparsity, train_loader, args.prune_hyper, checkpoint_path, prune_n=prune_n, prune_m=prune_m)
    else:
        prune_oats(model_adapter, tokenizer, args.sparsity, train_loader, args.prune_hyper, checkpoint_path, prune_n=prune_n, prune_m=prune_m)
    # =================================================

    # Run PPL Eval
    reset_model_device()
    eval_batch_size = "auto"
    
    if args.eval_ppl:
        hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=eval_batch_size) 
        with torch.no_grad():
            ppl_tasks = ["wikitext"]
            ppl_results = lm_eval.simple_evaluate(hflm, tasks=ppl_tasks, num_fewshot=None, batch_size=eval_batch_size)[
                    'results'
                ]
            
            ppl_vals = {task: round(result.get('word_perplexity,none', result['word_perplexity,none']), 4) for task, result in ppl_results.items()}

            for k, v in ppl_vals.items():
                print("Task Name: " + k + " Task Score: " + v)

    # ============== Run Zeroshot Eval ================
    if args.eval_zero_shot:
        hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=eval_batch_size) 
        with torch.no_grad():
            zero_shot_tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "rte", "openbookqa", "boolq"]

            ### LM Eval Harness ###
            zs_results = lm_eval.simple_evaluate(hflm, tasks=zero_shot_tasks, num_fewshot=0, batch_size=eval_batch_size)[
                'results'
            ]

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in zs_results.items()}
            acc_avg = calculate_avg_accuracy(zero_shot_tasks, zs_results)
            metric_vals['average_zero_shot'] = round(acc_avg, 4)
            
            for k, v in metric_vals.items():
                print("Task Name: " + k + " Task Score: " + v)
    
    if args.eval_mmlu:
        hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=eval_batch_size) 
        with torch.no_grad():
            print("Evaluating MMLU!")

            mmlu_tasks = ["mmlu_abstract_algebra", "mmlu_business_ethics", "mmlu_college_computer_science", \
                            "mmlu_college_mathematics", "mmlu_conceptual_physics", "mmlu_formal_logic", "mmlu_machine_learning",\
                                "mmlu_miscellaneous", "mmlu_philosophy", "mmlu_global_facts"]
            
            mmlu_results = lm_eval.simple_evaluate(hflm, tasks=mmlu_tasks, num_fewshot=5, batch_size=eval_batch_size)[
                'results'
            ]

            print(mmlu_results)

            metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in mmlu_results.items()}

            mmlu_avg = calculate_avg_accuracy(mmlu_tasks, mmlu_results)
            metric_vals['average_mmlu'] = round(mmlu_avg, 4)
            
            for k, v in metric_vals.items():
                print("Task Name: " + k + " Task Score: " + v)
            
    return

if __name__ == "__main__":

    run_id = int(sys.argv[1])

    exper_config = OATS_exper[run_id - 1]

    pruning_args =  ConfigDict(exper_config)
    process_pruning_args(pruning_args)

    checkpoint_path = "../" # Please fill in the desired directoy for checkpointing the pruned model

    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    pruning_main(pruning_args, checkpoint_path)
