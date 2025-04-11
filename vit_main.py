import logging
import os
import pathlib

import sys
import torch

from vit.vit_configs import vit_exper
from config import config
from vit.prep_imagenet import build_dataset
from vit.get_vit import get_vit, load_oats_vit
from vit.prune_oats_vit import prune_oats_vit
from vit.viz import viz_vit

from ml_collections import ConfigDict

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


def vit_main(args, checkpoint_path, rollout_path, eval_model=False) -> None:
    torch.manual_seed(args.seed)
    # Checkpointing for pruning
    if os.path.exists(checkpoint_path + "/prune_chkpt.pt"):
        model_adapter, image_processor = load_oats_vit(
                args.model,
                args.sparsity,
                args.prune_hyper['rank_ratio'],
                checkpoint_path,
                dtype=config.dtype
            )
    else:
         # load one of the pre-trained models
        model_adapter, image_processor = get_vit(
            args.model, dtype=config.dtype
        )
    
    model = model_adapter.model
    model.to(config.device)

    dataset_calib = build_dataset(args.cal_dataset, image_processor)
    calib_sampler = torch.utils.data.SubsetRandomSampler(torch.randperm(len(dataset_calib))[:args.cal_nsamples])
    calib_loader = torch.utils.data.DataLoader(
        dataset_calib, sampler=calib_sampler,
        batch_size=args.cal_batch_size,
        num_workers=4,
        pin_memory=False,
        drop_last=False
    )

    prune_oats_vit(model_adapter, args.sparsity, calib_loader, args.prune_hyper, checkpoint_path)
    print(model)

    if eval_model:
        dataset_val = build_dataset(args.eval_dataset, image_processor)
        eval_sampler = torch.utils.data.SequentialSampler(dataset_val)
        val_loader = torch.utils.data.DataLoader(
            dataset_val, sampler=eval_sampler,
            batch_size=256,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
        top_1, top_5 = evaluate(model, val_loader)
        print(f'Top-1 Accuracy: {top_1 * 100:.2f}%')
        print(f'Top-5 Accuracy: {top_5 * 100:.2f}%')
    
    split_parts = ["sparse", "low_rank"]
    for split_part in split_parts:
        viz_vit(model_adapter, split_part, image_processor, rollout_path)
        # re-load full model
        model_adapter, image_processor = load_oats_vit(
                args.model,
                args.sparsity,
                args.prune_hyper['rank_ratio'],
                checkpoint_path,
                dtype=config.dtype
            )
    return

def evaluate(model, test_loader):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch[0].to(config.device)
            labels = batch[1].to(config.device)

            outputs = model(inputs).logits

            _, top1_pred = torch.max(outputs, 1)
            top5_pred = outputs.topk(5, 1, True, True)[1]

            top1_correct += torch.sum(top1_pred == labels).item()
            top5_correct += torch.sum(top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred))).item()
            total += labels.size(0)
    
    top1_accuracy = top1_correct / total
    top5_accuracy = top5_correct / total

    return top1_accuracy, top5_accuracy

if __name__ == "__main__":

    run_id = int(sys.argv[1])

    exper_config = vit_exper[run_id - 1]

    pruning_args =  ConfigDict(exper_config)
    process_pruning_args(pruning_args)

    checkpoint_path = "../pruned_models/" + str(run_id) # Please fill in the desired directory for checkpointing the pruned models
    rollout_path = "../rollout/" + str(run_id)          # Please fill in the desired directory for saving attention rollout visualizations
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(rollout_path).mkdir(parents=True, exist_ok=True)

    vit_main(pruning_args, checkpoint_path, rollout_path)