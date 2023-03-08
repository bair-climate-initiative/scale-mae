import argparse
import os
from argparse import Namespace

import torch
import wandb
import yaml

api = wandb.Api()
import sys
sys.path.append('/home/jacklishufan/scale-mae/mae')
import util.misc as misc
from main_pretrain import get_args_parser as pretrain_get_args_parser
from main_pretrain import main as main_pretrain


def get_args_parser():
    parser = argparse.ArgumentParser("Eval controller", add_help=False)
    parser.add_argument(
        "--eval_config",
        default=os.path.join(os.path.dirname(__file__), "evalconf/demo-conf.yaml"),
        type=str,
        help="Eval config file",
    )

    ###################################
    # DISTRIBUTED TRAINING PARAMETERS #
    ###################################
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--knn", default=20, type=int, help="Number of neighbors to use for KNN"
    )

    parser.add_argument(
        "--eval_gsd",
        action="store_true",
        help="USE GSD Relative Embedding with base=224x224",
    )
    parser.add_argument(
        "--no-eval_gsd",
        action="store_false",
        help="USE GSD Relative Embedding with base=224x224",
        dest='eval_gsd'
    )
    parser.set_defaults(eval_gsd=True)
    parser.add_argument(
        "--eval_base_resolution",
        default=1.0,
        type=float,
        help="Global Multiplication factor of Positional Embedding Resolution in KNN",
    )
    return parser


def main(args):
    print("Starting eval")
    with open(args.eval_config) as f:
        config = yaml.safe_load(f.read())

    if "exp_ids" in config:
        exp_ids = config["exp_ids"]
    else:
        # TODO collect all run ids or specify other conditions
        pass
    misc.init_distributed_mode(args)
    is_main = False
    if args.rank == 0:
        wandb_args = dict(
            project="scale-mae-knn-reproduce",
            entity="bair-climate-initiative",
            resume="allow",
            )
        run = wandb.init(**wandb_args)
        run_id = run.id
        is_main= True
    default_args = pretrain_get_args_parser().parse_args([])
    for expid in exp_ids:
        try:
            # load the latest checkpoint
            # TODO allow different epochs, scales, datasets
            mdl_path = os.path.join(config["root"], str(expid), "checkpoint-latest.pth")
            mdl = torch.load(mdl_path, map_location="cpu")
            margs = mdl["args"] if "args" in mdl else Namespace()
            nepochs = mdl["epoch"] if "epoch" in mdl else 100
            if nepochs < 90:
                print(f"Skipping {expid} because it only has {nepochs} epochs")
                continue

            # add all of the eval params
            for k, v in config.items():
                setattr(margs, k, v)
            # set all of the distributed bits
            margs.eval_gsd = args.eval_gsd
            margs.eval_base_resolution = args.eval_base_resolution
            margs.knn = args.knn
            margs.local_rank = args.local_rank
            margs.dist_on_itp = args.dist_on_itp
            margs.dist_url = args.dist_url
            margs.world_size = args.world_size
            # only do evaluation
            margs.resume = mdl_path

            for eval_data in config["evals"]:
                eval_id = eval_data["id"]
                margs.eval_scale = eval_data["scales"]
                margs.eval_dataset = eval_id
                print(f"Starting {margs.eval_dataset} {margs.eval_scale}: {eval_id}")
                margs.eval_only = True
                margs.eval_train_fnames = os.path.join(eval_data["path"], "train.txt")
                margs.eval_val_fnames = os.path.join(eval_data["path"], "val.txt")

                arg_vals = {**vars(default_args), **vars(margs)}
                use_args = Namespace(**arg_vals)
                use_args.base_resolution = 2.0
                res = main_pretrain(use_args)

                if is_main:
                    wandb_run = api.run(
                        f"bair-climate-initiative/scale-mae-knn-reproduce/{run_id}"
                    )
                    for scale, acc in res.items():
                        wandb_run.summary[f"{eval_id}-knn-acc-{scale}"]= acc * 100.0
                    wandb_run.summary.update()
                    print("Sent results", res)
                    print("HERE")
        except Exception as err:
            print(f"Unable to process (will skip) {expid}: {err}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
