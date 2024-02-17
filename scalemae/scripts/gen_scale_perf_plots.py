import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import wandb
import yaml

api = wandb.Api()


max_res = {
    "resisc": 256,
    "ucmerced": 256,
    "whu-rs19": 256,
    "airound": 496,
    "mlrsnet": 256,
    "cvbrct": 496,
    "eurosat": 64,
    "optimal-31": 256,
}

res_lims = {
    "resisc": [0.3, 0.9],
    "ucmerced": [0.4, 0.9],
    "whu-rs19": [0.4, 1.0],
    "airound": [0.35, 0.8],
    "mlrsnet": [0.5, 1.0],
    "cvbrct": [0.4, 0.8],
    "eurosat": [0.5, 1.0],
    "optimal-31": [0.35, 0.8],
}


def get_args_parser():
    parser = argparse.ArgumentParser("Eval controller", add_help=False)
    parser.add_argument("--runs", nargs="*", type=str, default=["1dcghih0", "2y9klhll"])

    parser.add_argument("--evals", nargs="*", type=str, default=sorted(max_res.keys()))

    parser.add_argument(
        "--px", nargs="*", type=int, default=[16, 32, 64, 128, 256, 496]
    )

    parser.add_argument("--names", nargs="*", type=str, default=["Scale-MAE", "SatMAE"])

    parser.add_argument("--output", type=str, default="results.png")

    return parser


def main(args):
    pd.DataFrame()

    # for each eval dataset
    ## for each method
    ##  for each resolution
    if not os.path.exists("cached.pkl"):
        all_data = []
        for i, rid in enumerate(args.runs):
            wandb_run = api.run(f"bair-climate-initiative/multiscale_mae/{rid}")
            name = args.names[i]
            print(name)
            for eval in args.evals:
                print(eval)
                wtable = api.artifact(
                    f"bair-climate-initiative/multiscale_mae/run-{rid}-{eval.replace('-','')}_eval:latest"
                ).get(f"{eval}_eval")
                if wtable is None:
                    import ipdb

                    ipdb.set_trace()
                for datum in wtable.data:
                    data = {name: val for val, name in zip(datum, wtable.columns)}
                    if data["val_resolution"] in args.px:
                        all_data.append(
                            dict(
                                result=data["acc"],
                                px=data["val_resolution"],
                                name=name,
                                valset=eval,
                            )
                        )
        data = pd.DataFrame(all_data)
        pd.to_pickle(data, "cached.pkl")
    else:
        print("using cache")
        data = pickle.load(open("cached.pkl", "rb"))

    # generate the performance plots
    nvals = data.valset.nunique()
    ncols = 4
    nrows = (nvals + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=0, figsize=(21, 6))
    ct = 0
    for ax in axs.reshape(-1):
        if ct >= nvals:
            break
        subdata = data[(data.valset == args.evals[ct]) & (data.result > -1)]
        subdata.px /= float(max_res[args.evals[ct]])
        subdata = subdata.groupby(["name"])

        legend = ct == 0  # nvals - 1
        subdata.plot(
            kind="line",
            x="px",
            y="result",
            ax=ax,
            legend=False,
            title=args.evals[ct].upper(),
            style=".-",
        )
        if legend:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], [x for x in subdata.groups.keys()][::-1])
        ax.set_xticks(
            ticks=[0, 0.25, 0.5, 0.75, 1.0], labels=["0", "25%", "50%", "75%", "100%"]
        )
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_ylim(res_lims[args.evals[ct]])
        if ct % ncols == 0:
            ax.set_ylabel("KNN acc.")
        else:
            ax.set_ylabel("")
        if ct // ncols == nrows - 1:
            ax.set_xlabel("Relative GSD")
        else:
            ax.set_xlabel("")
        ct += 1
    plt.subplots_adjust(hspace=0.3)
    fig.savefig(args.output, bbox_inches="tight")


if __name__ == "__main__":
    print("starting eval")
    main(get_args_parser().parse_args())
