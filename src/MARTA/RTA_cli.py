#!/usr/bin/env python3
"""
CLI for the RTA analysis.
@author: Gionmattia Carancini
"""
# ~~~ Libraries ~~~
import argparse
import pandas as pd
import yaml
from importlib.resources import files
from MARTA.RTA_analysis import run_permutation_analysis

# ~~~ Functions ~~~
def get_default_config_path():
    return files("MARTA.config").joinpath("default_config.yaml")

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ~~~ Inputs Parsing ~~~
def parse_args():
    parser = argparse.ArgumentParser(
        prog="rta-analysis",
        description="Computes the RTA (Ratio of Translatinal Activity) between a baseline, target and noise control regions of a transcript."
                    "Requires the transcript coverage per position (.csv table) as input."
                    "Other options can be seen through the --help command."
    )

    # ~~~ Essentials - required at every run ~~~ #
    # Required argument - input path (to a csv table of coverage)
    parser.add_argument("--input", "-i",
        required=True,
        help="Path to the trancript coverage table (.csv format)"
    )

    # Required arguments - config file with analysis parameters
    parser.add_argument("--config", "-c", default=None,
                        help="Path to YAML config file. If not provided, uses package default."
                        )

    # Required argument - output path
    parser.add_argument("--output", "-o",
        required=True,
        help="Path to store the output of the analysis"
    )

    # ~~~ Optional parameters - Can be given through CLI or default to package config ~~~ #
    # Baseline coordinates
    parser.add_argument("--baseline", "-b",default=None,
                        help="Baseline region transcript coordinates (eg. '0,150')"
                        )
    # Target coordinates
    parser.add_argument("--target", "-t",default=None,
                        help="Target region transcript coordinates (eg. '150,300')"
                        )
    # Noise control coordinates
    parser.add_argument("--noise", "-n",default=None,
                        help="Noise control region transcript coordinates (eg. '300,450')"
                        )
    # Number of permutations - N
    parser.add_argument("--permutations", "-N", type=int, default=None,
                        help="Number of iterations (10000 in default_config.yml)"
                        )
    # Removed nucleotides - Remove nt at the edge of regions
    parser.add_argument("--remove", "-r", type=int, default=None,
        help="Nucleotides to remove from regions edges to exclude start/stop peaks "
             "(0 in default_config.yml, assumes user coordinates already accounted for)"
                        )
    # Significance threshold
    parser.add_argument("--pvalthreshold", "-p", type=float, default=None,
        help="Significance threshold for the p-value/padj (default: 0.05)"
                        )
    # Permutations seed
    parser.add_argument("--seed", "-s", type=float, default=None,
        help="Seed for permutations (None in default_config.yml)"
                        )
    # Permutations seed
    parser.add_argument("--jobs", "-j", type=int, default=None,
        help="Number of parallel jobs allowed (3 in default_config.yml)"
                        )

    return parser.parse_args()


def resolve_params(args):
    """
    Assigns parameters following priority: CLI > user YAML > default YAML.
    """
    if args.config is not None:
        config = load_config(args.config)
        print(f"[MARTA] Using config: {args.config}")
    else:
        default_config_path = get_default_config_path()
        config = load_config(default_config_path)
        print(f"[MARTA] No config provided — using package default config!!")


    # CLI arguments will override everything (but only if explicitly provided)
    if args.baseline is not None:
        config["baseline"] = args.baseline
    if args.target is not None:
        config["target"] = args.target
    if args.noise is not None:
        config["noise"] = args.noise
    if args.remove is not None:
        config["remove"] = args.remove
    if args.permutations is not None:
        config["N"] = args.permutations
    if args.pvalthreshold is not None:
        config["pval_threshold"] = args.pvalthreshold
    if args.seed is not None:
        config["seed"] = args.seed
    if args.jobs is not None:
        config["jobs"] = args.jobs

    return config



# ~~~ Main ~~~
def main():
    args = parse_args()
    config = resolve_params(args)

    # Parse coordinates, define coordinates dictionary
    coordinates = {}
    x_base = tuple(int(x) for x in config["baseline"].split(","))
    y_base = tuple(int(x) for x in config["target"].split(","))
    n_base = tuple(int(x) for x in config["noise"].split(","))

    # Need to correct by config["remove"]!!
    coordinates["x_slice"] = (x_base[0] + config["remove"], x_base[1] - config["remove"])
    coordinates["y_slice"] = (y_base[0] + config["remove"], y_base[1] - config["remove"])
    coordinates["n_slice"] = (n_base[0] + config["remove"], n_base[1] - config["remove"])

    # compute ci as integer
    ci = 100 - (config["pval_threshold"]*100)

    # open input dataframe
    df = pd.read_csv(args.input)

    # MARTA analysis
    output_df = run_permutation_analysis(df=df,
                                         coordinates=coordinates,
                                         ci = ci,
                                         N = config["N"],
                                         n_jobs = config["jobs"],
                                         random_state=config["seed"])

    output_df.to_csv(args.output, index=False)


# ~~~ Entry point guard ~~~
if __name__ == "__main__":
    main()