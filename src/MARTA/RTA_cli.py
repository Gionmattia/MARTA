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
    parser.add_argument("--alpha", "-a", type=float, default=None,
                        help="FDR significance threshold (default: 0.05)")
    # Confidence intervals boundaries
    parser.add_argument("--ci", type=int, default=None,
                        help="Confidence level (percent) for null/bootstrap CI bounds (default: 95)")
    # Permutations seed
    parser.add_argument("--seed", "-s", type=int, default=None,
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
    if args.alpha is not None:
        config["alpha"] = args.alpha
    if args.ci is not None:
        config["ci"] = args.ci
    if args.seed is not None:
        config["seed"] = args.seed
    if args.jobs is not None:
        config["jobs"] = args.jobs

    required = ["baseline", "target", "noise", "remove", "N", "alpha", "ci", "seed", "jobs"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

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
    r = config["remove"]
    coordinates["x_slice"] = (x_base[0] + r, x_base[1] - r)
    coordinates["y_slice"] = (y_base[0] + r, y_base[1] - r)
    coordinates["n_slice"] = (n_base[0] + r, n_base[1] - r)

    # Check. If a user removes more than the coordinates would allow (= generate an inverted slice...).
    for name, sl in coordinates.items():
        if sl[1] <= sl[0]:
            raise ValueError(
                f"Region '{name}' is empty or inverted after applying "
                f"--remove={config['remove']}: got slice {sl}. "
                f"Check that your region coordinates are large enough "
                f"to accommodate the trim you selected on both edges."
            )

    # compute ci as integer
    alpha = config["alpha"]
    ci = config["ci"]

    # open input dataframe
    df = pd.read_csv(args.input)

    # MARTA analysis
    output_df, random_seed = run_permutation_analysis(df=df,
                                         coordinates=coordinates,
                                         ci = ci,
                                         alpha=alpha,
                                         N = config["N"],
                                         n_jobs = config["jobs"],
                                         random_state=config["seed"])

    output_df.to_csv(args.output, index=True)

    # Save run provenance
    provenance = {
        "input": args.input,
        "output": args.output,
        "coordinates": {k: list(v) for k, v in coordinates.items()},
        "N": config["N"],
        "alpha": alpha,
        "ci": ci,
        "n_jobs": config["jobs"],
        "random_state": int(random_seed),
    }
    provenance_path = args.output.rsplit(".", 1)[0] + "_run_info.yaml"
    with open(provenance_path, "w") as f:
        yaml.safe_dump(provenance, f)

    print(f"[MARTA] Analysis complete.")
    print(f"[MARTA] Results: {args.output}")
    print(f"[MARTA] Run info: {provenance_path}")
    print(f"[MARTA] random_state used: {random_seed}")


# ~~~ Entry point guard ~~~
if __name__ == "__main__":
    main()