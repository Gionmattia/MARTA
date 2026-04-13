#!/usr/bin/env python3
"""
CLI for the extraction of coverage from bigwigs files. Uses Ensembl transcript ID.
@author: Gionmattia Carancini

Usage:
extract-coverage --bigwigs <path> --gtf <path> --txid ENSMUST00000180036 --output <path>
"""

# ~~~ Libraries ~~~
import argparse
from MARTA.Xtract_coverage import *

# ~~~ Inputs Parsing ~~~
def parse_args():
    parser = argparse.ArgumentParser(
        prog="extract-coverage",
        description="Extract the coverage (counts per nt position) from a series of bigwigs, given a transcript."
    )

    # ~~~ Essentials - required at every run ~~~ #
    # Required argument - input folder of bigwigs  ex. "/home/some/thing/m_musculus_bigwigs"
    parser.add_argument("--bigwigs", "-b", required=True,
        help="Path to the folder with the bigwigs "
    )

    # Required arguments - annotation file (gtf) ex. "/some/thing/Mus_musculus.GRCm39.113.gtf"
    parser.add_argument("--gtf", "-g", required=True,
                        help="Path to gtf annotation file. Must be same version as one used to generate the bigwigs."
                        )

    # Required arguments - Ensembl transcript ID ex. "ENSMUST00000180036"
    parser.add_argument("--txid", "-i", required=True,
                        help="Ensembl Transcript ID of interest."
                        )

    # Required argument - output path
    parser.add_argument("--output", "-o", required=True,
        help="Path to store the output table of coverage"
    )

    return parser.parse_args()



# ~~~  Main ~~~
def main():
    args = parse_args()

    # 0 List all bigwigs in the given directory
    folder_path = Path(args.bigwigs)
    if not folder_path.exists():
        raise FileNotFoundError(f"Bigwig folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Provided bigwig path is not a directory: {folder_path}")

    bigwig_files = [str(f.resolve()) for f in folder_path.glob("*.bw")]
    if not bigwig_files:
        raise FileNotFoundError(f"No bigwigs files found in: {folder_path}")

    # 1 Retrieve transcript ranges
    gr = read_gtf_annotation(args.gtf)
    transcript_ranges = extract_tx_pyranges(gr, args.txid)

    # 2  Retrieve transcript features (CDS, UTRs) ranges and strandedness
    all_ranges, strandedness = regions_df_formatter(transcript_ranges)

    # 3 Extract coverage from given bigwigs
    all_runs_df = extract_from_multiple_bigwigs(bigwig_files, all_ranges, strandedness, n_jobs=1)

    # 4 Remove genomic coordinate and save file
    all_runs_df = all_runs_df.drop(columns="Position")


    # 5 Assemble file name and output path
    out_name = args.txid + "_raw_counts.csv"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / out_name

    all_runs_df.to_csv(outpath, index=False)



# ~~~ Entry point guard ~~~
if __name__ == "__main__":
    main()
