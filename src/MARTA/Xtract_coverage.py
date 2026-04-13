"""
This module contains all the functions used to extract the coverage from BigWig Files, given an ensembl TRANSCRIPT ID
Author: Gionmattia Carancini
Date: 13-04-2026
"""

import pandas as pd
from joblib import Parallel, delayed
import multiprocessing as mp
import pyranges as pr
import pyBigWig
import random
import time
import os


def _extract_one(bw_path, regions_df):
    """
    :param bw_path: path to a signle .bigwig file
    :param regions_df: dataframe with genomic regions to extract
    :return: tuple. Study ID and dictionary of the three regions dataframes.
    """

    sid  = os.path.basename(bw_path).split(".")[0]
    out  = {}

    attempts = 0
    max_attempts = 2

    while attempts < max_attempts:
        try:
            bw = pyBigWig.open(bw_path)
            # iterate directly over the DataFrame
            for row in regions_df.itertuples(index=False):
                chrom, start, end, region = (
                    row.Chromosome, int(row.Start), int(row.End), row.Region
                )
                vals = bw.values(chrom, start, end, numpy=True)
                out[region] = pd.DataFrame({
                    "Position": range(start, end),
                    sid: vals
                })
            bw.close()
            break  # success, so exit the loop

        # Failsafe. Sometimes parallelisation cannot work if same file is opened at the same time by != forks.
        # We introduce a random sleep time for a worker, just to resolve the issue. If that does not work, then....
        # returns 0s instead.
        except:
            # Increase attempts counter
            attempts += 1
            if attempts < max_attempts:
                # Wait a random time to ensure no overlap
                wait = random.uniform(0.3, 1.5)
                time.sleep(wait)
            # If issue cannot be resolved by waiting, returns zeros.
            else:
                for row in regions_df.itertuples(index=False):
                    chrom, start, end, region = (
                        row.Chromosome, int(row.Start), int(row.End), row.Region
                    )
                    # Fill with zeros (or None if you prefer)
                    out[region] = pd.DataFrame({
                        "Position": range(start, end),
                        sid: None
                    })

    return sid, out


def extract_tx_pyranges(gr, txid):
    """
    Function to load the gtf annotation and extract the ranges for selected transcript

    :param gr: loaded gtf, as pyranges object
    :param txid: the stable (ensebml) transcript id
    :return: a dataframe with ranges for each feature of txid
    """
    # Read the gtf file
    transcript_ranges = gr[gr.transcript_id == txid]
    if len(transcript_ranges) == 0:
        raise KeyError(f"transcript id {txid} was not found in the annotation file.")

    return transcript_ranges


# Function to format the transcript ranges, for extraction of coverage from BigWigs
def regions_df_formatter(transcript_ranges):
    """
    :param transcript_ranges: a pyranges.pyranges_main.PyRanges object
    :param return_individual_regions: whether to return individual regions (3', 5', cds) dataframe coordinates or as a single one
    :return: tuple, a singe dataframe, with the coordinates for each exon fragment. The strandedness of the feature extracted.

    The "Region" tag is necessary to name each range differently, so that the _extract_one() helper can use these tags as key when
    building the dictionary of coverages per exonic fragment.
    """

    # First, convert object to a dataframe, for manipulation
    my_ranges = transcript_ranges.as_df()[['Chromosome', 'Feature', 'Start', 'End', 'transcript_id', 'gene_name', 'exon_number', 'exon_id', "Strand"]]

    # Check strandedness of the transcript features is unique (It should be, but errors in annotation happen).
    if my_ranges["Strand"].nunique() != 1:
        raise AssertionError("Transcript should be on one strand, but the selected one seems to have elements on both/none strands.")
    else:
        strandedness = my_ranges["Strand"].unique()

    # Extract features coordinates, divided between CDS, utr5 and utr3
    ranges_df = my_ranges[my_ranges["Feature"]== "exon"].copy()

    # Resets the index, based on the order of features
    ranges_df = ranges_df.sort_values(by = "Start", ascending = True).reset_index(drop = True)

    # Applies the index to the "Region" tag (used by _extract_one() later)
    ranges_df["Region"] = ranges_df["Feature"] + "|" + ranges_df["exon_number"]

    assert ranges_df["Start"].is_monotonic_increasing, "Something when wrong when extracting regions. Check regions_dataframe_formatter"

    return ranges_df, strandedness


def extract_features_length(ranges, strandedness):
    """

    :param all_ranges: A dataframe with the ranges for each region (CDS, utrs)
    :return: the length of the three regions
    """

    # Extract Feature informations available.
    df = ranges.as_df()

    features = df["Feature"].unique()
    # Calculate length of each individual feature
    df["length"] = df["End"] - df["Start"]
    # Calculate length by grouped feature (CDS, utrs)
    features_breakdown = df.groupby("Feature")["length"].sum()

    # Retrieve lengths based on presence of features on annotation (defaults to None)
    cds_len = features_breakdown.get("CDS", None)
    stop_len = features_breakdown.get("stop_codon", None)
    utr5_len = features_breakdown.get("five_prime_utr", None)
    utr3_len = features_breakdown.get("three_prime_utr", None)

    # Combine stop codon to cds
    cds_length = cds_len + stop_len

    # If utr length is missing (because not annotated) retrieves it from the genomic coordinates.
    if (utr5_len is None or utr3_len is None) and ("start_codon" in features and "stop_codon" in features):
        # padding changes depending on forward/reverse strand:
        if strandedness == "+":
            utr5_len = retrieve_5_pad(df, strandedness)
            utr3_len = retrieve_3_pad(df, strandedness)
        if strandedness == "-":
            utr5_len = retrieve_3_pad(df, strandedness)
            utr3_len = retrieve_5_pad(df, strandedness)

    return utr5_len, cds_length, utr3_len


#  helper function for extract_features_lengths
def retrieve_5_pad(ranges_df, strandedness):
    df = ranges_df[~ranges_df["Feature"].isin(["transcript", "CDS", "five_prime_utr", "three_prime_utr"])].copy()

    if strandedness == "+":
        # Retrieve start position (target)
        target = df[df["Feature"] == "start_codon"]["Start"].iloc[0]
        # Extract all exons starting before the start codon
        relevant_exons = df[df['Start'] <= target].copy()
        # Use clip to have a "roof value"
        relevant_exons['Clipped_End'] = relevant_exons['End'].clip(upper=target)
        # retrieve position
        left_pad = (relevant_exons['Clipped_End'] - relevant_exons['Start']).sum()

    elif strandedness == "-":
        # Retrieve start position (target)
        target = df[df["Feature"] == "stop_codon"]["Start"].iloc[0]

        # Extract all exons starting before the start codon
        relevant_exons = df[df['Start'] <= target].copy()
        # Use clip to have a "roof value"
        relevant_exons['Clipped_End'] = relevant_exons['End'].clip(upper=target)
        # retrieve position
        left_pad = (relevant_exons['Clipped_End'] - relevant_exons['Start']).sum()

    return left_pad

#  helper function for extract_features_lengths
def retrieve_3_pad(ranges_df, strandedness):
    df = ranges_df[~ranges_df["Feature"].isin(["transcript", "CDS", "five_prime_utr", "three_prime_utr"])].copy()

    if strandedness == "+":
        # Retrieve start position (target)
        target = df[df["Feature"] == "stop_codon"]["End"].iloc[0]
        # Extract all exons starting before the start codon
        relevant_exons = df[df['End'] >= target].copy()
        # Use clip to have a "roof value"
        relevant_exons['Clipped_Start'] = relevant_exons['Start'].clip(lower=target)
        # retrieve position
        right_pad = (relevant_exons['End'] - relevant_exons['Clipped_Start']).sum()

    elif strandedness == "-":
        # Retrieve start position (target)
        target = df[df["Feature"] == "start_codon"]["End"].iloc[0]

        # Extract all exons starting before the start codon
        relevant_exons = df[df['End'] >= target].copy()
        # Use clip to have a "roof value"
        relevant_exons['Clipped_Start'] = relevant_exons['Start'].clip(lower=target)
        # retrieve position
        right_pad = (relevant_exons['End'] - relevant_exons['Clipped_Start']).sum()

    return right_pad


# Function to extract coverage for a single transcript from a single bigwig
def extract_mRNA_regions_from_bigwig(all_ranges_df, bigwig_file):
    """
    :param all_ranges_df: dataframe with genomic ranges (coordinates) for all elements of the mRNA
    :param bigwig_file: the path to a BigWig file
    :return: full transcript coverage, as a dataframe. SRR is kept a column name

    This function uses the internal helper _extract_one to extract
    the coverage for each element (exon or exon fragment) making up the mRNA regions
    (utr5_tuple, cds_tuple, utr3_tuple).

    The result of the internal helper is parsed to create the full mRNA coverage dataframe.

    An assert statement makes sure the concatenation respected the sequence order, by checking
    the genomic coordinates associated with each position.

    """

    # Extract from a single bigwig the coverage for all regions whose coordinates are specified in the three dataframes
    all_regions_tuple = _extract_one(bigwig_file, all_ranges_df)

    # Concatenate the fragments into the full mature mRNA
    full_transcript = pd.concat(all_regions_tuple[1].values(), ignore_index=True)

    # Sanity check: if the dataframes have been correctly created, we expect the positions to be monotonic crescent
    assert full_transcript["Position"].is_monotonic_increasing, "Genomic coordinates are not in ascending order. Something went wrong"

    return full_transcript



# Function to extract the single transcript from multiple bigwigs
def extract_from_multiple_bigwigs(bigwig_files, all_ranges, strandedness, n_jobs = 1):
    """
    :param bigwig_files: List of paths pointing to bigwigs to use
    :param transcript_ranges: pyranges object from the .gtf annotation file
    :param n_jobs: how many cores to allocate
    :return: tuple. Study ID and dictionary of the three regions dataframes.
    """

    # Get the parent interpreter context to enable parallel execution
    # context  = mp.get_context("fork")

    # Extracts in parallel the coverages from each bigwig
    # NEEDS TO BE FIXED SO TO DIVIDE BIGWIGS INTO CHUNKS AND ASSIGN A CHUNK TO EACH WORKER!!
    results = Parallel(n_jobs=n_jobs,
                       backend="multiprocessing") (
        # Needs to be modified so to handle fused df...?
        delayed(extract_mRNA_regions_from_bigwig)(all_ranges, bw) for bw in bigwig_files
    )

    ### Need to define how to merge dataframes together, but it should be easy to do...
    all_studies_df = results[0]
    for df in results[1:]:
        all_studies_df = all_studies_df.merge(df, on="Position", how="outer")

    ### and flip here if the strand is - ... (we can flip all dfs at this point.
    if strandedness == "-":
        all_studies_df = all_studies_df.sort_values("Position", ascending=False).reset_index(drop=True)

    all_studies_df.fillna(0, inplace=True)

    return all_studies_df


def read_gtf_annotation(gtf_path):
    """
    This function loads a .gtf annotation file as a pyRanges object.
    Created so that data is cached and execution is faster on subsequent runs.

    :param gtf_path: path to annotation file
    :return: pyRanges object
    """
    gr = pr.read_gtf(gtf_path)

    return gr


