from functools import partial
import multiprocessing
import os
import argparse

from glob import glob
from tqdm import tqdm
from get_features import process_one

def process_batch(file_paths, max_workers):
    partial_process_one = partial(process_one, in_dir)
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(partial_process_one, file_paths), total=len(file_paths), desc="Feature_extract"))
    results = [result for result in results if result is not None]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir", type=str, default="dataset", help="path to input dir"
    )

    args = parser.parse_args()
    filenames = glob(f"{args.in_dir}/**/*.wav", recursive=True)+glob(f"{args.in_dir}/**/*.flac", recursive=True)  # [:10]
    in_dir = args.in_dir
    process_batch(filenames, 2)
