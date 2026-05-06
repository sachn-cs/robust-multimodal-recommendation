"""Utilities for downloading and extracting raw Amazon review datasets."""

import gzip
import os
import shutil
import urllib.request


def download_file(url: str, dest_path: str) -> None:
    """Download a remote file to a local path.

    Args:
        url: The remote URL to download from.
        dest_path: The local filesystem path to write to.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with urllib.request.urlopen(url) as response, open(
        dest_path, "wb"
    ) as out_file:
        shutil.copyfileobj(response, out_file)


def download_amazon_dataset(
    category: str, data_dir: str = "data/raw"
) -> None:
    """Download Amazon 5-core review and metadata JSON for a category.

    Files are downloaded as ``.json.gz``, decompressed to ``.json``, and the
    archive is removed.

    Args:
        category: The Amazon product category slug.
        data_dir: Directory in which to store the downloaded files.

    Returns:
        None
    """
    base_url = (
        "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/"
    )
    review_file = f"{category}_5.json.gz"
    meta_file = f"{category}_metadata.json.gz"
    for fname in [review_file, meta_file]:
        dest = os.path.join(data_dir, fname)
        if not os.path.exists(dest.replace(".gz", "")):
            print(f"Downloading {fname} ...")
            download_file(base_url + fname, dest)
            with gzip.open(dest, "rb") as f_in, open(
                dest.replace(".gz", ""), "wb"
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(dest)
