"""
Download data from Dryad and unzip zip files into the data/raw directory.

Usage:
    conda activate b2txt25
    python src/dataset/download.py
"""

import sys
import json
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional


class DryadDownloader:
    """Class to download and extract Dryad dataset files."""

    def __init__(
        self,
        doi: str = "10.5061/dryad.dncjsxm85",
        data_dir: Optional[Path] = None,
        skip_files: Optional[List[str]] = None,
    ):
        self.doi = doi
        self.root_url = "https://datadryad.org"
        self.data_dir = data_dir or Path.cwd() / "data" / "raw"
        self.skip_files = skip_files or ["README.md"]
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def display_progress_bar(
        self, block_num: int, block_size: int, total_size: int, message: str = ""
    ) -> None:
        """Display a progress bar for downloads."""
        bytes_downloaded = block_num * block_size
        mb_downloaded = bytes_downloaded / 1e6
        mb_total = total_size / 1e6
        sys.stdout.write(
            f"\r{message}\t\t{mb_downloaded:.1f} MB / {mb_total:.1f} MB"
        )
        sys.stdout.flush()

    def get_latest_file_infos(self) -> List[Dict]:
        """Get file info list from the latest Dryad dataset version."""
        urlified_doi = self.doi.replace("/", "%2F")
        versions_url = f"{self.root_url}/api/v2/datasets/doi:{urlified_doi}/versions"
        with urllib.request.urlopen(versions_url) as response:
            versions_info = json.loads(response.read().decode())

        versions = versions_info["_embedded"]["stash:versions"]
        latest_version = versions[-1]
        files_url_path = latest_version["_links"]["stash:files"]["href"]
        files_url = f"{self.root_url}{files_url_path}"

        with urllib.request.urlopen(files_url) as response:
            files_info = json.loads(response.read().decode())

        return files_info["_embedded"]["stash:files"]

    def download_file(self, url: str, dest: Path, message: str = "") -> None:
        """Download a file from a URL to a destination path with a progress bar."""
        if dest.exists():
            print(f"{dest.name} already exists, skipping download.")
            return
        urllib.request.urlretrieve(
            url,
            dest,
            reporthook=lambda *args: self.display_progress_bar(*args, message=message),
        )
        sys.stdout.write("\n")

    def extract_zip(self, filepath: Path) -> None:
        """Extract a zip file to the data directory."""
        print(f"Extracting files from {filepath.name} ...")
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(self.data_dir)

    def download_and_extract_files(self, file_infos: List[Dict]) -> None:
        """Download files and extract zip files if needed."""
        for file_info in file_infos:
            filename = file_info["path"]
            if filename in self.skip_files:
                continue

            download_path = file_info["_links"]["stash:download"]["href"]
            download_url = f"{self.root_url}{download_path}"
            download_to_filepath = self.data_dir / filename

            self.download_file(
                download_url, download_to_filepath, message=f"Downloading {filename}"
            )

            if file_info["mimeType"] == "application/zip":
                self.extract_zip(download_to_filepath)

    def run(self) -> None:
        """Run the download and extraction process."""
        file_infos = self.get_latest_file_infos()
        self.download_and_extract_files(file_infos)
        print(f"\nDownload complete. See data files in {self.data_dir}\n")


def main() -> None:
    """Main function to download and extract Dryad data files."""
    downloader = DryadDownloader()
    downloader.run()


if __name__ == "__main__":
    main()