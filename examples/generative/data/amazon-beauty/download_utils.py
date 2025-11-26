"""Utility functions for handling Amazon Beauty dataset files.

This module provides functions to check and extract dataset files.
"""

import gzip
import os
import shutil
from pathlib import Path


def extract_gz_file(gz_path, output_path):
    """Extract .gz file.
    
    Args:
        gz_path: Path to .gz file
        output_path: Path to save extracted file
    """
    try:
        print(f"\n📦 Extracting: {os.path.basename(gz_path)}")

        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"✅ Extraction complete: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def cleanup_gz_file(gz_path):
    """Delete .gz file to save space.
    
    Args:
        gz_path: Path to .gz file
    """
    try:
        if os.path.exists(gz_path):
            size_mb = os.path.getsize(gz_path) / (1024 * 1024)
            os.remove(gz_path)
            print(f"🗑️  Cleaned up: {os.path.basename(gz_path)} ({size_mb:.2f} MB)")
            return True
    except Exception as e:
        print(f"⚠️  Failed to cleanup {gz_path}: {e}")
    return False


def ensure_file_exists(filename, urls, data_dir, auto_download=True):
    """Ensure a file exists, download if necessary.

    Args:
        filename: Name of the file (e.g., 'meta_Beauty.json')
        urls: Download URL or list of URLs (not used, kept for compatibility)
        data_dir: Directory to save the file
        auto_download: Whether to show download instructions if file is missing

    Returns:
        Path to the file if successful, None otherwise
    """
    file_path = os.path.join(data_dir, filename)

    # File already exists
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"✅ File already exists: {filename} ({size_mb:.2f} MB)")
        return file_path

    # File doesn't exist
    if not auto_download:
        print(f"❌ File not found: {file_path}")
        return None

    # Show manual download instructions
    print(f"\n⚠️  File not found: {filename}")
    print(f"   Location: {file_path}")
    print(f"\n📖 Manual download instructions:")
    print(f"   1. Visit: https://nijianmo.github.io/amazon/index.html")
    print(f"   2. Select 'Beauty' category")
    print(f"   3. Fill the form to request access")
    print(f"   4. Download {filename}.gz")
    print(f"   5. Extract to: {data_dir}")
    print(f"   6. Run this script again")

    return None
