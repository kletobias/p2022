import os
import re
from typing import Dict, Optional, Tuple


def read_lines_with_pattern(filename: str, pattern: re.Pattern) -> Optional:
    """
    Read lines from a file and extract matching patterns.

    Args:
    filename (str): File path to read.

    Returns:
    Optional[Tuple[str]]: Tuple of matched strings, or None if the file can't be opened.
    """
    try:
        with open(filename, "r") as file:
            return tuple(
                pattern.search(line)[1] for line in file if pattern.search(line)
            )
    except OSError as e:
        print(f"Error reading file {filename}: {e}")
        return None


def name_pairs(original: str, updated: str) -> Dict[str, str]:
    """
    Create a dictionary mapping old filenames to new filenames.

    Args:
    original (str): File Path to the file with the original filenames.
    updated (str): File Path to the file with the updated filenames.

    Returns:
    Dict[str, str]: Dictionary where the keys are the original names and values are the updated names.
    """
    h3_pattern = re.compile(r'^###\s(.+)')
    
    original_h3 = read_lines_with_pattern(original,h3_pattern)
    updated_h3 = read_lines_with_pattern(updated, h3_pattern)

    if original_h3 is None or updated_h3 is None:
        return {}

    return(dict(zip(original_h3, updated_h3)))


original = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details/original.md"
updated = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details/updated.md"

pairs_dict = name_pairs(original,updated)

for key, value in pairs_dict.items():
    print('\n')
    print(key)
    print(value)
    print('\n\n')
