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
                pattern.search(line)[0] for line in file if pattern.search(line)
            )
    except OSError as e:
        print(f"Error reading file {filename}: {e}")
        return None


# def name_pairs(original: str, updated: str) -> Dict:
#     """
#     Get old and new file names and create dictionary from them.

#     Args:
#     original (str): File path to the file with the original filenames.

#     updated (str): File path to the file with the updated filenames.

#     Returns:
#      Dictionary: keys are original names and values are updated names.
#     """
#     h3_pattern = re.compile('^(###\s.+)')
#     original_h3 = []
#     updated_h3 = []


#     with open(original, 'r') as f:
#         original_names = f.readlines()

#     with open(updated, 'r') as f:
#         updated_names = f.readlines()

#     for old_line,new_line in zip(original_names,updated_names):
#         if h3_pattern.search(old_line) != None:
#             match_old = h3_pattern.search(old_line)[0]
