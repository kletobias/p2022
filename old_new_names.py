import os
import re
from typing import Dict, Optional
import json


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


def name_pairs(original: str, updated: str) -> Optional:
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


def dump_dict_to_json(file_path: str, data: Dict[str, str]) -> None:
    """
    Dumps a dictionary to a JSON file.

    Args:
    file_path (str): The file path where the JSON should be saved.
    data (Dict[str, str]): The dictionary to be dumped into the file.

    Returns:
    None
    """
    export_dir = os.path.dirname(file_path)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data,file,ensure_ascii=False, indent=4)
    



root_dir = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details/"
original = os.path.join(root_dir,"original.md")
updated = os.path.join(root_dir,"updated.md")

pairs_dict = name_pairs(original,updated)
dump_dict_to_json(os.path.join(root_dir,"export_old_new_name_pairs","old_new_names.json"),pairs_dict)

for key, value in pairs_dict.items():
    print('\n')
    print(key)
    print(value)
    print('\n\n')
