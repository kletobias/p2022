import os
import re
from typing import Dict, Tuple, List


# Constants
DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/_projects/"
EXPORT_DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/updated_yaml_header/"
TITLE_MAPPING_JSON_FILE = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/old_new_mapping.json"

def create_export_file_paths(input_dir: str, export_dir: str) -> List[str]:
    input_file_names = [filename for filename in os.listdir(input_dir) if filename.endswith('.md')]
    export_file_names = [os.path.splitext(filename)[0] + '_updated_header.md' for filename in input_file_names]

    return [os.path.join(export_dir,export_file_name) for export_file_name in export_file_names]

def write_string_to_file(file_path: str, string_content: str) -> None:
    """
    Write a string to a text file.

    Args:
    file_path (str): The path to the file where the string should be written.
    string_content (str): The string content to write to the file.

    Returns:
    None
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(string_content)
    except Exception as e:
        print(f"Error writing to file: {e}")

# write_string_to_file(file_path='path_to_file.txt', string_content=your_string_variable)

def load_json_from_file(file_path: str) -> dict:
    """
    Load JSON data from a file.

    Args:
    file_path (str): The path to the JSON file to be read.

    Returns:
    dict: The JSON data loaded from the file.
    """
    import json

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from file: {e}")
        return {}

# loaded_data = load_json_from_file(file_path=TITLE_MAPPING_JSON_FILE)

