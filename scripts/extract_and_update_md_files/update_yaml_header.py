from typing import Dict, Tuple, List, Optional
import os
import re
import yaml


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


def process_markdown_file(title_mapping_file_path: str, file_path: str) -> str:
    loaded_data = load_json_from_file(title_mapping_file_path)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Locate the yaml header
    if lines[0].strip() != '---':
        return "" # Not a valid YAML header
    header_end_idx = lines.index('---\n', 1)
    yaml_content = ''.join(lines[1:header_end_idx])

    # Parse YAML
    header = yaml.safe_load(yaml_content)
    # print(f"header {header}")

    # Update title if present in TITLE_MAPPING_JSON_FILE
    old_title = header.get('title','').replace('<br>', ' ').strip('\'"')
    print(old_title)
#    new_title = loaded_

    new_title = loaded_data.get(old_title,old_title)
    print(new_title)
    header['title'] = new_title

    # Serialize and rebuild file content
    updated_yaml = yaml.safe_dump(header, default_flow_style=False,sort_keys=False)
    print(f"updated_yaml \n {updated_yaml} \n")
    write_string_to_file(file_path=os.path.join("/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/updated_yaml_header/","updated_yaml.md"), string_content=updated_yaml)
    updated_content = '---\n' + updated_yaml + '---\n' + ''.join(lines[header_end_idx + 1:])

    return updated_content


def create_export_file_paths(input_dir: str, export_dir: str) -> List[str]:
    input_file_names = [filename for filename in os.listdir(input_dir) if filename.endswith('.md')]
    export_file_names = [os.path.splitext(filename)[0] + '_updated_header.md' for filename in input_file_names]

    return [os.path.join(export_dir,export_file_name) for export_file_name in export_file_names]


def export_updated_markdown_files(input_dir: str, export_dir: str, title_mapping_file_path: str) -> Optional[str]:
    for input_markdown_file in os.listdir(input_dir):
        if input_markdown_file.endswith('.md'):
            updated_content = process_markdown_file(TITLE_MAPPING_JSON_FILE,os.path.join(input_dir,input_markdown_file))
            assert updated_content != "", "updated content is not a valid file"
            # export_file_path
#            print(updated_content[0:500])
            
    
export_updated_markdown_files(DIR,EXPORT_DIR,TITLE_MAPPING_JSON_FILE)
