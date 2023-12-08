import os
import re
import json
from typing import Dict
import pprint


EXPORT_DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/"
EXPORT_JSON = "exported_article_details.json"
# File path of extracted markdown details from updated.md
EXPORT_JSON_PATH = os.path.join(EXPORT_DIR, EXPORT_JSON)
# old_title:new_title mappings
TITLE_MAPPING_JSON_FILE = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details/export_old_new_name_pairs/old_new_names.json"
# output file that leaves original file untouched as backup
FILE_TO_WRITE = os.path.join(EXPORT_DIR,"old_new_mapping.json")

def pretty_print_dict_items(dict_name: dict) -> None:
    """
    Pretty print key and value pairs of a dictionary.

    Args:
    dict_name (dict): The dictionary whose items are to be pretty printed.

    Returns:
    None
    """
    import pprint # Local import
    pp = pprint.PrettyPrinter(indent=4)
    for key, value in dict_name.items():
        print(f"Key: {key}\nValue:")
        pp.pprint(value)
        print()  # For an extra line after each item

def update_json_with_original_titles(EXPORT_JSON_PATH: str, FILE_TO_WRITE: str, title_mapping_file_path: str) -> None:
    """Opens the file with the extracted markdown details to accompany the updated titles.
    And the file containing the original title and replaces the current keys with the original name
    . Makes the original key, a child node of the new key for each file.
    """

    with open(title_mapping_file_path,'r',encoding='utf-8') as file:
        title_mapping_file = json.load(file)

    with open(EXPORT_JSON_PATH,'r',encoding='utf-8') as file:
        articles = json.load(file)

    updated_articles = {}
    for original_title, updated_title in title_mapping_file.items():
        if updated_title in articles:
            updated_articles[original_title] = articles[updated_title]
            updated_articles[original_title]['updated_title'] = updated_title

    pretty_print_dict_items(updated_articles)

    with open(FILE_TO_WRITE,'w') as f:
        json.dump(updated_articles,f, ensure_ascii=False, indent=4)


update_json_with_original_titles(EXPORT_JSON_PATH,FILE_TO_WRITE,TITLE_MAPPING_JSON_FILE)
