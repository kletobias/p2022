import os
import re
import json
from typing import Dict


EXPORT_DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/"
EXPORT_JSON = "exported_article_details.json"
# File path of extracted markdown details from updated.md
EXPORT_JSON_PATH = os.path.join(EXPORT_DIR, EXPORT_JSON)
# old_title:new_title mappings
TITLE_MAPPING_JSON_FILE = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details/export_old_new_name_pairs/old_new_names.json"


def update_json_with_original_titles(file_to_write: str, title_mapping_file: str) -> None:
    """Opens the file with the extracted markdown details to accompany the updated titles.
    And the file containing the original title and replaces the current keys with the original name
    . Makes the original key, a child node of the new key for each file."""
    with open(title_mapping_file,'r',encoding='utf-8') as file:
        title_mapping_file = json.load(file)

    with open(EXPORT_JSON_PATH,'r',encoding='utf-8') as file:
        articles = json.load(file)
