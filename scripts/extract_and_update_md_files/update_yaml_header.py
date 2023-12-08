import os
import re
from typing import Dict, Tuple, List


# Constants
DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/_projects/"
EXPORT_DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/updated_yaml_header/"

def create_export_file_paths(input_dir: str, export_dir: str) -> List[str]:
    input_file_names = [filename for filename in os.listdir(input_dir) if filename.endswith('.md')]
    export_file_names = [os.path.splitext(filename)[0] + '_updated_header.md' for filename in input_file_names]

    return [os.path.join(export_dir,export_file_name) for export_file_name in export_file_names]

export_paths = create_export_file_paths(DIR,EXPORT_DIR)

print(export_paths)
