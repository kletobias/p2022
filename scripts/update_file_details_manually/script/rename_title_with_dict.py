import os
import re
import json

def update_title(input_md: str,title_mapping: str, output_dir: str) -> None:
    with open(input_md, 'r') as file:
        input_title = file.readlines()[2]
    print(input_title)
    with open(title_mapping, 'r') as file:
        json_title_mapping = json.load(file)



input_md = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details_manually/data/input/deep-dive-tabular-data-pt-1.md"
output_dir = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details_manually/data/output/"

update_title(input_md,output_dir)
