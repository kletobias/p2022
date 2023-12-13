import os
import re
import json

pat_title = re.compile(r"^title:\s['\"]([^']+)['\"]",re.IGNORECASE)

def update_title(input_md: str,title_mapping: str, output_dir: str) -> None:
    with open(input_md, 'r') as file:
        input_file = file.readlines()

    if input_file[0] == '---\n':
        input_title = input_file[2]
        print(input_title,type(input_title))
        # print(input_title)
        input_title_regex = pat_title.search(input_file[2]).group(1) if pat_title.search(input_file[2]) else ""
        print(input_title_regex)
        with open(title_mapping, 'r') as file:
            json_title_mapping = json.load(file)
        # print(json_title_mapping)
        
        for key in json_title_mapping:
            if key == input_title_regex:
                print(key)
                input_file[2] = f"title: '{json_title_mapping[key]}'\n"
                print(input_file[2])

        output_file_path = os.path.join(output_dir,os.path.basename(input_md)[:-3] + 'updated_title.md')
        with open(output_file_path, 'w') as file:
            file.writelines(input_file)
    else:
        print(f"Input file does not start with '---' on line 0, actual value is: {input_file[0]}")



input_dir = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details_manually/data/input/"
input_md = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details_manually/data/input/deep-dive-tabular-data-pt-1.md"
title_mapping = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details_manually/data/input/old_new_names.json"
output_dir = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details_manually/data/output/"

files_to_update = [os.path.join(input_dir,file) for file in os.listdir(input_dir) if file.endswith('.md')]

for file in files_to_update:
    update_title(file,title_mapping,output_dir)
