import os
import json
from typing import Dict

with open("/Users/tobias/all_code/projects/portfolio-website-2022/scripts/update_file_details/export_old_new_name_pairs/old_new_names.json",'r') as file:
    pairs_dict = json.load(file)

for key, value in pairs_dict.items():
    print(key)
    print(value)


