from typing import Dict, Tuple, List, Optional
import json
import os
import re


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

# Constants
# INPUT_UPDATED = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/input"
# EXPORT_DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/"
# EXPORT_JSON = "exported_article_details.json"
# EXPORT_JSON_PATH = os.path.join(EXPORT_DIR, EXPORT_JSON)


def export_article_details_as_json(input_dir: str, export_json_path: str) -> None:
    """Exports article details as a JSON file."""
    all_articles = {}
    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                articles_details = extract_details(content)
                all_articles.update(articles_details)

    with open(export_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(all_articles, json_file, indent=4)

# export_article_details_as_json(INPUT_UPDATED, EXPORT_JSON_PATH)

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

# --- Subheader: write_string_to_file --- {{{

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

# End Subheader write_string_to_file }}}


def extract_values_from_nested_dict(nested_dict: dict, target_key: str) -> Dict:
    """Extract values from a nested dictionary based on a target key.

    Args:
        nested_dict (dict): The dictionary to extract the values under the target key, if found.
        target_key (str): The key to match from another dictionary.

    Returns:
        dict: A dictionary of values extracted from the child of the matched key, or an empty dictionary if the key is not found.
    """

    if target_key in nested_dict:
        return nested_dict[target_key]
    else:
        return {}


# Regular Expressions to identify sections
PATT_ARTICLE = re.compile(r"^### .+\n\n- \*\*Description:\*\* .+\n- \*\*Tags:\*\* .+\n- \*\*Category:\*\* .+\n", re.MULTILINE)
PATT_TITLE = re.compile(r"^### (.+)$", re.MULTILINE)
PATT_DESCRIPTION = re.compile(r"- \*\*Description:\*\* (.+)")
PATT_TAGS = re.compile(r"- \*\*Tags:\*\* \['(.+)'\]")
PATT_CATEGORY = re.compile(r"- \*\*Category:\*\* _(.+)_")
PATT_WORD_COUNT = re.compile(r"\*\*Word Count:\*\* (\d+)")
PATT_FULL_ARTICLE = re.compile(r"\*\*\[Full Article\]\((.+)\)\*\*")


def extract_details(content: str) -> Dict[str, Dict[str, str]]:
    """Extracts article details from markdown content."""
    articles = {}
    for article in PATT_ARTICLE.findall(content):
        title = PATT_TITLE.search(article).group(1) if PATT_TITLE.search(article) else ""
        description = PATT_DESCRIPTION.search(article).group(1) if PATT_DESCRIPTION.search(article) else ""
        tags = PATT_TAGS.search(article).group(1) if PATT_TAGS.search(article) else ""
        category = PATT_CATEGORY.search(article).group(1) if PATT_CATEGORY.search(article) else ""
        word_count = PATT_WORD_COUNT.search(article).group(1) if PATT_WORD_COUNT.search(article) else ""
        full_article = PATT_FULL_ARTICLE.search(article).group(1) if PATT_FULL_ARTICLE.search(article) else ""

        articles[title] = {
            'description': description,
            'tags': tags.split("', '"),
            'category': category,
            'word_count': int(word_count),
            'full_article': full_article
        }
    return articles


def process_markdown_file(title_mapping_file_path: str, file_path: str) -> str:
    # Updated header data for all keys
    loaded_data = load_json_from_file(title_mapping_file_path)
    for key in loaded_data:
        print(loaded_data[key]['tags'])

    # Markdown article file path
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        print(lines[:10])

    old_header_values = {}
    old_header_values['title_current_file'] = lines[2]
    assert lines[2].startswith("title"), "Line 2 does not start with title"
    old_header_values['description_current_file'] = lines[4]
    assert lines[4].startswith("description"), "Line 4 does not start with description"
    old_header_values['tags_current_file'] = lines[6]
    assert lines[6].startswith("tags"), "Line 6 does not start with tags"
    old_header_values['category_current_file'] = lines[7]
    assert lines[7].startswith("category"), "Line 7 does not start with category"

    line_numbers = [2,4,6,7]
    print(old_values)
    # articles = extract_details(lines)

    # for idx, old_values in zip(line_numbers, old_header_values.values()):
    #     new_line = 

    
    # # Locate the yaml header
    # if lines[0].strip() != '---':
    #     return "" # Not a valid YAML header
    # header_end_idx = lines.index('---\n', 1)
    # yaml_content = ''.join(lines[1:header_end_idx])

    # # Parse YAML
    # header = yaml.safe_load(yaml_content)
    # # print(f"header {header}")

    # # Update title if present in TITLE_MAPPING_JSON_FILE
    # old_title = header.get('title','').replace('<br>', ' ').strip('\'"')
    # print(old_title)
# #    new_title = loaded_

    # new_title = loaded_data.get(old_title,old_title)
    # print(new_title)
    # header['title'] = new_title

    # # Serialize and rebuild file content
    # updated_yaml = yaml.safe_dump(header, default_flow_style=False,sort_keys=False)
    # print(f"updated_yaml \n {updated_yaml} \n")
    # write_string_to_file(file_path=os.path.join("/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/updated_yaml_header/","updated_yaml.md"), string_content=updated_yaml)
    # updated_content = '---\n' + updated_yaml + '---\n' + ''.join(lines[header_end_idx + 1:])

    # return updated_content


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
            
    
# export_updated_markdown_files(DIR,EXPORT_DIR,TITLE_MAPPING_JSON_FILE)
process_markdown_file(os.path.join(DIR,'automation-using-a-test-harness-br-for-deep-learning-br-part-2.md'),TITLE_MAPPING_JSON_FILE)
