import os
import re
import json
from typing import Dict

# Constants
INPUT_UPDATED = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/input"
EXPORT_DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/scripts/extract_and_update_md_files/export/"
EXPORT_JSON = "exported_article_details.json"
EXPORT_JSON_PATH = os.path.join(EXPORT_DIR, EXPORT_JSON)

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

export_article_details_as_json(INPUT_UPDATED, EXPORT_JSON_PATH)
