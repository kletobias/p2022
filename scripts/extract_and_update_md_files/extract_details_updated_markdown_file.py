import os
import re
from typing import List, Dict

# Constants
DIR = "/path/to/markdown/files/"  # Replace with your actual directory
EXPORT_FILE = "exported_article_details.md"

# Regular Expressions to identify sections
PATT_TITLE = re.compile(r"^### (.+)$", re.MULTILINE)
PATT_DESCRIPTION = re.compile(r"- \*\*Description:\*\* (.+)")
PATT_TAGS = re.compile(r"- \*\*Tags:\*\* \['(.+)'\]")
PATT_CATEGORY = re.compile(r"- \*\*Category:\*\* _(.+)_")
PATT_WORD_COUNT = re.compile(r"\*\*Word Count:\*\* (\d+)")
PATT_FULL_ARTICLE = re.compile(r"\*\*\[Full Article\]\((.+)\)\*\*")

def extract_details(content: str) -> Dict[str, str]:
    """Extracts article details from markdown content."""
    details = {
        'title': PATT_TITLE.search(content).group(1) if PATT_TITLE.search(content) else "",
        'description': PATT_DESCRIPTION.search(content).group(1) if PATT_DESCRIPTION.search(content) else "",
        'tags': PATT_TAGS.search(content).group(1) if PATT_TAGS.search(content) else "",
        'category': PATT_CATEGORY.search(content).group(1) if PATT_CATEGORY.search(content) else "",
        'word_count': PATT_WORD_COUNT.search(content).group(1) if PATT_WORD_COUNT.search(content) else "",
        'full_article': PATT_FULL_ARTICLE.search(content).group(1) if PATT_FULL_ARTICLE.search(content) else ""
    }
    return details

def format_article_details(details: Dict[str, str]) -> str:
    """Formats the extracted details into the desired Markdown format."""
    return (f"### {details['title']}\n\n"
            f"- **Description:** {details['description']}\n"
            f"- **Tags:** ['{details['tags']}']\n"
            f"- **Category:** _{details['category']}_ | **Word Count:** {details['word_count']} | **[Full Article]({details['full_article']})**\n\n")

def export_article_details(directory: str, export_file: str) -> None:
    """Exports article details for all markdown files in a directory."""
    exported_content = ""
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                article_details = extract_details(content)
                formatted_details = format_article_details(article_details)
                exported_content += formatted_details

    with open(export_file, 'w', encoding='utf-8') as file:
        file.write(exported_content)

export_article_details(DIR, EXPORT_FILE)
