import os
import re
import itertools
from typing import List, Dict, Tuple
from markdown2 import Markdown
import spacy

# Constants
DIR = "/Users/tobias/all_code/projects/portfolio-website-2022/_projects/"
OUTDIR = "/Users/tobias/all_code/projects/portfolio-website-2022/references/"
HOST = "https://deep-learning-mastery.com/projects/"

# Precompiled Regex Patterns
PATT_TITLE = re.compile("^title:\s['\"](.+)['\"]$", re.IGNORECASE)
PATT_DESCRIPTION = re.compile("^description:\s['\"](.+)['\"]$", re.IGNORECASE)
PATT_CATEGORY = re.compile("^category:\s\[['\"](.+)['\"]\]$", re.IGNORECASE)
PATT_TAGS = re.compile("^tags:\s(\[.+\])$", re.IGNORECASE)
PATT_SUB = re.compile("<br>")
PATT_HTML = re.compile(r"<[^>]+>|\\n", re.MULTILINE)

# Load Spacy Model
nlp = spacy.load("en_core_web_sm")


def create_links(files: List[str], host: str) -> List[str]:
    """Create markdown links from file names."""
    links = []
    for file in files:
        if file.endswith(".md"):
            file_base = file[:-3]
            links.append(f"[{file_base}]({host}{file_base}/)")
    return links


def get_word_count(text: str) -> int:
    """Calculate word count using Spacy."""
    doc = nlp(PATT_HTML.sub("", text))
    words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return len(words)


def parse_file_content(filepath: str) -> Dict[str, str]:
    """Parse file content for title, description, category, and tags."""
    with open(filepath, "r") as file:
        content = {}
        for line in itertools.islice(file, 2, 8):
            if (match := PATT_TITLE.search(line)) is not None:
                content["title"] = PATT_SUB.sub(" ", match.group(1))
            elif (match := PATT_DESCRIPTION.search(line)) is not None:
                content["description"] = PATT_SUB.sub(" ", match.group(1))
            elif (match := PATT_CATEGORY.search(line)) is not None:
                content["category"] = PATT_SUB.sub(" ", match.group(1))
            elif (match := PATT_TAGS.search(line)) is not None:
                content["tags"] = PATT_SUB.sub(" ", match.group(1))
    return content


def get_title_description(directory: str, files: List[str], host: str) -> List[str]:
    """Process markdown files to extract title, description, and generate HTML content."""
    cv_text_all = []
    for file in files:
        if file.endswith(".md"):
            file_path = os.path.join(directory, file)
            file_base = file[:-3]

            # Get word count
            with open(file_path, "r") as article:
                word_count = get_word_count(article.read())

            # Parse file content
            content = parse_file_content(file_path)
            content["url"] = f"[Full Article]({host}{file_base}/)"
            content["word_count"] = word_count

            # Generate HTML content
            cv_text = (f'<p><H3>{content["title"]}</H3></p>'
                       f'<p>**Description:** {content["description"]}<br>'
                       f'**Tags:** {content["tags"]}<br>'
                       f'
