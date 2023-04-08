---
layout: distill
title: 'Python Script To Set Up Paired Jupyter Notebooks Using Jupytext'
date: 2023-04-08
description: 'I use jupytext to create three empty paired notebooks, making the setup for new articles easier'
tags: ['writing', 'automation', 'python', 'jupytext', 'jupyter-notebook']
category: 'scripting'
comments: true
---
<br>
<d-contents>
  <nav class="l-text figcaption">
  <h3>Contents</h3>
    <div class="no-math"><a href="#description">Description</a></div>
    <div class="no-math"><a href="#the-script">The Script</a></div>
  </nav>
</d-contents>

# Python Script To Set Up Paired Jupyter Notebooks In A New Project

## Description

The process of creating new articles can be tedious, particularly if you use
Jupyter Notebooks or JupyterLab. My best solution to date has been to use a
library called
[*jupytext*](https://jupytext.readthedocs.io/en/latest/install.html). Using it,
you can create different formats of Jupyter Notebooks.

A function in this script creates three copies of a new Jupyter Notebook, given
a base directory `base`. The files represent the same Notebook, so they can be
kept in sync. There are three file extensions: .ipynb, .py, .md. The parameters
can be adjusted according to your needs. 


## The Script

```python
import os
from time import strftime
from slugify import slugify
import subprocess
from pathlib import Path


# Base Path
base = Path('/Users/tobias/all_code/projects/python_projects/portfolio_articles/_posts/')

def create_new_article(base=base,name=None,date=None,date_in_fname=False):

    """Function to create a new directory for the new article,
    three synced jupytext files with extensions: ipynb, py, md """
    # default for date is None, which means the current date is used. Use a
    # string value for a different date.
    if type(date) == str:
        date = date
    else:
        date=strftime("%Y-%m-%d")
    # Instructions for the case using user input for name of new article
    print("""

Enter the name of the article you want to create with or without spaces. Input
will be converted to all lower-case and cleaned. Results might not be what is
expected, if non English alphanumeric characters are used.

    """)
    # User input for parameter name (default) or set a value by passing a string
    # value for parameter name. 
    if type(name) == str:
        name = slugify(name)
    elif name == None:
        name = slugify(input("Name: "))

    # Print the name entered by the user
    print(f'Name is: {name}')

    # Define complete name of new directory using `date` and `name`
    article_dir = Path(base,f"{date}-{name}/")

    # Create new directory for article to be written. Will throw an error, if it
    # already exists
    article_dir.mkdir()

    # Use touch to create an empty markdown file for jupytext. Will throw an
    # error, if it already exists.
    if date_in_fname == True:
        article_file = article_dir.joinpath(f'{date}-{name}.md')
        article_file.touch(mode=0o755)
    else:
        article_file = article_dir.joinpath(f'{name}.md')
        article_file.touch(mode=0o755)

    # Run jupytext command line tool via subprocess to do the setup.
    # jupytext will create three files, a jupyter-notebook, a python file using
    # `# %%` as markers for code cells and `# %% [markdown]` for markdown cells.
    # The markdown file created earlier is used for the markdown version of the
    # notebook. Using --sync on the .ipynb file will sync all three versions.
    subprocess.run([f'jupytext --set-formats ipynb,py:percent,md {article_file}'],shell=True)

    # Check output
    print(f'\n {os.listdir(article_dir)} \n')

create_new_article(date_in_fname=True)
```

    
    
    Enter the name of the article you want to create with or without spaces. Input
    will be converted to all lower-case and cleaned. Results might not be what is
    expected, if non English alphanumeric characters are used.
    
        


    Name:  best-article


    Name is: best-article
    [jupytext] Reading /Users/tobias/all_code/projects/python_projects/portfolio_articles/_posts/2023-04-08-best-article/2023-04-08-best-article.md in format md
    [jupytext] Updating notebook metadata with '{"jupytext": {"formats": "ipynb,py:percent,md"}}'
    [jupytext] Updating /Users/tobias/all_code/projects/python_projects/portfolio_articles/_posts/2023-04-08-best-article/2023-04-08-best-article.ipynb
    [jupytext] Updating /Users/tobias/all_code/projects/python_projects/portfolio_articles/_posts/2023-04-08-best-article/2023-04-08-best-article.md
    [jupytext] Updating /Users/tobias/all_code/projects/python_projects/portfolio_articles/_posts/2023-04-08-best-article/2023-04-08-best-article.py
    
     ['2023-04-08-best-article.py', '2023-04-08-best-article.ipynb', '2023-04-08-best-article.md'] 
    

