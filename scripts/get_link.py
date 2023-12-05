import os
import itertools
import re
from markdown2 import Markdown
import spacy

nlp = spacy.load("en_core_web_sm")
import en_core_web_sm

nlp = en_core_web_sm.load()


dir = "/Users/tobias/all_code/projects/portfolio-website-2022/_projects/"
outdir = "/Users/tobias/all_code/projects/portfolio-website-2022/references/"
files = os.listdir(dir)
host = "https://deep-learning-mastery.com/projects/"
# for f in files:
#     print(f)
#     print(f'is .md: {f[-3:] == ".md"}')


def cl(files, host):
    md_files = []
    link = {}
    for f in files:
        if f[-3:] == ".md":
            md_files.append(f[:-3])
            link[f[:-3]] = host + f"{f[:-3]}" + "/"
        else:
            continue
    linkf = []
    for k, v in link.items():
        linkf.append(f"[{k}]({v})")

    return linkf


mdf = cl(files, host)
print(mdf)


def get_title_description(dir, files):
    patt = re.compile("^title:\s['\"](.+)['\"]$", re.IGNORECASE)
    patd = re.compile("^description:\s['\"](.+)['\"]$", re.IGNORECASE)
    patc = re.compile("^category:\s\[['\"](.+)['\"]\]$", re.IGNORECASE)
    patta = re.compile("^tags:\s(\[.+\])$", re.IGNORECASE)
    patsub = re.compile("<br>")
    pat_html = re.compile(r"<[^>]+>|\\n", re.MULTILINE)
    md_files = []
    cv_text_all = []
    td = {}
    link = {}
    linkf = []
    for f in files:
        if f[-3:] == ".md":
            td[f[:-3]] = {}
            md_files.append(f[:-3])
            td[f[:-3]]["url"] = f"[Full Article]({host}{f[:-3]}/)"

            # open file and use spacy to get count of words
            with open(os.path.join(dir, f), "r") as a:
                article = a.readlines()
                doc = nlp(pat_html.sub("", str(article)))
                words = [
                    token.text
                    for token in doc
                    if token.is_stop != True and token.is_punct != True
                ]
                td[f[:-3]]["word_count"] = len(words)

            # open file and search for matches using the regex patterns
            with open(os.path.join(dir, f), "r") as ff:
                for line in itertools.islice(ff, 2, 8):
                    ttl = patt.search(line)
                    ddl = patd.search(line)
                    ccl = patc.search(line)
                    tal = patta.search(line)
                    if ttl != None:
                        ttls = patsub.sub(" ", ttl[1])
                        td[f[:-3]]["title"] = ttls
                    elif ddl != None:
                        ddls = patsub.sub(" ", ddl[1])
                        td[f[:-3]]["description"] = ddls
                    elif ccl != None:
                        ccls = patsub.sub(" ", ccl[1])
                        td[f[:-3]]["category"] = ccls
                    elif tal != None:
                        tals = patsub.sub(" ", tal[1])
                        td[f[:-3]]["tags"] = tals
                    else:
                        continue
    for k, v in link.items():
        linkf.append(f"[{k}]({v})")
    for k in sorted(td.keys()):
        print(td[k].items())
        cv_text = (
            f'<p><H3>{td[k]["title"]}</H3></p>'
            f'<p>**Description:** {td[k]["description"]}<br>'
            f'**Tags:** {td[k]["tags"]}<br>'
            f'**Category:** *{td[k]["category"]}* | **Word Count:** {td[k]["word_count"]} | **{td[k]["url"]}**</p><br>'
            f"<br><br>"
            f"\n"
        )
        md = Markdown()
        print(f"CONVERTED: {md.convert(cv_text)}")
        cv_text_all.append(md.convert(cv_text))

    with open(os.path.join(outdir, "cv_articles.md"), "w+") as f:
        for item in cv_text_all:
            f.write(item)
    return cv_text_all


td = get_title_description(dir, files)
for item in td:
    print()
    print(item)
    print()
