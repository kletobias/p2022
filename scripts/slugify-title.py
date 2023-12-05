from slugify import slugify
import re
import os
import itertools

indir = "/Users/tobias/all_code/projects/portfolio-website-2022/_projects"
files = os.listdir(indir)


def get_title_description(dir, files):
    patt = re.compile("^title:\s['\"](.+)['\"]$", re.IGNORECASE)
    md_files = {}
    cv_text_all = []
    td = {}
    link = {}
    linkf = []
    for f in files:
        if f[-3:] == ".md":
            td[f[:-3]] = {}
            md_files[f"{f[:-3]}"] = {"old": [], "new": []}
            md_files[f"{f[:-3]}"]["old"].append(f[["old"].append(f[:-3]) : -3])
            with open(os.path.join(dir, f), "r") as ff:
                for line in itertools.islice(ff, 2, 8):
                    ttl = patt.search(line)
                    if ttl != None:
                        md_files[f[:-3]]["new"].append(ttl[1])
                    else:
                        continue
    return md_files


md_files = get_title_description(indir, files)


def change_old(dir=indir, md_files=md_files, files=files):
    search_pat = []
    for key in md_files.keys():
        md_files[key]["slug"] = []
        slg = slugify(md_files[key]["new"][0])
        search_pat.append((md_files[key]["old"].value(), slg))
        md_files[key]["slug"].append(slg)
        with open(os.path.join(dir, md_files[key], ".md"), "w") as ff:
            for old in search_pat:
                for line in ff.readlines():
                    ttl = re.sub(f"({old})", line)
                    if ttl != None:
                        md_files[f[:-3]]["new"].append(ttl[1])
                    else:
                        continue
    print(md_files)


change_old()
# for k,v in link.items():
# linkf.append(f'[{k}]({v})')
# for k in sorted(td.keys()):
# print(td[k].items())
# cv_text = (f'<p><H3>{td[k]["title"]}</H3></p>'
#        f'<p>**Description:** {td[k]["description"]}<br>'
#        f'**Tags:** {td[k]["tags"]}<br>'
#        f'**Category:** *{td[k]["category"]}* | **Word Count:** {td[k]["word_count"]} | **{td[k]["url"]}**</p><br>'
#        f'<br><br>'
#        f'\n'
#        )
# md = Markdown()
# print(f'CONVERTED: {md.convert(cv_text)}')
# cv_text_all.append(md.convert(cv_text))

# with open(os.path.join(outdir,'cv_articles.md'),'w+') as f:
# for item in cv_text_all:
# f.write(item)
# return cv_text_all
