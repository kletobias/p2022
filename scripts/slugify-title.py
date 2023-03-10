from slugify import slugify
import re
import os

indir='/Users/tobias/all_code/projects/portfolio-website-2022/_projects'
files = os.listdir(indir)




def get_title_description(dir,files):
    patt = re.compile('^title:\s[\'\"](.+)[\'\"]$',re.IGNORECASE)
    md_files = {}
    cv_text_all = []
    td = {}
    link = {}
    linkf = []
    for f in files:
        if f[-3:] == '.md':
            td[f[:-3]] = {}
            md_files[f'{f[:-3]}']['old'].append(f[:-3])
            # td[f[:-3]]['url'] = f'[Full Article]({host}{f[:-3]}/)'
            # with open(os.path.join(dir,f),'r') as a:
            #     article = a.readlines()
            #     doc = nlp(pat_html.sub('',str(article)))
            #     words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
            #     td[f[:-3]]['word_count'] = len(words)
            # with open(os.path.join(dir,f),'r') as ff:
            #     for line in itertools.islice(ff,2,8):
            #         ttl = patt.search(line)
            #         if ttl != None:
            #             ttls = patsub.sub(' ',ttl[1])
            #             td[f[:-3]]["title"] = ttls
            #         else:
            #             continue
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

td = get_title_description(dir,files)
for item in td:
    print()
    print(item)
    print()
