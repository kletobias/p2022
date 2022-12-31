import re
import os
import shutil
import subprocess
import numpy as np

def make_toc(file:list,adir='/Users/tobias/all_code/projects/portfolio-website-2022/_projects/'):
    pat = re.compile(r'^##\s\w.+$')
    h2r = []
    h2c2nd = []
    h2c1st = []
    fname = [f for f in os.listdir(adir) if f in file]
    print(fname)
    for n in range(len(fname)):
        with open(f'{adir}/{fname[n]}','r') as f:
            content = f.readlines()
            f.close()
            for line in content:
                sc = pat.search(line)
                if sc != None:
                    h2r.append(sc[0])
        for i in h2r:
            mc2 = re.sub(r'^#{2}\s','',i)
            mc2 = re.sub(r'<br>',' ',mc2)
            h2c2nd.append(mc2)
            mc1 = re.sub(r'\w',lambda m: m[0].lower(),mc2)
            mc1 = re.sub(r'(\W)','-',mc1)
            mc1 = re.sub(r'-{2,}','-',mc1)
            mc1 = re.sub(r'_','-',mc1)
            h2c1st.append(mc1)
        cf = fname[n][:-3]
        with open(f'toc-output-{cf}.html','w+') as toc:
            toc.write('<d-contents>\n')
            toc.write('  <nav class="l-text figcaption">\n')
            toc.write('  <h3>Contents</h3>\n')
            for z in zip(h2c1st,h2c2nd):
                line = f'    <div class=\"no-math\"><a href=\"#{z[0]}\">{z[1]}</a></div>\n'
                toc.write(line)
            toc.write('  </nav>\n')
            toc.write('</d-contents>')

make_toc(file=['data_prep_2b.md','data_prep_1.md'])
