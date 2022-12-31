import re
import os
import shutil
import subprocess
import numpy as np
import time
import glob

# Use True for Tobias machine, False for Ashish machine.
tobias = True
if tobias == True:
    projects_dir='/Users/tobias/all_code/projects/portfolio-website-2022/_projects/'
else:
    projects_dir = 'INSERT HERE: ashish-absolute-path-to_projects-dir'

def make_toc(remove_previous=False, file=None,adir=projects_dir,all=False):
    if remove_previous == True:
        file_list=glob.glob(f'{adir[:-10]}toc_script/*.html')
        for h in file_list:
            os.remove(h)
    pat = re.compile(r'^##\s\w.+$')
    pat_ext = re.compile(r'\.md$',re.IGNORECASE)
    dt = time.strftime("%Y%m%d-%H")
    if all == True:
        fname = [f for f in os.listdir(adir) if pat_ext.search(f)]
    else:
        fname = [f for f in os.listdir(adir) if f in file]
    print(f'Selected files, that will have TOC created for: {fname}')
    for n in range(len(fname)):
        with open(f'{adir}{fname[n]}','r') as f:
            h2c2nd = []
            h2c1st = []
            h2r = []
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
            #mc1 = re.sub(r'([^.:a-z0-9&_])','-',mc1)
            mc1 = re.sub(r'(\s+)','-',mc1)
            mc1 = re.sub(r'([&.:!;@()/\\])','',mc1)
#            mc1 = re.sub(r'-{2,}','-',mc1)
            mc1 = re.sub(r'(\d)\.(\d)','\1\2',mc1)
#            mc1 = re.sub(r'_','-',mc1)
            h2c1st.append(mc1)
        cf = fname[n][:-3]
        with open(f'{adir[:-10]}toc_script/{dt}-toc-output-{cf}.html','w+') as toc:
            toc.write('<d-contents>\n')
            toc.write('  <nav class="l-text figcaption">\n')
            toc.write('  <h3>Contents</h3>\n')
            for z in zip(h2c1st,h2c2nd):
                line = f'    <div class=\"no-math\"><a href=\"#{z[0]}\">{z[1]}</a></div>\n'
                toc.write(line)
            toc.write('  </nav>\n')
            toc.write('</d-contents>')
#            toc.close()

make_toc(all=True,remove_previous=True)
