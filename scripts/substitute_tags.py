import os
import re
import itertools


dir = '/Users/tobias/all_code/projects/portfolio-website-2022/_projects'

# pat = re.compile(r'(?:tags:\s\[[^ ]+)(?:.*(\w\s\w)[^\']+)+')
pat = re.compile(r'(?:tags:\s)\[([^]]+)\]$')
pat_sub = re.compile(r'\s')
# pat = re.compile(r'(?:^tags:[^ ]+)(,\s+)')
# pat = re.compile(r'(?:^tags:[^ ](,\s+))')

files = [f for f in os.listdir(dir) if f[-3:] == '.md']

def find_wsc(files,pat):
    all_found = []
    all_found_ws = []
    all_subbed = []
    for f in files:
        with open(os.path.join(dir,f),'r') as ff:
            for n,l in zip(range(6,7),itertools.islice(ff,6,7)):
                # print(n,l)
                found = pat.search(l)
                if found != None:
                    # print(f'found is: {found[0]}')
                    # print(f'found[1] is: {found[1]}')
                    # print(f'found split: {found[1].split(",")}')
                    foundsplit = [re.sub("'",'',f.strip()) for f in found[1].split(",")]
                    foundsplitws = [f for f in foundsplit if re.search(r'\s',str(f))]
                    subbed = [pat_sub.sub('-',f) for f in foundsplitws]
                    # print(f'subbed {subbed}')
                    # print(f'foundsplit: {foundsplit}')
                    # print(f'foundsplitws: {foundsplitws}')
                    all_found.extend(foundsplit)
                    all_found_ws.extend(foundsplitws)
                    all_subbed.extend(subbed)
            ff.close() 
    # print(set(all_found))
    # print(set(all_found_ws))
    subd = dict(zip(sorted(set(all_found_ws)),sorted(set(all_subbed))))
    for aa in zip(sorted([*set(all_found_ws)]),sorted([*set(all_subbed)])):

        assert pat_sub.sub('-',str(aa[0])) == aa[1]
    for f in files:
        with open(os.path.join(dir,f),'r') as ff:
            # for l in itertools.islice(ff,6,7):
                file_data = ff.readlines()
                for i in subd.keys():
                    # print(str(i))
                    old = str(i)
                    new = str(subd[i])
                    file_data[6] = file_data[6].replace(old,new)
                    patt = re.search(r'tags:\s\[(.+\w[-]\w.+)+[\n\r]',file_data[6])
                    if patt != None:
                        print(f'tags: {patt[0]}')
        with open(os.path.join(dir,f),'w') as ff:
            # for l in itertools.islice(ff,6,7):
            ff.writelines(file_data)



#    print(subd)
    return set(all_found)

find_wsc(files,pat)
