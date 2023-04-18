import os
import re

dirs= {
    "base": "/Users/tobias/all_code/projects/portfolio-website-2022/",
    "projects": "_projects",
    "posts": "_posts"
    }

# The pattern to search for
pattern = r'(["\']/Users/tobias/(?:[^/]+/)*)([^/]+/?)(?:["\']|$)'

# A regular expression object for the pattern
regex = re.compile(pattern,re.I)

# The string to replace the pattern with
replacement_base = "/path/to/destination/"

for i in [os.path.join(dirs['base'],dirs[k]) for k in dirs.keys() if k != "base"]:
    os.chdir(i)
    print(f'os.getcwd() \n {os.getcwd()} \n')
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk("."):
        # Check if any of the files are markdown files
        for file in files:
                if file.endswith(".md"):
                    # Construct the full path to the file
                    filepath = os.path.join(root, file)
                    print(f'filepath \n {filepath} \n')
                    # Open the file and read its contents
                    with open(filepath, "r") as f:
                        contents = f.read()
                    # Split the contents into lines
                    lines = contents.split('\n')
                    # Substitute the pattern with the replacement string for each line
                    new_lines = []
                    for line in lines:
                        new_line = line
                        for match in regex.finditer(line):
                            excluded_part = match.group(1)
                            print(f'excluded_part \n {excluded_part} \n')
                            replacement = replacement_base + excluded_part
                            print(f'replacement \n {replacement} \n')
                            new_line = new_line.replace(match.group(0), replacement)
                        new_lines.append(new_line)
                    # Join the lines back into a single string
                    new_contents = '\n'.join(new_lines)
                    # Write the new contents back to the file
                    with open(filepath, "w") as f:
                        f.write(new_contents)
