import os
import re

# The pattern to search for
pattern = r"(/Users/tobias/[^/]+/.+$)"

# A regular expression object for the pattern
regex = re.compile(pattern)

# The string to replace the pattern with
replacement = "your_replacement_string_here"

# Walk through all directories and subdirectories
for root, dirs, files in os.walk("."):
    # Check if any of the files are markdown files
    for file in files:
        if file.endswith(".md"):
            # Construct the full path to the file
            filepath = os.path.join(root, file)
            # Open the file and read its contents
            with open(filepath, "r") as f:
                contents = f.read()
            # Substitute the pattern with the replacement string
            new_contents = regex.sub(replacement, contents)
            # Write the new contents back to the file
            with open(filepath, "w") as f:
                f.write(new_contents)
