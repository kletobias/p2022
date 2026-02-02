---
layout: distill  
title: 'Accelerating Tar Archive Creation with Parallel Execution'  
date: 2023-09-05  
description: 'Unveiling a Python wrapper script that leverages multi-threading for swift archive creation. The script employs pgiz to perform parallel execution, speeding up the tar archive creation process.'  
tags: ['tar', 'archive', 'subprocess', 'pigz', 'multi-threading']  
category: 'scripting'  
comments: true
---

<br>

# Accelerating Tar Archive Creation with Parallel Execution

## Summary

Ever had to create a `.tar.gz` archive for a large directory and found yourself twiddling your thumbs as you waited for the process to complete? My Python wrapper script could be the answer you're looking for. Designed for execution in the shell, this script facilitates the creation of `.tar.gz` archives from any directory. It utilizes `pgiz` for parallel execution, which makes the process considerably faster. 

In this blog post, we'll cover how to install required libraries, use the script, and delve into the Python code that powers it all.

## Prerequisites

Before we get started, make sure the following Python libraries are installed and available:

- `os`
- `argparse`
- `subprocess`

Additionally, `pigz` needs to be installed on your system. On a Mac, you can install it using Homebrew with the following command:

```bash
brew install pigz
```

## Installation

Simply copy the Python script from the bottom of this post into a file named `parallelized_tar.py` or any other name you prefer.

## How to Use

Execute the script in the shell using the following command:

```python
python script_name.py /path/to/input/folder /path/to/output/folder --threads 10
```

In the command, `script_name.py` serves as a stand-in for whatever name you've assigned to the saved script. Substitute `/path/to/input/folder` with the specific directory you wish to compress into an archive, and replace `/path/to/output/folder` with the target directory where you want the resulting `.tar.gz` archive to reside. By default, the archive will inherit its name from the last segment of the provided input folder path.

## Command-Line Arguments

The script uses command-line arguments for better flexibility:

- `input_folder`: The path to the folder you want to archive.
- `output_folder`: The path where you'd like the `.tar.gz` archive to be saved.
- `--threads`: Optional. The number of threads that `pigz` will use for compression. The default is 10 threads.

## Code Explanation

Here's a quick rundown of what the code does:

```python
import os
import argparse
import subprocess

def create_tar_gz(input_folder, output_folder, num_threads):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Use the last part of the input_folder path as the archive name
    archive_name = os.path.basename(os.path.normpath(input_folder)) + '.tar.gz'
    output_path = os.path.join(output_folder, archive_name)

    try:
        # Use subprocess to run the tar and pigz commands in parallel
        with open(output_path, 'wb') as f_out:
            p1 = subprocess.Popen(["tar", "cf", "-", input_folder], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["pigz", "-p", str(num_threads)], stdin=p1.stdout, stdout=f_out)
            p2.communicate()
        
        print(f"Archive created successfully at {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a tar.gz archive from an input folder.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder.')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads to use.')
    
    args = parser.parse_args()
    
    create_tar_gz(args.input_folder, args.output_folder, args.threads)
```

## Potential Use-Cases

- **Backup**: If you have a directory of crucial files that you'd like to backup, this script makes the process swift and efficient.
- **Data Transfer**: Archiving directories can simplify the process of transferring multiple files between servers or systems.
- **Storage Optimization**: Compressed archives take up less disk space, allowing you to store more data.

## Troubleshooting

If you encounter any errors while using the script, the most common issues usually relate to:

- Incorrect paths: Make sure the directory paths you're providing exist.
- Permissions: Ensure you have read and write permissions for the respective directories.
- Missing dependencies: Double-check that `pigz` and the required Python libraries are installed.

## Conclusion

In the fast-paced world of technology, time is of the essence. With this Python wrapper script that facilitates parallelized tar archive creation, you can optimize your archiving operations and save valuable time. Whether you're backing up important data or transferring files, this tool makes the process smooth, swift, and efficient.

Feel free to download the script, customize it as per your requirements, and let us know your thoughts in the comments section below.

Happy archiving!

---

**© Tobias Klein 2023 · All rights reserved**<br>
**LinkedIn: https://www.linkedin.com/in/deep-learning-mastery/**
