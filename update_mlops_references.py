#!/usr/bin/env python3
"""
Script to update MLOps project references in portfolio website.
Updates GitHub URLs, removes sensitive file references, and adds disclaimers.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

# Define replacements
URL_REPLACEMENTS = [
    ("https://github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc", 
     "https://github.com/kletobias/advanced-mlops-demo"),
    ("github.com/kletobias/advanced-mlops-lifecycle-hydra-mlflow-optuna-dvc",
     "github.com/kletobias/advanced-mlops-demo"),
]

PATH_REPLACEMENTS = [
    ("/Users/tobias/.local/projects/portfolio_medical_drg_ny", "./project"),
    ("/Users/tobias/.local/share/mamba/envs/ny", "$PYTHON_ENV"),
]

SENSITIVE_FILE_REPLACEMENTS = [
    # Replace specific file references with generic descriptions
    ("dependencies/transformations/agg_severities.py", 
     "dependencies/transformations/[medical_transform_removed].py"),
    ("dependencies/transformations/drop_rare_drgs.py",
     "dependencies/transformations/[medical_transform_removed].py"),
    ("dependencies/transformations/ratio_drg_facility_vs_year.py",
     "dependencies/transformations/[medical_transform_removed].py"),
    ("dependencies/templates/generate_dvc_yaml_core.py",
     "scripts/orchestrate_dvc_flow.py"),
    ("dependencies/ingestion/download_and_save_data.py",
     "dependencies/ingestion/ingest_data.py"),
    # Update TRANSFORMATIONS registry references
    ("scripts/universal_step.py#L74-L147",
     "scripts/universal_step.py"),
]

# Patterns to redact
REDACTION_PATTERNS = [
    # Replace agg_severities mentions in logs
    (r"agg_severities", "[medical_transform]"),
    (r"drop_rare_drgs", "[medical_transform]"),
    (r"ratio_drg_facility_vs_year", "[medical_transform]"),
]

DISCLAIMER = """
> **Note**: This article references the academic demonstration version of the pipeline.  
> Some implementation details have been simplified or removed for IP protection.  
> Full implementation available under commercial license.
"""

def update_file(filepath: Path) -> bool:
    """Update a single file with all replacements."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply URL replacements
        for old, new in URL_REPLACEMENTS:
            content = content.replace(old, new)
        
        # Apply path replacements
        for old, new in PATH_REPLACEMENTS:
            content = content.replace(old, new)
        
        # Apply sensitive file replacements
        for old, new in SENSITIVE_FILE_REPLACEMENTS:
            content = content.replace(old, new)
        
        # Apply redaction patterns
        for pattern, replacement in REDACTION_PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        # Add disclaimer if not already present and file mentions MLOps
        if "MLOps" in content and "Note**: This article references the academic" not in content:
            # Find the first heading after front matter
            lines = content.split('\n')
            in_frontmatter = False
            insert_index = -1
            
            for i, line in enumerate(lines):
                if line.strip() == '---':
                    if not in_frontmatter:
                        in_frontmatter = True
                    else:
                        # End of frontmatter
                        # Look for first heading
                        for j in range(i+1, len(lines)):
                            if lines[j].startswith('#'):
                                insert_index = j + 1
                                break
                        break
            
            if insert_index > 0:
                lines.insert(insert_index, DISCLAIMER)
                content = '\n'.join(lines)
        
        # Only write if changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Process all relevant files in the portfolio website."""
    
    base_dir = Path("/Users/tobias/all_code/projects/portfolio-website-2022")
    
    # Define file patterns to process
    patterns = [
        "_posts/*mlops*.md",
        "_posts/*spotlight*.md",
        "_projects/*mlops*.md",
        "_projects/*modular*.md",
        "_projects/*transformation*.md",
        "_projects/*feature-engineering*.md",
        "_projects/*dvc*.md",
        "_projects/*jinja*.md",
        "_projects/*logging*.md",
        "_projects/*hyperparameter*.md",
    ]
    
    files_to_process = set()
    for pattern in patterns:
        files_to_process.update(base_dir.glob(pattern))
    
    print(f"Found {len(files_to_process)} files to process")
    
    updated_count = 0
    for filepath in sorted(files_to_process):
        print(f"Processing: {filepath.name}")
        if update_file(filepath):
            updated_count += 1
            print(f"  ✓ Updated")
        else:
            print(f"  - No changes needed")
    
    print(f"\nUpdated {updated_count} files")

if __name__ == "__main__":
    main()