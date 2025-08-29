#!/bin/bash

# Script to convert old MLOps repository to a redirect
# Run this in the old repository directory

echo "🔄 Setting up repository redirect..."

# Create a backup branch first
echo "📦 Creating backup branch..."
git checkout -b backup-before-redirect-$(date +%Y%m%d)
git push origin backup-before-redirect-$(date +%Y%m%d)

# Switch to main branch
echo "🔀 Switching to main branch..."
git checkout main

# Remove all files except .git
echo "🗑️  Removing old files..."
find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} \;

# Copy the redirect README
echo "📝 Creating redirect README..."
cp ../portfolio-website-2022/README_REDIRECT_FOR_OLD_REPO.md ./README.md

# Create a simple .gitignore
echo "📄 Creating .gitignore..."
cat > .gitignore << 'EOF'
.DS_Store
*.log
.env
EOF

# Add redirect notice in multiple formats for better SEO
echo "🔍 Creating additional redirect files..."

# Create index.html for GitHub Pages (if enabled)
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Repository Moved - Redirecting...</title>
    <meta http-equiv="refresh" content="5; url=https://github.com/kletobias/advanced-mlops-demo">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            max-width: 600px;
            margin: 100px auto;
            padding: 20px;
            text-align: center;
        }
        h1 { color: #24292e; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .button {
            display: inline-block;
            padding: 12px 24px;
            background: #0366d6;
            color: white;
            border-radius: 6px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>📦 Repository Has Moved</h1>
    <p>This repository has been relocated to:</p>
    <h2><a href="https://github.com/kletobias/advanced-mlops-demo">github.com/kletobias/advanced-mlops-demo</a></h2>
    <p>You will be redirected in 5 seconds...</p>
    <a href="https://github.com/kletobias/advanced-mlops-demo" class="button">Go to New Repository Now</a>
</body>
</html>
EOF

# Create REDIRECT.md for clarity
cat > REDIRECT.md << 'EOF'
# ➡️ NEW LOCATION: https://github.com/kletobias/advanced-mlops-demo

This repository has moved. Please update your bookmarks.
EOF

# Create a GitHub issue template to notify about the move
mkdir -p .github/ISSUE_TEMPLATE
cat > .github/ISSUE_TEMPLATE/config.yml << 'EOF'
blank_issues_enabled: false
contact_links:
  - name: New Repository Location
    url: https://github.com/kletobias/advanced-mlops-demo
    about: This project has moved. Please visit the new repository.
  - name: Commercial Licensing
    url: mailto:your-email@example.com
    about: Interested in the full implementation? Contact for commercial licensing.
EOF

# Commit all changes
echo "💾 Committing redirect setup..."
git add -A
git commit -m "feat: convert repository to redirect

- Repository moved to github.com/kletobias/advanced-mlops-demo
- Added redirect README with project information
- Created HTML redirect page for GitHub Pages
- Set up issue template to guide users

The new repository contains a demo version for academic presentation.
Full implementation available for commercial licensing."

echo "📤 Ready to push. Run 'git push origin main' when ready."
echo ""
echo "✅ Redirect setup complete!"
echo ""
echo "Next steps:"
echo "1. Review the changes: git status"
echo "2. Push to GitHub: git push origin main"
echo "3. Enable GitHub Pages (optional) from Settings > Pages"
echo "4. Update repository description on GitHub to:"
echo "   '➡️ MOVED to github.com/kletobias/advanced-mlops-demo'"
echo "5. Consider archiving the repository (Settings > General > Archive)"