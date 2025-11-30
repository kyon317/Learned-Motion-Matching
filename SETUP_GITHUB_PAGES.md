# GitHub Pages Setup Guide

This guide will help you set up GitHub Pages for hosting the web demo.

## Prerequisites

1. **Install Git LFS** (if not already installed):
   ```bash
   git lfs install
   ```

2. **Initialize Git LFS for binary files**:
   ```bash
   git add .gitattributes
   git add resources/*.bin
   git add resources/difffusion/*.bin
   ```

## Steps to Enable GitHub Pages

1. **Commit the setup files**:
   ```bash
   git add .gitattributes
   git add .gitignore
   git add .github/workflows/build-and-deploy.yml
   git add README.md
   git commit -m "Set up GitHub Pages with Git LFS for web demo"
   git push
   ```

2. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Pages**
   - Under **Source**, select:
     - **Branch**: `gh-pages`
     - **Folder**: `/ (root)`
   - Click **Save**

3. **Wait for the workflow to complete**:
   - Go to **Actions** tab in your repository
   - The workflow will automatically build and deploy when you push to `main`
   - You can also manually trigger it from the Actions tab

4. **Update the README**:
   - After the first deployment, update the Live Demo link in `README.md`
   - Replace `yourusername` with your actual GitHub username
   - The URL format will be: `https://yourusername.github.io/Motion-Matching/controller.html`

## Manual Build (Optional)

If you want to test the build locally before pushing:

```bash
# Set up emscripten
source emsdk_env.sh  # or emsdk_env.bat on Windows

# Build for web
make PLATFORM=PLATFORM_WEB

# Test locally
python wasm-server.py
# Then visit http://localhost:8080/controller.html
```

## Troubleshooting

- **Large file errors**: Make sure Git LFS is installed and `.gitattributes` is committed
- **Build failures**: Check the Actions tab for error logs
- **Models not loading**: Ensure `resources/` folder is copied to `docs/` in the workflow
- **404 on GitHub Pages**: Wait a few minutes after deployment, or check the Pages settings

## Notes

- The workflow automatically builds on every push to `main`
- Model files are stored using Git LFS to handle large files
- The web demo includes all pre-trained models in the `resources/` folder

