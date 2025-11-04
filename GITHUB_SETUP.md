# 🚀 GitHub Setup Guide

## Step 1: Install Git (if not installed)

If you see "git is not recognized", install Git first:
- Download from: https://git-scm.com/download/win
- Install and restart your terminal

## Step 2: Create GitHub Repository

1. Go to https://github.com
2. Click "New repository" (or "+" → "New repository")
3. Name it (e.g., "deepfake-detection-system")
4. Choose Public or Private
5. **DO NOT** initialize with README, .gitignore, or license
6. Click "Create repository"

## Step 3: Initialize Git and Push

Open terminal in your project directory and run:

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Deepfake Detection System with Advanced Ensemble"

# Add remote repository (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 4: Verify

Check your GitHub repository - all files should be there!

## Alternative: Using GitHub Desktop

1. Install GitHub Desktop: https://desktop.github.com/
2. Open GitHub Desktop
3. File → Add Local Repository → Select your project folder
4. Click "Publish repository" to GitHub
5. Choose name and visibility
6. Click "Publish repository"

## What's Included in .gitignore

✅ **Excluded** (not uploaded):
- `venv/` - Virtual environment
- `__pycache__/` - Python cache
- `*.pth`, `*.pt` - Large model files
- `data/faceforensics_data/` - Large video datasets
- `preprocessing/frames/`, `preprocessing/faces/` - Processed data
- `*.log`, `*.tmp`, `*.bak` - Temporary files

✅ **Included** (uploaded):
- All source code (`.py` files)
- All documentation (`.md` files)
- Configuration files
- Requirements files
- Web templates
- Project structure

## Quick Commands Reference

```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull latest changes
git pull
```

---

**Note**: Large model files (`.pth`, `.pt`) are excluded by `.gitignore`. 
If you want to share models, use Git LFS (Large File Storage) or upload separately.

