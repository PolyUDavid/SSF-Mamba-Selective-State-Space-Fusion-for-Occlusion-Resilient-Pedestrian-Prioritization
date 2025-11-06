# GitHub Setup and Push Guide

Complete guide to upload your P-SAFE code to GitHub.

**Author:** David KO  
**Date:** November 2025

---

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ A GitHub account ([Create one here](https://github.com/join) if needed)
- ✅ Git installed on your computer
- ✅ Terminal/Command Prompt access

### Check if Git is Installed

Open terminal and run:
```bash
git --version
```

If you see a version number (e.g., `git version 2.39.0`), you're good to go!

If not installed:
- **macOS**: `brew install git` or download from [git-scm.com](https://git-scm.com/)
- **Windows**: Download from [git-scm.com](https://git-scm.com/)
- **Linux**: `sudo apt-get install git` (Ubuntu/Debian)

---

## 🚀 Step-by-Step Guide

### Step 1: Navigate to Your Project Folder

Open terminal and go to your project directory:

```bash
cd "/Volumes/Shared U/PI_BREPSC/CVPR_Submission"
```

### Step 2: Initialize Git Repository

Run this command to initialize a new Git repository:

```bash
git init
```

You should see:
```
Initialized empty Git repository in /Volumes/Shared U/PI_BREPSC/CVPR_Submission/.git/
```

### Step 3: Configure Git (First Time Only)

If this is your first time using Git, configure your name and email:

```bash
git config --global user.name "David KO"
git config --global user.email "your-email@example.com"
```

### Step 4: Add All Files to Git

Add all your project files:

```bash
git add .
```

This stages all files for commit. The `.` means "all files in current directory".

### Step 5: Create Your First Commit

Commit the files with a message:

```bash
git commit -m "Initial commit: P-SAFE complete implementation"
```

You should see output like:
```
[main (root-commit) abc1234] Initial commit: P-SAFE complete implementation
 30 files changed, 8500 insertions(+)
 create mode 100644 README.md
 ...
```

### Step 6: Create a New GitHub Repository

1. Go to [GitHub](https://github.com) and log in
2. Click the **"+"** icon in the top right
3. Select **"New repository"**

**Repository Settings:**
- **Repository name:** `P-SAFE` (or your preferred name)
- **Description:** `P-SAFE: Multi-Modal AI Framework for Pedestrian Safety`
- **Visibility:** 
  - Choose **Public** if you want it openly accessible
  - Choose **Private** if you want to control access (you can add reviewers later)
- **DO NOT** check "Initialize this repository with a README" (we already have one)
- Click **"Create repository"**

### Step 7: Link Your Local Repository to GitHub

After creating the repo, GitHub will show you commands. Copy your repository URL (it looks like: `https://github.com/your-username/P-SAFE.git`)

Run these commands (replace with your actual URL):

```bash
# Add the remote repository
git remote add origin https://github.com/your-username/P-SAFE.git

# Rename the branch to 'main' (if needed)
git branch -M main
```

### Step 8: Push Your Code to GitHub

Push your code to GitHub:

```bash
git push -u origin main
```

**If prompted for username and password:**
- **Username:** Your GitHub username
- **Password:** Use a **Personal Access Token** (not your GitHub password)

**How to create a Personal Access Token:**
1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "CVPR Submission")
4. Select scope: Check **"repo"** (full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as your password when pushing

After successful push, you should see:
```
Enumerating objects: 35, done.
Counting objects: 100% (35/35), done.
...
To https://github.com/your-username/P-SAFE.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

### Step 9: Verify on GitHub

1. Go to your repository URL: `https://github.com/your-username/P-SAFE`
2. You should see all your files!
3. Check that README.md displays correctly

---

## 🔒 If You Made the Repository Private

If your repository is private and you want to give access to CVPR reviewers:

### Option 1: Add Reviewers as Collaborators

1. Go to your repository
2. Click **Settings** → **Collaborators and teams**
3. Click **Add people**
4. Enter the reviewer's GitHub username or email
5. They'll receive an invitation

### Option 2: Make Repository Public Temporarily

During review period:
1. Go to repository **Settings**
2. Scroll to bottom "Danger Zone"
3. Click **Change visibility** → **Make public**

After review, you can make it private again.

---

## 📝 Share Your Repository

Your repository is now publicly accessible (if you chose Public visibility):

Repository URL:
```
https://github.com/your-username/P-SAFE
```

Description:
```
Code Repository: Complete model implementations, baseline comparisons,
and experimental results for P-SAFE framework.
```

---

## 🔄 Making Updates After Initial Push

If you need to update your code later:

```bash
# 1. Make your changes to files

# 2. Check what changed
git status

# 3. Add the changes
git add .

# 4. Commit with a message
git commit -m "Update: description of changes"

# 5. Push to GitHub
git push origin main
```

---

## 🎨 Optional: Make Your README Look Better

### Add Topics/Tags

On your GitHub repository page:
1. Click the ⚙️ (gear icon) next to "About"
2. Add topics: `pedestrian-safety`, `multi-modal-ai`, `traffic-control`, `computer-vision`, `mamba`
3. Save changes

---

## ❓ Troubleshooting

### Problem: "Permission denied (publickey)"

**Solution:** Use HTTPS instead of SSH for simpler setup.
Make sure your remote URL uses HTTPS:
```bash
git remote -v
# Should show: https://github.com/...

# If it shows git@github.com, change it:
git remote set-url origin https://github.com/your-username/P-SAFE.git
```

### Problem: "Updates were rejected because the remote contains work"

**Solution:** Pull first, then push:
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Problem: "Large files" or "Push rejected"

GitHub has a 100MB file size limit. Check file sizes:
```bash
find . -type f -size +50M
```

If you have large files, add them to `.gitignore`:
```bash
echo "large_file.bin" >> .gitignore
git rm --cached large_file.bin
git commit -m "Remove large file"
git push origin main
```

### Problem: Git is too slow

If you have many files in the parent directory:
```bash
# Make sure you're only tracking the CVPR_Submission folder
cd "/Volumes/Shared U/PI_BREPSC/CVPR_Submission"
# NOT the parent PI_BREPSC folder
```

---

## 📧 Need Help?

If you encounter any issues:
- Check [GitHub's documentation](https://docs.github.com/)
- Open a GitHub issue in your repository

---

## ✅ Final Checklist

Before sharing your repository, verify:

- [ ] All files pushed to GitHub successfully
- [ ] README.md displays correctly on GitHub
- [ ] Repository is accessible (public or reviewers invited)
- [ ] No sensitive data (training data, passwords, etc.)
- [ ] All baseline models included
- [ ] All result JSON files included
- [ ] Example inference script works
- [ ] requirements.txt is complete
- [ ] LICENSE file is present
- [ ] Author name is correct: David KO

---

## 🎉 You're Done!

Your code is now on GitHub and ready to share!

**Your repository URL:** `https://github.com/your-username/P-SAFE`

Good luck with your research! 🚀

