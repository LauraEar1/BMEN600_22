# 🔀 Git Guide for Our Project Group

**Repo:** https://github.com/LauraEar1/BMEN600_22.git

Git tracks changes to our files so we can all work at the same time without overwriting each other. We're all collaborators on this repo, so **no forking is needed**: everyone clones the same repo and pushes their own branch to it. This guide uses **GitHub Desktop** (recommended) or the **GitHub website**.

---

## 1. Key terms

| Term | Meaning |
|---|---|
| **Repository (repo)** | The project folder tracked by git |
| **Commit** | A saved snapshot of your changes, with a short message |
| **Branch** | Your own line of work, so your changes don't affect others until you're ready |
| **Fetch / Pull** | Download the latest changes from GitHub |
| **Push** | Upload your commits to GitHub |
| **Merge** | Combine one branch into another |
| **Pull Request (PR)** | A request to merge your branch into `main`, so a teammate can review it first |

---

## 2. One-time setup

1. **Accept the collaborator invite.** Check your email (or github.com/notifications) for the invitation to `BMEN600_22` and click **Accept**. You can't push until you do.
2. Install **GitHub Desktop** from https://desktop.github.com and sign in with your GitHub account.
3. **File → Clone repository → URL** tab, paste `https://github.com/LauraEar1/BMEN600_22.git`, choose a local folder, and click **Clone**.

---

## 3. Option A: GitHub Desktop (recommended)

### Everyday workflow
1. **Get the latest version first.** Make sure **Current Branch** is `main`, then click **Fetch origin**. If it changes to **Pull origin**, click it.
2. **Create your own branch.** **Current Branch → New Branch**, name it `yourname-short-description` (e.g. `laura-data-cleaning`), and click **Create Branch**.
3. **Do your work** in Jupyter, VS Code, etc. and save your files.
4. **Review your changes** in the **Changes** tab on the left. Tick only the files you want to include.
5. **Commit.** Write a clear summary at the bottom left (e.g. *"Add data cleaning for sleep dataset"*) and click **Commit to your-branch**.
6. **Push.** Click **Publish branch** the first time, or **Push origin** afterwards.
7. **Open a Pull Request.** Click **Create Pull Request**. This opens GitHub in your browser. Describe what you did and ask a teammate to review.
8. **Merge.** Once it's approved, click **Merge pull request** on GitHub, then **Delete branch** when it offers.
9. **Update your computer.** Switch back to `main` in GitHub Desktop and click **Fetch origin → Pull origin**.

### Keeping your branch up to date
While working, regularly click **Branch → Update from main** so your branch picks up your teammates' latest changes.

### Switching branches
Click **Current Branch** and pick one. If you have uncommitted changes, Desktop will ask whether to **stash** them (shelve for later) or bring them along.

---

## 4. Option B: GitHub website only

Good for small edits (README, text files). For notebooks and code you'll run, GitHub Desktop is easier.

1. Open the repo and click the **branch dropdown** (says `main`). Type a new branch name and click **Create branch**.
2. To **edit a file**: open it, click the ✏️ pencil icon, make your changes, then click **Commit changes**. Choose your branch (not `main`) and write a short message.
3. To **add files**: **Add file → Upload files**, drag them in, and commit to your branch.
4. Click **Compare & pull request** (or **Pull requests** tab → **New pull request**). Describe your changes and request a review.
5. After a teammate approves, click **Merge pull request → Confirm merge**.

---

## 5. Merge conflicts (don't panic!)

A conflict happens when two people edit the same lines of the same file.

**In GitHub Desktop:** it lists the conflicting files. Click **Open in Visual Studio Code** (or your editor) and find sections like this:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> main
```

Decide what the final text should be, delete the three marker lines, save, then return to Desktop and click **Commit merge**.

**On GitHub:** if a PR says *"This branch has conflicts"*, click **Resolve conflicts**, fix the markers, click **Mark as resolved**, then **Commit merge**.

If you're unsure which version to keep, ask the group first.

---

## 6. Group rules

- ✅ **Never commit directly to `main`.** As collaborators we *can* push to it, and GitHub won't stop us, so this is a team agreement. Always use a branch and a pull request.
- ✅ **Fetch/pull before you start working**, every time.
- ✅ **Commit small and often**, with clear messages ("Fix windowing bug", not "stuff").
- ✅ **Split up files.** Avoid two people editing the same notebook at once.
- ✅ **Don't merge your own PR** without at least one teammate looking at it.
- ⚠️ **Notebooks (`.ipynb`) conflict badly.** Before committing, clear outputs (*Kernel → Restart & Clear All Outputs*).
- ⚠️ **Don't commit large datasets, passwords, or API keys.** Right-click a file in the **Changes** tab and choose **Ignore file**.
- ⚠️ **Tell the group before deleting or renaming shared files.**

---

## 7. Quick cheat sheet

**Pull `main` → New Branch → work → Commit → Push → Create Pull Request → teammate reviews → Merge → Pull `main` again**
