---
name: version-bump
description: "Bump the libnpy library version for release. Use when: updating the version number, preparing a release, changing the version to a specific value, writing a CHANGELOG entry from recent commits."
argument-hint: "Target version, e.g. 2.2.0"
---

# Version Bump

Bump the libnpy version across all files and generate a CHANGELOG entry from recent commits.

## When to Use

- The maintainer asks to bump / change / update the version (e.g. "Bump the version to 2.2.0")
- Preparing a new release of the library

## Procedure

### 1. Parse the requested version

Extract the target version string (`MAJOR.MINOR.PATCH`) from the user's request.
Read the current version from the `VERSION` file in the repository root.
Abort if the requested version equals the current version.

### 2. Update every version location

Apply the new version to **all** of the following files. No other files contain the version.

| File | What to change |
|------|---------------|
| `VERSION` | Replace the entire file content with the new version string (no trailing newline beyond what exists). |
| `vcpkg.json` (repo root) | Update the `"version"` field value. |
| `ports/libnpy/vcpkg.json` | Update the `"version"` field value. |
| `include/npy/npy.h` | Update **all four** preprocessor defines: `NPY_VERSION_MAJOR`, `NPY_VERSION_MINOR`, `NPY_VERSION_PATCH`, and `NPY_VERSION_STRING`. |

Do **not** touch `CMakeLists.txt` — it reads the version dynamically from the `VERSION` file.

### 3. Generate the CHANGELOG entry

Run `git log` to collect commits on the current branch since the latest release tag.
Use a command like:

```bash
git log --oneline --no-decorate $(git describe --tags --abbrev=0)..HEAD
```

If no tags exist, fall back to:

```bash
git log --oneline --no-decorate -20
```

Compose a new entry and **prepend** it to `CHANGELOG.md` immediately after the `# Changelog` heading, using this exact format:

```markdown
## [YYYY-MM-DD - Version X.Y.Z](https://github.com/matajoh/libnpy/releases/tag/vX.Y.Z)

One-sentence summary of the release theme.

Improvements:
- Item 1
- Item 2

Bugfixes:
- Item 1
```

Rules for the entry:
- Use today's date in `YYYY-MM-DD` format.
- Categorise changes under **Improvements** and/or **Bugfixes** (omit a category if empty).
- Write entries in simple past tense, concise, one line each.
- Do not include merge commits, version-bump commits, or trivial CI-only changes unless they are user-facing.
- Match the tone and style of the existing entries in `CHANGELOG.md`.

### 4. Update RELEASE_NOTES

Replace the entire content of the `RELEASE_NOTES` file with the body of the new changelog entry (everything after the `## [...]` heading line). This file is plain text with no markdown heading.

### 5. Verify

After all edits, run a search for the **old** version string across the repository to confirm no stale references remain. Report the result to the maintainer.

### 6. Present changes for review

List every file modified and summarise the changes so the maintainer can review before committing.
