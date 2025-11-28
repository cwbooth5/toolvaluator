# Release Process

This document describes how to create a new release of toolvaluator.

## Overview

The release process is automated using GitHub Actions. The workflow will:

1. Run ruff checks to ensure code quality
2. Run all unit tests to ensure everything works
3. Bump the version using semantic versioning
4. Build the package (wheel and source distribution)
5. Create a GitHub release with the built artifacts

## Prerequisites

- You must have write access to the repository
- All changes should be merged to `main` branch
- All tests should be passing

## Creating a Release

### Step 1: Navigate to GitHub Actions

1. Go to the repository on GitHub
2. Click on the "Actions" tab
3. Select the "Release" workflow from the left sidebar

### Step 2: Run the Workflow

1. Click the "Run workflow" button
2. Select the version bump type:
   - **patch**: Bug fixes and minor changes (0.1.0 → 0.1.1)
   - **minor**: New features, backward compatible (0.1.0 → 0.2.0)
   - **major**: Breaking changes (0.1.0 → 1.0.0)
3. Click "Run workflow"

### Step 3: Monitor the Workflow

The workflow will:
- Check out the code
- Set up Python 3.12
- Install dependencies
- Run `ruff check` (will fail if code quality issues exist)
- Run `pytest` (will fail if tests don't pass)
- Bump the version in `src/toolvaluator/__init__.py`
- Commit and push the version bump
- Build the package
- Create a GitHub release with tag `v{version}`
- Upload wheel and source distribution to the release

### Step 4: Verify the Release

1. Check the "Releases" section of the repository
2. Verify the new release appears with the correct version
3. Download and test the artifacts if needed

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **Major version** (X.0.0): Incompatible API changes
- **Minor version** (0.X.0): Add functionality in a backward compatible manner
- **Patch version** (0.0.X): Backward compatible bug fixes

## What Gets Released

Each release includes:

1. **Source Distribution** (`toolvaluator-{version}.tar.gz`):
   - Full source code
   - Tests
   - README and other documentation

2. **Wheel** (`toolvaluator-{version}-py3-none-any.whl`):
   - Pre-built package
   - Faster installation

3. **Git Tag** (`v{version}`):
   - Points to the commit with the version bump

## Troubleshooting

### Workflow fails on ruff check

Fix code quality issues:
```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Workflow fails on tests

Run tests locally to debug:
```bash
pytest tests/ -v
```

### Version already exists

If you need to re-release, delete the tag and release first:
```bash
git tag -d v{version}
git push origin :refs/tags/v{version}
```
Then delete the release from GitHub's releases page.

## Manual Release (Advanced)

If you need to create a release manually:

```bash
# Install hatch
pip install hatch

# Bump version
hatch version patch  # or minor, or major

# Build package
hatch build

# Create git tag
git tag v$(hatch version)
git push origin v$(hatch version)

# Upload to GitHub manually or use gh CLI
gh release create v$(hatch version) \
  --title "Release v$(hatch version)" \
  --notes "Manual release" \
  dist/*
```

## CI/CD

The `ci.yml` workflow runs automatically on:
- Every push to `main`
- Every pull request to `main`

This ensures code quality is maintained before releases.
