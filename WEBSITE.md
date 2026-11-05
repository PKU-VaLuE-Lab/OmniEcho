# OmniEcho website

The project homepage and separate Audio & Vision leaderboard are static pages in `docs/`.
The site includes an online project demo, local fonts, a white theme, a spatial globe with pointer parallax,
responsive layouts, accessible sorting and search, and downloadable benchmark data.

## Preview

```sh
python -m http.server 8765 --directory docs
```

Open `http://localhost:8765/` and `http://localhost:8765/leaderboard/`.
No package installation or build is required to serve the website.

## Publish with GitHub Pages after migration

The website is stored alongside the source in this repository. Its publishing
configuration targets GitHub, even while the repository is hosted elsewhere.
The intended GitHub repository is `PKU-VaLuE-Lab/OmniEcho`.

After migrating the complete repository, including `.github/workflows/pages.yml`:

1. Open the GitHub repository's **Settings → Pages**.
2. Under **Build and deployment → Source**, select **GitHub Actions**.
3. Open **Actions → Deploy project website → Run workflow**, select the default
   branch, and run it once to publish the initial site.
4. Wait for the workflow to succeed. Future changes to `docs/` on the default
   branch publish automatically.

The workflow supports both `main` and `master` and deploys only the default branch
on GitHub. It uses the built-in GitHub token and does not require a personal token.
It publishes the contents of `docs/`, without running the model code or installing
its dependencies. The existing `docs/benchmark_data.md` remains in place.

The expected URLs after a successful deployment are:

- https://pku-value-lab.github.io/OmniEcho/
- https://pku-value-lab.github.io/OmniEcho/leaderboard/

Set the repository's **About → Website** field to the homepage URL. Deployment is
not performed by pushing to the current non-GitHub host. If the GitHub organization
or repository name changes, update the canonical/social URLs in `docs/index.html`
and `scripts/build_leaderboard.py`, the README links, and this guide; then regenerate
the leaderboard. Internal navigation and asset references are relative paths.

## Update the leaderboard

Edit the model records in `scripts/build_leaderboard.py`, then run:

```sh
python scripts/build_leaderboard.py
```

This regenerates the accessible HTML table, JSON metadata and CSV download together.
If the number of models changes, update the summary count and descriptive copy in the
template in that script. The complete table is readable without JavaScript.

The initial table combines the paper's Audio & Vision results with additional results
provided by the project maintainer. OmniEcho's Camera Rotation and 3D Localization
scores use the confirmed paper values, 41.7 and 14.2. Overall is preserved as reported;
it is not recomputed as the unweighted mean of task columns. Audio-only results and
Qwen3.8-omni-flash-realtime are excluded. The Input column is intentionally omitted.

## Assets and attribution

- Demo playback uses the maintainer-provided HTTPS video URL directly:
  `https://cdn-uploads.huggingface.co/production/uploads/66cd4cb53f233bf2c7953c12/P395iT-mTRonbT96yxafl.mp4`.
  The MP4 is not included in the website package. The player loads it on demand;
  no video upload to GitHub is required. An Open video link is available below the player.
- `docs/assets/media/demo-poster.jpg`: the existing demo cover image, bundled locally.
- DM Sans and Space Grotesk are bundled locally under their SIL Open Font Licenses,
  included in `docs/assets/fonts/`.
- Research text, author list, statistics and citation are based on the project README
  and [arXiv:2609.23407](https://arxiv.org/abs/2609.23407).
- The decorative spatial globe is drawn locally with Canvas in `cosmic.js`;
  `light.css` provides the white theme with blue and violet accents. The homepage includes an animation
  pause control and respects the system reduced-motion preference.

The website does not use analytics, third-party JavaScript or external font requests.
The model and data release status continues to be managed in the main README.
