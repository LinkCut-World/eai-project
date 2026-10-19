# Project website

The website is a static GitHub Pages site with no build dependencies.

- Publish from `main` / `/docs` in Settings > Pages > Deploy from a branch.
- Public URL: https://linkcut-world.github.io/eai-project/
- Preview locally: `python -m http.server 8000 --directory docs`.

The two demo videos appear side by side on desktop and stack on narrow screens.
The full project report appears below them, with direct open and download links.
Its self-hosted PDF.js reader includes page navigation and zoom, and loads only
when the reader approaches the viewport. It does not depend on a browser PDF plugin
or any external CDN.

## Assets

- `assets/demo-1.mp4`: the supplied first demo, encoded as 1080p H.264/AAC for
  browser playback and GitHub's file-size limit; its original duration is retained.
- `assets/demo-2.mp4`: the supplied second demo, remuxed without re-encoding.
- `assets/report.pdf`: the supplied project report, copied without changes.
- `assets/demo-1-poster.jpg` and `assets/demo-2-poster.jpg`: frames from the demos.

Both MP4 files have their metadata at the beginning for progressive playback.
Desktop originals are not modified.

## Third-party library

`vendor/pdfjs/` contains PDF.js 5.6.205 from Mozilla's `pdfjs-dist` distribution,
including the matching web worker, standard fonts, WASM decoders, and license
notices. See `vendor/pdfjs/LICENSE` and the font and WASM license files. Upstream:
https://github.com/mozilla/pdf.js

## Source snapshot

The project source comes from the default `main`
branch of `git@git.tsinghua.edu.cn:lcw24/eai-project-clean.git`, snapshot
`c070c6211f922da6532d7fe70a1b35132a0bbaa3`. The root README additionally lists
the project authors. The GitHub repository was initialized
independently, with one root commit and no imported Git history. The website and
its assets are added in `docs/` in that same initial commit.
