# Between Rivers: viewing and publishing

The English-language website focuses on a seven-angle comparison of the Yellow and Mississippi Rivers: hydrology, sediment, dams, flood risk, ecology, riverside communities, and cultural memory. It includes a seasonal-flow illustration, a short introduction to the available discharge models, and an illustrative six-day discharge scenario displayed for 1–6 January 2027. The scenario reuses unchanged seed-42 model outputs for 1–6 January 2022, generated from a 31 December 2021 forecast origin; it is not a newly computed 2027 forecast. The original CSV files remain in the published research artifacts with their 2022 dates and values, but are not linked below the scenario table. The table retains its scenario and illustrative labels without a separate explanatory footer. The scenario table and model downloads also work without JavaScript. Technical experiments remain in the project, but transfer-learning tables and detailed model cards are not displayed on the page. Training is not required to serve it. The linked comparison article, data notes, model notes, navigation, chart labels, and 404 page are also in English. Original source archives and research metadata are preserved without translation.

## Local preview

The scenario panel pairs its daily table with a two-river line chart. Both series share a zero-based discharge axis in m³/s; distinct colors, line styles and point shapes distinguish the stations. The chart reads its six dates and twelve values directly from the table, uses straight segments, and redraws to fit its available width. The static table remains available without JavaScript. No forecast values, model files or observation records are changed by the chart.

From the project root:

```bash
python3 -m http.server 8080
```

Open [http://localhost:8080/website/](http://localhost:8080/website/). Keep the website, experiments, scripts and documentation in their relative directories so downloads remain available.

## Portable static export

```bash
python3 scripts/export_river_site.py --output public_river_lab_new
python3 -m http.server 8080 --directory public_river_lab_new
```

The exporter checks the source-file hashes and redistribution permissions, then includes only the two-station experiment, its model code, documentation and website assets. Existing output directories are never overwritten. The publication manifest records each exported file's size and SHA-256 digest.

Use the exported directory's contents as a static hosting root. Its root page redirects to `website/index.html`. Check all seven comparison topics, the article navigation, seasonal illustration, model downloads, mobile layout and consolidated source references before publishing.

The export excludes legacy six-river simulations, the project database, environments, credentials and backups. The 2027 scenario is an illustrative reuse of historical predictions, not a live water-monitoring feed. Reproduction details are in [the model study](observed_model_study.md); source attribution and data licenses are in [the source note](website_data_status.md).

Updating local files or creating this export does not deploy the public website or push GitHub.

## Cloudflare Pages

Build with `python3 -B scripts/build_cloudflare_pages.py`, then use `dist` as the Pages build output directory. The Pages build contains a complete homepage at `/`, retains `/website/`, and includes download headers and a custom 404 page. It uses only Python's standard library, not the model-training dependencies.

For Git integration, select **None** as the framework and set the build environment variable **`SKIP_DEPENDENCY_INSTALL=1`** for both production and previews. For direct upload, `python3 -B scripts/build_cloudflare_pages.py --zip cloudflare_pages.zip` creates a fresh ZIP with the files directly at its root. See [the Cloudflare Pages guide](cloudflare_pages.md) for all settings, safe rebuild behavior and deployment checks.
