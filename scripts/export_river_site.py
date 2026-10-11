#!/usr/bin/env python3
"""Export the observed two-station model website and its reproducible release."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = Path('experiments/observed_rivers/provenance.json')
RESULTS = Path('experiments/observed_rivers/website_data.json')
RAW_ALLOWLIST = {
    'yellow_river': 'experiments/observed_rivers/raw/lijin_daily_2015_2021.xlsx',
    'mississippi': 'experiments/observed_rivers/raw/usgs_st_louis_daily.json',
}


def reviewed_inputs():
    provenance = json.loads((ROOT / PROVENANCE).read_text(encoding='utf-8'))
    if provenance.get('data_status') != 'observed' or provenance.get('model_status') != 'trained_observed_daily':
        raise ValueError('The observed model release must be completed before publication')
    results = json.loads((ROOT / RESULTS).read_text(encoding='utf-8'))
    if results.get('data_status') != 'observed' or results.get('model_status') != 'trained_observed_daily':
        raise ValueError('Missing completed observed-model results')
    if set(results.get('rivers', {})) != set(RAW_ALLOWLIST):
        raise ValueError('Only the two reviewed river stations may be published')
    sources = provenance.get('sources', [])
    if len(sources) != 2 or {s.get('river_id') for s in sources} != set(RAW_ALLOWLIST):
        raise ValueError('Expected exactly the two reviewed source archives')
    required = [PROVENANCE, RESULTS, Path('website/assets/observed_river_data.js')]
    required += [Path('website') / name for name in ['index.html', 'styles.css', 'app.js']]
    required += [Path('docs') / name for name in ['website_data_status.md', 'river_comparative_study.md', 'observed_model_study.md', 'publishing_guide.md', 'cloudflare_pages.md']]
    required += [Path('scripts') / name for name in ['export_river_site.py', 'build_cloudflare_pages.py', 'run_observed_rivers.py', 'predict_observed_river.py', 'verify_observed_rivers.py']]
    required += [Path('src/data/observed_rivers.py'), Path('src/models/observed_rivers.py')]
    required += [Path('tests/test_observed_rivers.py'), Path('tests/test_source_only_site.py'), Path('tests/test_cloudflare_pages.py')]
    required += [Path('cloudflare/_headers'), Path('cloudflare/404.html'), Path('wrangler.toml')]
    required += [Path('requirements_observed.txt'), Path('LICENSE.observed-models')]
    for source in sources:
        path = source.get('raw_file')
        if path != RAW_ALLOWLIST[source['river_id']] or source.get('redistribution_allowed') is not True:
            raise ValueError('Unreviewed source path or redistribution permission')
        original = ROOT / path
        if original.is_symlink() or not original.is_file():
            raise ValueError('Original source archive missing or not a regular file: ' + path)
        if hashlib.sha256(original.read_bytes()).hexdigest() != source['raw_sha256']:
            raise ValueError('Source checksum mismatch: ' + path)
        required.append(Path(path))
    artifacts = results.get('artifacts', [])
    if not artifacts:
        raise ValueError('No completed model artifacts to export')
    for artifact in artifacts:
        path = Path(artifact['path'])
        allowed = path.as_posix().startswith('experiments/observed_rivers/')
        if not allowed or path.is_absolute() or '..' in path.parts or path.suffix not in {'.pt', '.csv', '.json', '.xlsx'}:
            raise ValueError('Unreviewed experiment artifact: ' + str(path))
        if 'raw' in path.parts and path.as_posix() not in set(RAW_ALLOWLIST.values()):
            raise ValueError('Unreviewed original archive: ' + str(path))
        if any(part.startswith('.') for part in path.parts):
            raise ValueError('Hidden artifact path: ' + str(path))
        required.append(path)
    for path in required:
        file = ROOT / path
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(ROOT):
            raise ValueError('Missing reviewed publication input: ' + str(path))
    html = (ROOT / 'website/index.html').read_text(encoding='utf-8')
    app = (ROOT / 'website/app.js').read_text(encoding='utf-8')
    if any(token in html + app for token in ['assets/river_lab_data.js', 'window.RIVER_LAB_DATA', 'experiments/multi_river/']):
        raise ValueError('Website still references withdrawn synthetic results')
    return provenance, sorted(set(required))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'public_river_lab')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == ROOT or destination in ROOT.parents:
        parser.error('Export destination must not be the project or an ancestor')
    if destination.exists():
        parser.error('Export destination already exists; choose a new folder to preserve its contents')
    try:
        provenance, required = reviewed_inputs()
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    destination.mkdir(parents=True)
    for relative in required:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (destination / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=website/index.html"><title>Between Rivers</title><a href="website/index.html">Explore Between Rivers</a></html>\n', encoding='utf-8')
    (destination / '.nojekyll').touch()
    manifest = {
        'data_status': provenance['data_status'], 'model_status': provenance['model_status'],
        'snapshot_date': provenance['snapshot_date'],
        'excludes': 'Legacy synthetic river experiments and website assets, project database, environments, secrets and backups',
        'files': {},
    }
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            manifest['files'][path.relative_to(destination).as_posix()] = {
                'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    (destination / 'publication_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Observed two-station model export: {destination}\nFiles: {len(manifest["files"])}')


if __name__ == '__main__':
    main()
