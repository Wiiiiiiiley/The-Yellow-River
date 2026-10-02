#!/usr/bin/env python3
"""Package only reviewed public river-lab artifacts as a portable static site."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'public_river_lab')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination == ROOT or destination in ROOT.parents:
        parser.error('Export destination must not be the project or an ancestor')
    if destination.exists():
        parser.error('Export destination already exists; select a new folder to preserve its contents')
    data_path = ROOT / 'website' / 'assets' / 'river_lab_data.js'
    data = json.loads(data_path.read_text(encoding='utf-8').removeprefix('window.RIVER_LAB_DATA = ').strip().removesuffix(';'))
    required_docs = ['multi_river_study.md', 'yellow_mississippi_comparison.md', 'multi_river_results.md', 'publishing_guide.md']
    required = [ROOT / 'website' / name for name in ['index.html', 'styles.css', 'app.js']] + [data_path]
    required += [ROOT / 'docs' / name for name in required_docs]
    artifact_root = (ROOT / data.get('artifact_root', 'experiments/multi_river')).resolve()
    if not artifact_root.is_relative_to(ROOT) or artifact_root == ROOT:
        parser.error('Invalid generated artifact directory')
    for river in data['rivers']:
        for value in river['downloads'].values():
            path = (ROOT / 'website' / value).resolve()
            if not path.is_relative_to(artifact_root):
                parser.error('Export contains an unsupported artifact path: ' + value)
            required.append(path)
    # Explicit publication allowlist: models, summaries and prediction exports.
    # Full raw observation archives and the original database are not packaged.
    for directory in ['models', 'predictions', 'metrics', 'transfer']:
        required += [path for path in (artifact_root / directory).rglob('*') if path.is_file()]
    required += [artifact_root / 'website_data.json']
    required += [ROOT / 'configs' / name for name in ['rivers.json', 'river_sources.json']]
    required += [ROOT / 'src' / 'data' / 'multi_river.py', ROOT / 'src' / 'models' / 'multi_river.py']
    required += [ROOT / 'scripts' / name for name in ['run_multi_river.py', 'predict_river.py', 'export_river_site.py']]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Missing generated inputs: ' + ', '.join(missing))
    destination.mkdir(parents=True)
    for path in sorted(set(required)):
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (destination / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=website/index.html"><title>大河研究室</title><a href="website/index.html">进入大河研究室</a></html>\n', encoding='utf-8')
    (destination / '.nojekyll').touch()
    manifest = {'artifact_root': data.get('artifact_root'), 'data_status': data['data_status'],
                'excludes': '原项目数据库、虚拟环境、密钥与完整原始输入档案', 'files': {}}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            manifest['files'][path.relative_to(destination).as_posix()] = {'bytes': path.stat().st_size,
                                                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination / 'publication_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Static export: {destination}\nServe this folder as the website root.')


if __name__ == '__main__':
    main()
