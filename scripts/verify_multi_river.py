#!/usr/bin/env python3
"""Verify generated artifact integrity and forecasts from restored checkpoints."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.multi_river import chronological_windows, load_observed_csv
from src.models.multi_river import FeatureScaler, RiverGraphGRU
from scripts.run_multi_river import physical_predict


def verify():
    torch.set_num_threads(2)
    data = json.loads((ROOT / 'experiments/multi_river/website_data.json').read_text(encoding='utf-8'))
    artifact_root = ROOT / data['artifact_root']
    manifest = json.loads((artifact_root / 'manifest.json').read_text(encoding='utf-8'))
    for relative, record in manifest['files'].items():
        path = artifact_root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise AssertionError('Artifact checksum changed: ' + relative)
    model_count = 0
    for path in sorted(artifact_root.rglob('*.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['protocol_version'] != 3:
            raise AssertionError('Unexpected protocol: ' + str(path))
        model = RiverGraphGRU(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        FeatureScaler.from_dict(checkpoint['scaler'])
        model_count += 1
    for river in data['rivers']:
        checkpoints = [artifact_root / 'models' / f"{river['id']}_{method}.pt" for method in ['physics_guided', 'data_only']]
        for path in checkpoints:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model = RiverGraphGRU(**checkpoint['model_config'])
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = FeatureScaler.from_dict(checkpoint['scaler'])
            synthetic_path = artifact_root / 'data' / f"{river['id']}.csv"
            if data['data_status'] == 'synthetic':
                series, _ = load_observed_csv(synthetic_path, checkpoint['profile'])
                split = chronological_windows(series)
                expected = FeatureScaler().fit(series[:split['train_end']])
                np.testing.assert_allclose(scaler.mean, expected.mean, rtol=1e-5, atol=1e-5)
                latest = series[-24:][None]
                restored = physical_predict(model, scaler.transform(latest), scaler)[0]
                expected_forecast = np.asarray(river['forecast'][checkpoint['protocol']['method']])
                np.testing.assert_allclose(restored, expected_forecast, rtol=1e-5, atol=1e-5)
        for download in river['downloads'].values():
            if not (ROOT / 'website' / download).resolve().is_file():
                raise AssertionError('Missing website download: ' + download)
        path = (ROOT / 'website' / river['downloads']['predictions']).resolve()
        with path.open(encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 90 or any(row['observed'] for row in rows):
            raise AssertionError('Future CSV must have 6×5×3 predictions and no fabricated observations')
        if {row['origin'] for row in rows} != {river['forecast']['issued_at']}:
            raise AssertionError('Forecast issue timestamps differ')
    print(json.dumps({'rivers': len(data['rivers']), 'restorable_checkpoints': model_count,
                      'verified_artifacts': len(manifest['files']), 'forecast_reload': 'passed',
                      'download_files': 'passed', 'future_observations': 'blank'}, ensure_ascii=False))


if __name__ == '__main__':
    verify()
