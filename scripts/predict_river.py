#!/usr/bin/env python3
"""Run a saved multi-river model on an explicitly selected hourly CSV."""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.multi_river import load_observed_csv
from src.models.multi_river import FeatureScaler, RiverGraphGRU
from scripts.run_multi_river import export_prediction_rows, physical_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--river', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-status', choices=['observed', 'synthetic'], required=True,
                        help='Explicit provenance of the supplied CSV, independent of checkpoint training provenance')
    args = parser.parse_args()
    path = args.checkpoint or ROOT / 'experiments' / 'multi_river' / 'models' / f'{args.river}_physics_guided.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('protocol_version') != 3:
        parser.error('Only multi-river protocol 3 checkpoints are supported')
    profiles = json.loads((ROOT / 'configs' / 'rivers.json').read_text(encoding='utf-8'))
    if isinstance(profiles, dict) and 'rivers' in profiles:
        profiles = profiles['rivers']
    if isinstance(profiles, list):
        profiles = {item['id']: item for item in profiles}
    if args.river not in profiles or checkpoint['profile'].get('id') != args.river:
        parser.error('River ID does not match the saved checkpoint')
    if checkpoint['protocol']['data_status'] == 'synthetic' and args.source_status == 'observed':
        print('模型使用模拟数据训练；以下结果属于真实输入上的未校准研究推断。', file=sys.stderr)
    series, timestamps = load_observed_csv(args.input, checkpoint['profile'])
    if len(series) < 24:
        parser.error('At least 24 complete hourly timestamps are required')
    model = RiverGraphGRU(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = FeatureScaler.from_dict(checkpoint['scaler'])
    values = physical_predict(model, scaler.transform(series[-24:][None]), scaler)[0]
    method = checkpoint['protocol'].get('method', 'physics_guided')
    issued_at = str(timestamps[-1])
    issued_date = datetime.fromisoformat(issued_at.replace('Z', '+00:00'))
    future = [(issued_date + timedelta(hours=i)).isoformat() for i in range(1, 7)]
    export_prediction_rows(args.output, future, values, method, args.source_status, origins=[issued_at])
    args.output.with_suffix('.metadata.json').write_text(json.dumps({
        'river': args.river, 'issued_at': issued_at, 'input_status': args.source_status,
        'training_status': checkpoint['protocol']['data_status'], 'model_checkpoint': str(path), 'method': method,
        'horizon_hours': 6, 'features': checkpoint['features'][:3], 'units': checkpoint['units'][:3],
        'parameter_status': checkpoint['protocol']['parameter_status']}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
