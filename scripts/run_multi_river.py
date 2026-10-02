#!/usr/bin/env python3
"""Reproducible multi-river training, held-out evaluation and website export.

The default experiment is a synthetic research demonstration. Observed CSV
inputs must be explicitly supplied; there is no silent synthetic fallback.
"""
import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.multi_river import (chronological_windows, generate_series,
                                  load_observed_csv, load_profiles)
from src.models.multi_river import (FeatureScaler, RiverGraphGRU, metrics,
                                    physics_penalty, predict)

FEATURES = ['water_depth', 'discharge', 'sediment', 'temperature', 'rainfall', 'season_sin']
UNITS = ['m', 'm³/s', 'kg/m³', '°C', 'mm/h', '1']
MODEL_CONFIG = dict(n_nodes=5, n_features=6, hidden_size=32, horizon=6)
PROTOCOL_VERSION = 3


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def physical_predict(model, inputs, scaler):
    return np.maximum(np.asarray(scaler.inverse(predict(model, inputs))), 0)


def train_model(train, val, scaler, profile, seed, epochs, physics_weight,
                initial_state=None):
    """Same architecture, initialization and target-update budget for ablations."""
    set_seed(seed)
    model = RiverGraphGRU(**MODEL_CONFIG)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    tx = torch.as_tensor(scaler.transform(train['x']), dtype=torch.float32)
    ty = torch.as_tensor(scaler.transform(train['y']), dtype=torch.float32)
    vx = torch.as_tensor(scaler.transform(val['x']), dtype=torch.float32)
    vy = torch.as_tensor(scaler.transform(val['y']), dtype=torch.float32)
    best = float('inf')
    best_state = None
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tx))
        loss_sum = 0.0
        for ids in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(tx[ids])
            data_loss = (predicted - ty[ids]).square().mean()
            penalty = physics_penalty(predicted, scaler, profile) if physics_weight else data_loss * 0
            loss = data_loss + physics_weight * penalty
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: ' + profile['name_en'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            validation_loss = float((model(vx) - vy).square().mean())
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(tx), 'validation_mse_normalized': validation_loss})
        if validation_loss < best:
            best = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def save_checkpoint(path, model, scaler, profile, protocol, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'protocol_version': PROTOCOL_VERSION,
                'model_config': MODEL_CONFIG,
                'model_state_dict': model.state_dict(),
                'scaler': scaler.to_dict(), 'profile': profile,
                'protocol': protocol, 'features': FEATURES, 'units': UNITS,
                'history': history}, path)


def export_series(path, series, timestamps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['timestamp', 'station_id', 'source_status'] + FEATURES)
        for t, timestamp in enumerate(timestamps):
            for node in range(series.shape[1]):
                writer.writerow([timestamp, 'r' + str(node + 1), 'synthetic'] + [round(float(v), 7) for v in series[t, node]])


def export_prediction_rows(path, timestamps, values, model_name, source_status,
                           truth=None, lower=None, upper=None, origins=None):
    """Forecast CSV keeps unknown future observations blank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if values.ndim == 3:
        values = values[None]
        timestamps = [timestamps]
        truth = None if truth is None else truth[None]
        lower = None if lower is None else lower[None]
        upper = None if upper is None else upper[None]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['origin', 'timestamp', 'lead_hours', 'station_id', 'model', 'source_status', 'feature', 'unit', 'prediction', 'observed', 'lower_90', 'upper_90'])
        for sample in range(len(values)):
            for lead in range(6):
                for node in range(5):
                    for feature in range(3):
                        writer.writerow([origins[sample] if origins is not None else '', timestamps[sample][lead], lead + 1,
                                         'r' + str(node + 1), model_name, source_status, FEATURES[feature], UNITS[feature],
                                         float(values[sample, lead, node, feature]),
                                         '' if truth is None else float(truth[sample, lead, node, feature]),
                                         '' if lower is None else float(lower[sample, lead, node, feature]),
                                         '' if upper is None else float(upper[sample, lead, node, feature])])


def mean_nrmse(result):
    values = [result[feature]['nrmse'] for feature in FEATURES[:3] if result[feature]['nrmse'] is not None]
    return float(np.mean(values)) if values else None


def summarize(values):
    values = [value for value in values if value is not None]
    return (float(np.mean(values)), float(np.std(values))) if values else (None, None)


def run_transfer(states, args, output):
    directions = []
    for source_id, target_id in [('yellow_river', 'mississippi'), ('mississippi', 'yellow_river')]:
        if source_id not in states or target_id not in states:
            continue
        source, target = states[source_id], states[target_id]
        train = target['splits']['train']
        ids = np.arange(0, len(train['x']), 5)
        few = {key: value[ids] for key, value in train.items() if key in ('x', 'y', 'origins')}
        rows = []
        for seed in args.transfer_seeds:
            if seed == args.seed:
                pretrained = source['model']
            else:
                pretrained, _ = train_model(source['splits']['train'], source['splits']['val'], source['scaler'], source['profile'], seed, args.epochs, args.physics_weight)
            target_x = target['scaler'].transform(target['splits']['test']['x'])
            zero_pred = physical_predict(pretrained, target_x, target['scaler'])
            zero_protocol = dict(target['protocol'], seed=seed, method='zero_shot', transfer_source=source_id,
                                 source_epochs=args.epochs, target_epochs=0, epochs=0,
                                 target_label_windows=0, checkpoint_selection='source validation only; target scaler calibration')
            save_checkpoint(output / 'transfer' / 'models' / f'{source_id}_to_{target_id}_zero_shot_{seed}.pt',
                            pretrained, target['scaler'], target['profile'], zero_protocol, [])
            rows.append({'seed': seed, 'method': 'zero_shot', 'metrics': metrics(zero_pred, target['splits']['test']['y'])})
            for method, initial in [('target_only', None), ('fine_tuned', pretrained.state_dict())]:
                fitted, history = train_model(few, target['splits']['val'], target['scaler'], target['profile'], seed,
                                               args.transfer_epochs, args.physics_weight, initial)
                pred = physical_predict(fitted, target_x, target['scaler'])
                score = metrics(pred, target['splits']['test']['y'])
                rows.append({'seed': seed, 'method': method, 'metrics': score})
                protocol = dict(target['protocol'], transfer_source=source_id if method == 'fine_tuned' else None,
                                seed=seed, method=method, target_label_windows=len(ids), source_epochs=args.epochs if initial is not None else 0,
                                epochs=args.transfer_epochs, target_epochs=args.transfer_epochs)
                save_checkpoint(output / 'transfer' / 'models' / f'{source_id}_to_{target_id}_{method}_{seed}.pt', fitted, target['scaler'], target['profile'], protocol, history)
            print(f'  transfer {source_id} → {target_id}, seed {seed} complete', flush=True)
        summaries = []
        for method in ['target_only', 'zero_shot', 'fine_tuned']:
            scores = [row['metrics'] for row in rows if row['method'] == method]
            values = [mean_nrmse(score) for score in scores]
            average, spread = summarize(values)
            q_average, q_spread = summarize([v['discharge']['nrmse'] for v in scores])
            summaries.append({'method': method, 'nrmse_mean': average, 'nrmse_std': spread,
                              'discharge_nrmse_mean': q_average, 'discharge_nrmse_std': q_spread})
        scratch = summaries[0]['nrmse_mean']
        tuned = summaries[2]['nrmse_mean']
        change = (scratch - tuned) / scratch * 100 if scratch is not None and scratch > 0 and tuned is not None else None
        domain = '模拟测试' if target['protocol']['data_status'] == 'synthetic' else '观测测试'
        conclusion = (f'在这组{domain}中，微调相对目标河独立少样本训练的平均归一化误差变化为 {change:+.1f}%（正值表示误差降低）。'
                      if change is not None else '测试目标缺少变化或独立训练误差为零，无法计算相对归一化误差改善。')
        directions.append({'source': source_id, 'target': target_id, 'rows': rows, 'summary': summaries,
                           'few_shot_fraction': len(ids) / len(train['x']), 'target_label_windows': len(ids),
                           'target_validation_windows': len(target['splits']['val']['x']),
                           'target_training_windows': len(train['x']), 'source_epochs': args.epochs,
                           'target_epochs': args.transfer_epochs, 'conclusion': conclusion})
    result = {'directions': directions, 'seeds': args.transfer_seeds,
              'scaling': '每条河仅使用其训练时段的无标签输入拟合缩放器；零样本指目标监督训练为零，仍使用目标训练输入校准。',
              'validation': '少样本训练取目标训练窗口每五个中的一个；各方法共享完整目标验证集和测试集。验证标签不计入训练窗口预算。',
              'compute': '独立训练与微调目标更新轮数相同；迁移额外使用完整源河预训练计算。',
              'scope': '三个随机种子反映初始化变化；同一组合成序列不代表跨气候、跨站点或真实数据泛化。'}
    write_json(output / 'transfer' / 'results.json', result)
    return result


def write_report(data, path):
    lines = ['# 多河流实验结果', '', f"生成时间：{data['created_at']}。数据状态：{data['data_status']}。", '',
             '所有结果来自本次实际运行；模拟数值的来源是情景配置，不是参考文献中的实测值。', '',
             '| 河流 | 方法 | 水深 RMSE (m) | 流量 RMSE (m³/s) | 泥沙 RMSE (kg/m³) |',
             '|---|---|---:|---:|---:|']
    for river in data['rivers']:
        for method, score in river['metrics'].items():
            lines.append(f"| {river['name_zh']} | {method} | {score['water_depth']['rmse']:.4f} | {score['discharge']['rmse']:.2f} | {score['sediment']['rmse']:.4f} |")
    lines += ['', '## 黄河与密西西比河的双向迁移', '']
    for direction in data['transfer']['directions']:
        lines += [f"### {direction['source']} → {direction['target']}", '', direction['conclusion'], '',
                  '| 方法 | 三变量平均 NRMSE | 随机种子标准差 |', '|---|---:|---:|']
        for row in direction['summary']:
            average = f"{row['nrmse_mean']:.4f}" if row['nrmse_mean'] is not None else '—'
            spread = f"{row['nrmse_std']:.4f}" if row['nrmse_std'] is not None else '—'
            lines.append(f"| {row['method']} | {average} | {spread} |")
    lines += ['', '## 解释边界', ''] + ['- ' + item for item in data['limitations']]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--hours', type=int, default=8760)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rivers', nargs='+', default=None)
    parser.add_argument('--data-dir', type=Path, help='Explicit observed input folder: <river_id>.csv for every selected river')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments' / 'multi_river')
    parser.add_argument('--physics-weight', type=float, default=0.05)
    parser.add_argument('--transfer-epochs', type=int, default=15)
    parser.add_argument('--transfer-seeds', default='42,43,44')
    parser.add_argument('--skip-transfer', action='store_true')
    args = parser.parse_args()
    args.transfer_seeds = [int(value) for value in args.transfer_seeds.split(',')]
    if args.epochs < 1 or args.transfer_epochs < 1 or args.hours < 720 or not args.transfer_seeds or args.physics_weight < 0:
        parser.error('epochs≥1, hours≥720, nonempty transfer seeds, physics-weight≥0 required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    profiles = load_profiles(ROOT / 'configs' / 'rivers.json')
    selected = args.rivers or list(profiles)
    unknown = set(selected) - set(profiles)
    if unknown:
        parser.error('Unknown river IDs: ' + ', '.join(sorted(unknown)))
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error('--output must be inside this project so website download links remain portable')
    output.mkdir(parents=True, exist_ok=True)
    source_status = 'observed' if args.data_dir else 'synthetic'
    sources = json.loads((ROOT / 'configs' / 'river_sources.json').read_text(encoding='utf-8'))
    if isinstance(sources, dict):
        sources = sources['sources']
    states = {}
    river_exports = []
    for river_id in selected:
        print(f'Training {river_id} ({source_status})...', flush=True)
        profile = profiles[river_id]
        if args.data_dir:
            series, timestamps = load_observed_csv(args.data_dir / f'{river_id}.csv', profile)
        else:
            series, timestamps = generate_series(profile, args.hours, args.seed)
            export_series(output / 'data' / f'{river_id}.csv', series, timestamps)
        timestamps = [str(value) for value in timestamps]
        splits = chronological_windows(series)
        scaler = FeatureScaler().fit(series[:splits['train_end']])
        protocol = {'version': PROTOCOL_VERSION, 'data_status': source_status, 'seed': args.seed,
                    'epochs': args.epochs, 'hours': len(series), 'history_hours': 24, 'horizon_hours': 6,
                    'split': '70/15/15 chronological', 'stride_hours': 24,
                    'normalization': 'training period only; all methods use the same scaler',
                    'start': timestamps[0], 'end': timestamps[-1],
                    'train_end_exclusive': splits['train_end'], 'val_end_exclusive': splits['val_end'],
                    'split_counts': {part: len(splits[part]['x']) for part in ['train', 'val', 'test']},
                    'physics_weight': args.physics_weight,
                    'architecture': MODEL_CONFIG, 'checkpoint_selection': 'minimum validation normalized MSE',
                    'parameter_status': 'illustrative_uncalibrated', 'torch_version': str(torch.__version__)}
        trained = {}
        scores = {}
        predictions = {}
        for method, weight in [('physics_guided', args.physics_weight), ('data_only', 0.0)]:
            model, history = train_model(splits['train'], splits['val'], scaler, profile, args.seed, args.epochs, weight)
            trained[method] = model
            ckpt_protocol = dict(protocol, method=method, physics_weight=weight)
            save_checkpoint(output / 'models' / f'{river_id}_{method}.pt', model, scaler, profile, ckpt_protocol, history)
            predictions[method] = physical_predict(model, scaler.transform(splits['test']['x']), scaler)
            scores[method] = metrics(predictions[method], splits['test']['y'])
            write_json(output / 'models' / f'{river_id}_{method}_history.json', history)
        persistence = np.repeat(splits['test']['x'][:, -1:, :, :3], 6, axis=1)
        predictions['persistence'] = persistence
        scores['persistence'] = metrics(persistence, splits['test']['y'])
        val_predictions = physical_predict(trained['physics_guided'], scaler.transform(splits['val']['x']), scaler)
        residual = np.quantile(np.abs(val_predictions - splits['val']['y']), 0.9, axis=0)
        coverage = {FEATURES[f]: float(np.mean(np.abs(predictions['physics_guided'][..., f] - splits['test']['y'][..., f]) <= residual[..., f])) for f in range(3)}
        history_values = series[-24:, :, :3]
        issued_at = timestamps[-1]
        issued_date = datetime.fromisoformat(issued_at.replace('Z', '+00:00'))
        future_times = [(issued_date + timedelta(hours=h)).isoformat() for h in range(1, 7)]
        latest_x = scaler.transform(series[-24:][None])
        forecasts = {method: physical_predict(model, latest_x, scaler)[0] for method, model in trained.items()}
        forecasts['persistence'] = np.repeat(series[-1:, :, :3], 6, axis=0)
        lower = np.maximum(0, forecasts['physics_guided'] - residual)
        upper = forecasts['physics_guided'] + residual
        test_times = [[timestamps[int(origin) + h] for h in range(1, 7)] for origin in splits['test']['origins']]
        test_origins = [timestamps[int(origin)] for origin in splits['test']['origins']]
        for method, predicted in predictions.items():
            export_prediction_rows(output / 'predictions' / f'{river_id}_{method}_backtest.csv', test_times, predicted, method, source_status, splits['test']['y'], origins=test_origins)
            export_prediction_rows(output / 'predictions' / f'{river_id}_{method}_forecast.csv', future_times, forecasts[method], method, source_status,
                                   lower=lower if method == 'physics_guided' else None,
                                   upper=upper if method == 'physics_guided' else None, origins=[issued_at])
        write_json(output / 'metrics' / f'{river_id}.json', {'metrics': scores, 'protocol': protocol, 'interval_coverage': coverage})
        reach_names = profile.get('reach_names', profile.get('station_names', ['示意河段 ' + str(i) for i in range(1, 6)]))
        relative_output = '../' + output.relative_to(ROOT).as_posix() if output.is_relative_to(ROOT) else ''
        download_base = relative_output
        export = {'id': river_id, 'name_zh': profile['name_zh'], 'name_en': profile['name_en'], 'region': profile['region'],
                  'summary': profile['summary'], 'hydrology': profile['hydrology'], 'profile': profile,
                  'reaches': reach_names, 'metrics': scores, 'protocol': protocol,
                  'history': {'timestamps': timestamps[-24:], 'values': history_values.tolist()},
                  'forecast': {'issued_at': issued_at, 'timestamps': future_times, **{name: values.tolist() for name, values in forecasts.items()},
                               'lower': lower.tolist(), 'upper': upper.tolist(), 'interval_label': '验证集绝对残差90%经验区间', 'interval_coverage': coverage},
                  'backtest': {'issued_at': test_origins[-1], 'timestamps': test_times[-1], 'observed': splits['test']['y'][-1].tolist(),
                               **{name: values[-1].tolist() for name, values in predictions.items()}},
                  'downloads': {'checkpoint': f'{download_base}/models/{river_id}_physics_guided.pt',
                                'data_checkpoint': f'{download_base}/models/{river_id}_data_only.pt',
                                'predictions': f'{download_base}/predictions/{river_id}_physics_guided_forecast.csv',
                                'backtest': f'{download_base}/predictions/{river_id}_physics_guided_backtest.csv',
                                'metrics': f'{download_base}/metrics/{river_id}.json'}}
        river_exports.append(export)
        states[river_id] = {'profile': profile, 'splits': splits, 'scaler': scaler, 'model': trained['physics_guided'], 'protocol': protocol}
        print(f"  done; discharge RMSE: guided={scores['physics_guided']['discharge']['rmse']:.2f}, data={scores['data_only']['discharge']['rmse']:.2f}, persistence={scores['persistence']['discharge']['rmse']:.2f}", flush=True)
    transfer = {'directions': []} if args.skip_transfer else run_transfer(states, args, output)
    created_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    data = {'schema_version': 1, 'created_at': created_at, 'artifact_root': output.relative_to(ROOT).as_posix(), 'title': '大河研究室 · River Learning Lab',
            'data_status': source_status, 'model_label': '物理引导图时序网络', 'protocol': river_exports[0]['protocol'],
            'sources': sources, 'rivers': river_exports, 'transfer': transfer,
            'limitations': ['默认实验使用合成数据，河宽、坡度、糙率和泥沙参数为未校准的情景假设。',
                            '预测针对示意河段的水深，不是监测站相对基准的水位；预测起点为输入序列的末时刻。',
                            '物理正则使用矩形断面曼宁关系和非负约束，没有求解完整圣维南或 Exner 方程。',
                            '90%区间是验证残差的经验范围，需结合独立测试覆盖率阅读，不保证未来覆盖。',
                            '迁移结果仅表示这组模拟序列和随机种子下的表现；源河预训练增加计算，验证标签另计。',
                            '输入中没有未来降雨，六小时预测不代表长期气候预测。'],
            'comparison': {'dimensions': [
                {'label': '学习重点', 'yellow_river': '水沙耦合、泥沙滞后与河床变化', 'mississippi': '支流汇流、河道工程与泛滥平原交换'},
                {'label': '共享知识', 'yellow_river': '洪峰传播、时间记忆、物理量纲', 'mississippi': '洪峰传播、时间记忆、物理量纲'},
                {'label': '重新校准', 'yellow_river': '断面、水位基准、调度与含沙关系', 'mississippi': '断面、水位基准、汇流与回水关系'}],
                'lessons': [{'title': '共享动力过程，保留河流差异', 'body': '共享图时序网络权重，河流独立使用输入归一化与情景水力参数；迁移效果通过双向实验检验。'},
                            {'title': '误差下降需要对照证据', 'body': '同时比较物理引导、相同网络的纯数据学习和持续性基线，并呈现全部方法的实测实验误差。'}]},
            'docs': {'study': '../docs/multi_river_study.md', 'comparison': '../docs/yellow_mississippi_comparison.md',
                     'results': '../docs/multi_river_results.md', 'publishing': '../docs/publishing_guide.md'}}
    write_json(output / 'website_data.json', data)
    assets = ROOT / 'website' / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    (assets / 'river_lab_data.js').write_text('window.RIVER_LAB_DATA = ' + json.dumps(data, ensure_ascii=False, allow_nan=False).replace('</', '<\\/') + ';\n', encoding='utf-8')
    write_report(data, ROOT / 'docs' / 'multi_river_results.md')
    manifest = {'created_at': created_at, 'data_status': source_status, 'files': {}}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name != 'manifest.json':
            manifest['files'][str(path.relative_to(output))] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    write_json(output / 'manifest.json', manifest)
    print(f'Complete: {output}\nWebsite: {ROOT / "website" / "index.html"}', flush=True)


if __name__ == '__main__':
    main()
