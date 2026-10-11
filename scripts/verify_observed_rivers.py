#!/usr/bin/env python3
"""Verify original sources, no-leakage protocol, reloads and all published data.

This is a read-only audit. It never retrains models or rewrites source archives.
Single-day test metrics and historical recursive forecasts are checked as
different experiments; the latter have no evaluation labels in the site data.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.observed_rivers import daily_windows, load_sources, sha256
from src.models.observed_rivers import DischargeScaler, load_checkpoint, metrics, predict, recursive_forecast, summarize_metrics


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-6)


def assert_metrics(actual, expected):
    assert set(actual) == set(expected)
    for key,value in expected.items():
        if value is None:
            assert actual[key] is None, key
        else:
            close(actual[key], value)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def check_prediction_csv(path, dates, prediction, observed=None, origin=None):
    rows = read_csv(path)
    assert [row["date"] for row in rows] == dates, str(path)
    close([float(row["prediction"]) for row in rows], prediction)
    if observed is None:
        assert all(row["observed"] == "" for row in rows), str(path)
    else:
        close([float(row["observed"]) for row in rows], observed)
    if origin is not None:
        assert all(row["origin_date"] == origin for row in rows)
        assert [int(row["lead_days"]) for row in rows] == list(range(1,len(rows)+1))
    assert all(row["unit"] == "m³/s" for row in rows)
    return rows


def verify():
    torch.set_num_threads(1)
    output = ROOT/"experiments/observed_rivers"
    result = json.loads((output/"results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    assert result["schema_version"] == 2 and result["data_status"] == "observed"
    assert result["model_status"] == "trained_observed_daily"
    assert protocol["lookback_days"] == 30 and protocol["horizon_days"] == 1
    assert protocol["seeds"] == [42,43,44] and protocol["epochs"] == 30 and protocol["target_epochs"] == 30
    assert protocol["forecast_origin"] == "2021-12-31" and protocol["forecast_steps"] == 6
    assert protocol["forecast_mode"] == "recursive_historical_extension"
    assert json.loads((output/"protocol.json").read_text(encoding="utf-8")) == protocol
    declared_files = set()
    for record in manifest["files"]:
        path = (ROOT/record["path"]).resolve()
        assert path.is_relative_to(output.resolve()), record["path"]
        assert record["path"] not in declared_files
        declared_files.add(record["path"])
        assert path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"], record["path"]
    actual_files = {path.relative_to(ROOT).as_posix() for path in output.rglob("*") if path.is_file() and path.name!="manifest.json"}
    assert declared_files == actual_files, "Manifest omissions or unexpected outputs"
    observations, provenance = load_sources(ROOT)
    assert result["provenance"] == provenance
    assert provenance["data_status"] == "observed" and provenance["model_status"] == "trained_observed_daily"
    hashes = {source["river_id"]:source["raw_sha256"] for source in provenance["sources"]}
    assert manifest["source_hashes"] == hashes
    usgs = observations["mississippi"]
    assert len(usgs) == 2557 and Counter(row["approval_status"] for row in usgs) == {"Approved":2557}
    assert sum(bool(row["qualifier"]) for row in usgs) == 41
    assert Counter(flag for row in usgs for flag in row["qualifier"].split(";") if flag) == {"ESTIMATED":39,"EQUIP":15,"FORCEINTERPOLATION":2}
    parts, scalers, source_saved = {}, {}, {}
    checkpoint_count = 0
    for river_id,records in observations.items():
        assert len(records) == 2557 and records[0]["date"] == "2015-01-01" and records[-1]["date"] == "2021-12-31"
        river = result["rivers"][river_id]
        assert river["observations"] == [{"date":row["date"],"discharge":row["discharge"]} for row in records]
        processed = read_csv(ROOT/river["downloads"]["data"])
        assert len(processed) == len(records)
        for row,expected in zip(processed,records):
            assert row["date"] == expected["date"] and float(row["discharge"]) == expected["discharge"]
            assert row["unit"] == "m³/s" and row["source_status"] == "observed"
            assert row["qualifier"] == expected["qualifier"] and row["approval_status"] == expected["approval_status"]
        parts[river_id],raw_train = daily_windows(records,provenance["common_period"],30)
        scalers[river_id] = DischargeScaler().fit(raw_train)
        assert river["scaler"] == scalers[river_id].to_dict()
        assert len(raw_train) == 1789
        # Independent raw-training boundary calculation, not overlapping windows.
        training_q = [row["discharge"] for row in records if row["date"] <= "2019-11-24"]
        close(river["scaler"]["mean"],np.mean(training_q))
        close(river["scaler"]["std"],np.std(training_q))
        for name,part in parts[river_id].items():
            declared = river["splits"][name]
            assert part["start"] == declared["start"] and part["end"] == declared["end"]
            assert len(part["dates"]) == declared["n_windows"] and part["skipped_windows"] == 0
            assert part["dates"][0] == declared["first_target"] and part["dates"][-1] == declared["last_target"]
            assert all(part["start"] <= day <= part["end"] for day in part["dates"])
        assert parts[river_id]["train"]["dates"][-1] == "2019-11-24"
        assert parts[river_id]["validation"]["dates"][-1] == "2020-12-12"
        assert parts[river_id]["test"]["dates"][0] == "2020-12-13"
        assert len(parts[river_id]["test"]["dates"]) == 384
        assert set(parts[river_id]["train"]["dates"]).isdisjoint(parts[river_id]["test"]["dates"])
        assert set(parts[river_id]["validation"]["dates"]).isdisjoint(parts[river_id]["test"]["dates"])
        test = parts[river_id]["test"]
        truth = test["y"].reshape(-1)
        persistent = test["x"][:,-1,0].astype(np.float64)
        seed_backtests,seed_forecasts = {},{}
        final_history = [row["discharge"] for row in records[-30:]]
        dates = [(date(2021,12,31)+timedelta(days=i+1)).isoformat() for i in range(6)]
        for run in river["runs"]:
            seed = run["seed"]
            model,scaler,saved = load_checkpoint(ROOT/run["checkpoint"])
            checkpoint_count += 1
            source_saved[river_id,seed] = saved
            assert saved["source"] == river_id and saved["target"] == river_id and saved["method"] == "gru" and saved["seed"] == seed
            assert saved["source_hashes"] == hashes and saved["protocol"] == protocol
            assert scaler.to_dict() == scalers[river_id].to_dict()
            assert saved["target_supervised_indices"] == [] and len(saved["history"]) == 30
            assert saved["optimizer_updates"] == math.ceil(len(parts[river_id]["train"]["dates"])/64)*30 == run["optimizer_updates"]
            assert bool((saved["model_state_dict"]["output.weight"].abs()>0).any()), "Untrained persistence-only output head"
            predicted = predict(model,test["x"],scaler).reshape(-1)
            forecasted = recursive_forecast(model,final_history,scaler,6)
            assert_metrics(run["metrics"],metrics(predicted,truth))
            check_prediction_csv(ROOT/run["backtest"],test["dates"],predicted,truth)
            check_prediction_csv(ROOT/run["forecast"],dates,forecasted,origin="2021-12-31")
            seed_backtests[seed],seed_forecasts[seed] = predicted,forecasted
        assert river["metrics"]["gru"] == summarize_metrics(river["runs"])
        baseline_runs = [{"metrics":metrics(persistent,truth)} for _ in protocol["seeds"]]
        assert river["metrics"]["persistence"] == summarize_metrics(baseline_runs)
        ensemble = np.mean(list(seed_backtests.values()),axis=0)
        ensemble_forecast = np.mean(list(seed_forecasts.values()),axis=0)
        assert_metrics(river["ensemble_metrics"],metrics(ensemble,truth))
        assert [row["date"] for row in river["backtest"]] == test["dates"]
        close([row["prediction"] for row in river["backtest"]],ensemble)
        close([row["observed"] for row in river["backtest"]],truth)
        close([row["persistence"] for row in river["backtest"]],persistent)
        assert [row["date"] for row in river["forecast"]] == dates
        close([row["prediction"] for row in river["forecast"]],ensemble_forecast)
        assert all(row["observed"] is None and row["origin_date"] == "2021-12-31" for row in river["forecast"])
        for seed in protocol["seeds"]:
            close([row["predictions_by_seed"][str(seed)] for row in river["backtest"]],seed_backtests[seed])
            close([row["predictions_by_seed"][str(seed)] for row in river["forecast"]],seed_forecasts[seed])
        check_prediction_csv(ROOT/river["downloads"]["backtest"],test["dates"],ensemble,truth)
        check_prediction_csv(ROOT/river["downloads"]["forecast"],dates,ensemble_forecast,origin="2021-12-31")
        all_q = np.asarray([row["discharge"] for row in records],dtype=np.float64)
        for name,value in {"mean":all_q.mean(),"min":all_q.min(),"max":all_q.max(),"std":all_q.std(),"cv":all_q.std()/all_q.mean()}.items():
            close(river["stats"][name],value)
        for row in river["seasonality"]:
            close(row["mean"],np.mean([record["discharge"] for record in records if int(record["date"][5:7]) == row["month"]]))
    for direction_id,direction in result["transfer"].items():
        source,target = direction["source"],direction["target"]
        assert direction_id == source+"_to_"+target and direction["backtest_seed"] == 42
        test = parts[target]["test"]
        truth = test["y"].reshape(-1)
        selected = list(range(0,len(parts[target]["train"]["dates"]),5))
        assert direction["target_label_indices"] == selected and len(selected) == 352
        assert direction["target_label_dates"] == [parts[target]["train"]["dates"][index] for index in selected]
        assert direction["n_target_train_windows"] == 1759 and direction["n_target_labeled_windows"] == 352
        assert [row["date"] for row in direction["backtest"]] == test["dates"]
        budgets = {}
        for method,details in direction["methods"].items():
            for run in details["runs"]:
                seed = run["seed"]
                if method == "persistence":
                    predicted = test["x"][:,-1,0].astype(np.float64)
                    assert run["checkpoint"] is None and run["optimizer_updates"] == 0 and run["n_target_training_labels"] == 0
                else:
                    model,scaler,saved = load_checkpoint(ROOT/run["checkpoint"])
                    checkpoint_count += 1
                    assert saved["source"] == source and saved["target"] == target and saved["method"] == method and saved["seed"] == seed
                    assert saved["source_hashes"] == hashes and saved["protocol"] == protocol
                    assert scaler.to_dict() == scalers[target].to_dict()
                    assert saved["target_supervised_indices"] == ([] if method=="zero_shot" else selected)
                    assert len(saved["history"]) == (0 if method=="zero_shot" else 30)
                    assert run["n_target_training_labels"] == (0 if method=="zero_shot" else 352)
                    assert saved["optimizer_updates"] == run["optimizer_updates"] == (0 if method=="zero_shot" else 180)
                    if method=="zero_shot":
                        for key,value in saved["model_state_dict"].items():
                            torch.testing.assert_close(value,source_saved[source,seed]["model_state_dict"][key],rtol=0,atol=0)
                    else:
                        budgets[method,seed] = saved["optimizer_updates"]
                    predicted = predict(model,test["x"],scaler).reshape(-1)
                assert_metrics(run["metrics"],metrics(predicted,truth))
                check_prediction_csv(ROOT/run["backtest"],test["dates"],predicted,truth)
                if seed==42:
                    close([row[method] for row in direction["backtest"]],predicted)
            assert details["summary"] == summarize_metrics(details["runs"])
        for seed in protocol["seeds"]:
            assert budgets["target_only",seed] == budgets["fine_tuned",seed]
        target_error = direction["methods"]["target_only"]["summary"]["rmse"]["mean"]
        fine_error = direction["methods"]["fine_tuned"]["summary"]["rmse"]["mean"]
        close(direction["error_change_pct"],100*(fine_error/target_error-1))
        aggregate_csv = read_csv(ROOT/direction["downloads"]["backtest"])
        assert [row["date"] for row in aggregate_csv] == test["dates"]
        for method in direction["methods"]:
            close([float(row[method]) for row in aggregate_csv],[row[method] for row in direction["backtest"]])
    assert checkpoint_count == 24
    assert len(list((output/"models").glob("*.pt"))) == 24
    assert json.loads((output/"website_data.json").read_text(encoding="utf-8")) == result
    embedded = (ROOT/"website/assets/observed_river_data.js").read_text(encoding="utf-8")
    assert json.loads(embedded.split("window.OBSERVED_RIVERS_DATA = ",1)[1].strip().removesuffix(";")) == result
    assert len({row["path"] for row in result["artifacts"]}) == len(result["artifacts"])
    for artifact in result["artifacts"]:
        path = (ROOT/artifact["path"]).resolve()
        assert path.is_relative_to(output.resolve()) and path.is_file(), artifact["path"]
    for path in result["downloads"]["source_code"]:
        assert (ROOT/path).is_file(), path
    print(f"PASS: {len(declared_files)} observed artifacts hashed, {checkpoint_count} models reloaded; source units/flags, causal splits, training-only scalers, all seed/ensemble CSV metrics and historical recursive forecasts reproduced.")
    for river_id,river in result["rivers"].items():
        score = river["metrics"]["gru"]["rmse"]
        baseline = river["metrics"]["persistence"]["rmse"]["mean"]
        print(f"{river_id}: GRU RMSE {score['mean']:.6f} ± {score['std']:.6f} m³/s; persistence {baseline:.6f}; ensemble {river['ensemble_metrics']['rmse']:.6f}")
    for direction_id,direction in result["transfer"].items():
        print(f"{direction_id}: fine-tuned vs target-only RMSE change {direction['error_change_pct']:+.6f}%")


if __name__ == "__main__":
    verify()
