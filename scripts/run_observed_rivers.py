#!/usr/bin/env python3
"""Train real two-station forecasts and bidirectional daily-flow transfer.

Uses frozen attributed source archives only. Existing derived outputs are
regenerated; raw files are never modified. Test targets are held out from
normalization, fitting and checkpoint selection. No synthetic fallback.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.observed_rivers import daily_windows, load_sources, sha256
from src.models.observed_rivers import (PROTOCOL_VERSION, DailyDischargeGRU, DischargeScaler,
                                       metrics, predict, recursive_forecast, summarize_metrics)

OUTPUT = ROOT / "experiments/observed_rivers"
MODEL_CONFIG = {"hidden_size": 16}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty CSV export")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(train_part, validation, scaler, seed, epochs, initial=None):
    seed_everything(seed)
    model = DailyDischargeGRU(**MODEL_CONFIG)
    if initial is not None:
        model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    tx = torch.as_tensor(scaler.transform(train_part["x"]), dtype=torch.float32)
    ty = torch.as_tensor(scaler.transform(train_part["y"]), dtype=torch.float32)
    vx = torch.as_tensor(scaler.transform(validation["x"]), dtype=torch.float32)
    vy = torch.as_tensor(scaler.transform(validation["y"]), dtype=torch.float32)
    best_loss, best_state, history, updates = float("inf"), None, [], 0
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.
        for ids in torch.randperm(len(tx)).split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(tx[ids])-ty[ids]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite observed model loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            updates += 1
            loss_sum += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            val_loss = float((model(vx)-vy).square().mean())
        history.append({"epoch":epoch+1,"train_mse_normalized":loss_sum/len(tx),"validation_mse_normalized":val_loss})
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), history, updates


def save_checkpoint(path, model, scaler, source, target, method, seed, protocol, history, updates, indices, source_hashes):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"protocol_version":PROTOCOL_VERSION,"data_status":"observed","variable":"discharge","unit":"m³/s",
                "model_config":MODEL_CONFIG,"model_state_dict":model.state_dict(),"scaler":scaler.to_dict(),"source":source,"target":target,
                "method":method,"seed":seed,"protocol":protocol,"history":history,"optimizer_updates":updates,
                "target_supervised_indices":list(map(int,indices)),"source_hashes":source_hashes}, path)


def prediction_rows(dates, observations, prediction, seed, method, origin=None):
    return [{"date":day,"origin_date":origin if origin is not None else (date.fromisoformat(day)-timedelta(days=1)).isoformat(),
             "lead_days":i+1 if origin is not None else 1,"unit":"m³/s","method":method,"seed":seed,
             "prediction":float(prediction[i]),"observed":"" if observations is None else float(observations[i]),
             "prediction_status":"model_prediction","input_source_status":"observed","evaluation":"recursive_historical_extension" if origin is not None else "rolling_one_day_test_backtest"}
            for i,day in enumerate(dates)]


def aggregate_baseline(predicted, truth, seeds):
    runs = [{"seed":seed,"metrics":metrics(predicted,truth),"optimizer_updates":0} for seed in seeds]
    return {"summary":summarize_metrics(runs),"runs":runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--target-epochs", type=int, default=30)
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if args.epochs<1 or args.target_epochs<1 or len(seeds)<2 or len(set(seeds))!=len(seeds) or 42 not in seeds:
        raise ValueError("Positive epochs and distinct multiple seeds including 42 required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observations, provenance = load_sources(ROOT)
    period = provenance["common_period"]
    if period != {"start":"2015-01-01","end":"2021-12-31"}:
        raise ValueError("This published experiment requires the attributed 2015–2021 common period")
    sources = {source["river_id"]:source for source in provenance["sources"]}
    source_hashes = {river_id:source["raw_sha256"] for river_id,source in sources.items()}
    protocol = {"version":PROTOCOL_VERSION,"data_status":"observed","variable":"daily_mean_discharge","unit":"m³/s",
                "model":"single-station residual GRU, not a physics-informed model","model_config":MODEL_CONFIG,
                "lookback_days":30,"horizon_days":1,"seeds":seeds,"epochs":args.epochs,"target_epochs":args.target_epochs,
                "batch_size":64,"optimizer":"Adam","learning_rate":.002,"gradient_clip":1.,
                "split_fractions":[.70,.15,.15],"split_basis":"common chronological raw calendar dates",
                "normalization":"mean/std of each river's raw training period only; target full training-input statistics also used for zero-shot",
                "checkpoint_selection":"minimum normalized MSE on full validation labels, never test labels",
                "missing_data":"no interpolation/imputation; windows containing missing calendar days are skipped",
                "historical_context":"validation/test inputs may include previously observed days from earlier partitions; targets remain disjoint",
                "test_evaluation":"rolling single-day prediction with observed past context, not multi-year free-running forecast",
                "target_label_stride":5,"target_label_budget":"every fifth eligible training window; approximately 20% of target supervised LOSS windows, not 20% of observed inputs",
                "transfer_budget":"target-only/fine-tuned have same labeled windows, batches, epochs and updates; full target validation labels and training-input scaler outside reduced budget; source pretraining adds labels and compute",
                "source_train_cutoff":"2019-11-24","source_validation_cutoff":"2020-12-12","target_test_start":"2020-12-13",
                "nrmse_definition":"RMSE divided by target TEST observed range","std_definition":"sample standard deviation across seeds, ddof=1; not a confidence interval",
                "forecast_origin":"2021-12-31","forecast_steps":6,"forecast_mode":"recursive_historical_extension",
                "forecast_note":"2022-01-01 through 2022-01-06 from the archived 2021-12-31 cutoff; not a current 2026 forecast; no future observed labels supplied",
                "nonnegativity":"learned physical predictions clipped at zero","daily_dates":"native source calendar labels retained, not simultaneous UTC measurements",
                "seasonality":"descriptive monthly mean over the common period, not a model input or training normalization statistic"}
    published_provenance = copy.deepcopy(provenance)
    published_provenance.update({"data_status":"observed","model_status":"trained_observed_daily","publication_note":"Real daily discharge models for Lijin and St. Louis only. Legacy synthetic experiments are not used by this site. Forecasts extend a historical cutoff and are not live operational forecasts."})
    result = {"schema_version":2,"data_status":"observed","model_status":"trained_observed_daily","generated_at":datetime.now(timezone.utc).isoformat(),
              "provenance":published_provenance,"protocol":protocol,"rivers":{},"transfer":{},"artifacts":[],
              "downloads":{"results":"experiments/observed_rivers/results.json","protocol":"experiments/observed_rivers/protocol.json","manifest":"experiments/observed_rivers/manifest.json","sources":"experiments/observed_rivers/provenance.json",
                           "source_code":["src/data/observed_rivers.py","src/models/observed_rivers.py","scripts/run_observed_rivers.py","scripts/predict_observed_river.py","scripts/verify_observed_rivers.py"]},
              "environment":{"python":sys.version.split()[0],"torch":str(torch.__version__),"numpy":str(np.__version__),"device":"CPU","deterministic_algorithms":True,"threads":1}}
    parts, scalers, source_states = {}, {}, {}
    for river_id, records in observations.items():
        parts[river_id], raw_train = daily_windows(records, period, protocol["lookback_days"])
        scalers[river_id] = DischargeScaler().fit(raw_train)
        source = sources[river_id]
        values = np.asarray([record["discharge"] for record in records],dtype=np.float64)
        river = {"name":source["river_name"],"station":source["station"],"station_id":source.get("station_id","lijin"),"frequency":"daily","unit":"m³/s","period":dict(period),
                 "observations":[{"date":record["date"],"discharge":record["discharge"]} for record in records],
                 "stats":{"n":len(records),"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),"std":float(values.std()),"cv":float(values.std()/values.mean()) if values.mean() else None},
                 "seasonality":[{"month":month,"mean":float(np.mean([record["discharge"] for record in records if int(record["date"][5:7])==month]))} for month in range(1,13)],
                 "splits":{name:{"start":part["start"],"end":part["end"],"n_windows":len(part["dates"]),"first_target":part["dates"][0],"last_target":part["dates"][-1],"skipped_windows":part["skipped_windows"]} for name,part in parts[river_id].items()},
                 "scaler":scalers[river_id].to_dict(),"runs":[],"backtest":[],"forecast":[],"metrics":{},"downloads":{"data":f"experiments/observed_rivers/data/{river_id}.csv","backtest":f"experiments/observed_rivers/backtests/{river_id}_ensemble.csv","forecast":f"experiments/observed_rivers/forecasts/{river_id}_ensemble.csv","models":[]}}
        write_csv(ROOT/river["downloads"]["data"],[{"date":record["date"],"station_id":river["station_id"],"discharge":record["discharge"],"unit":"m³/s","source_status":"observed","qualifier":record["qualifier"],"approval_status":record["approval_status"]} for record in records])
        test = parts[river_id]["test"]
        truth = test["y"].reshape(-1)
        persistent = test["x"][:,-1,0].astype(np.float64)
        seed_backtests, seed_forecasts = {}, {}
        final_history = np.asarray([record["discharge"] for record in records[-30:]],dtype=np.float64)
        if records[-1]["date"] != protocol["forecast_origin"] or any((date.fromisoformat(records[-30+i]["date"])-date.fromisoformat(records[-30]["date"])).days!=i for i in range(30)):
            raise ValueError("Forecast needs the last 30 consecutive observed days ending at the explicit cutoff")
        forecast_dates = [(date.fromisoformat(protocol["forecast_origin"])+timedelta(days=i+1)).isoformat() for i in range(6)]
        for seed in seeds:
            print(f"Training {river_id} seed {seed}",flush=True)
            model,history,updates = train(parts[river_id]["train"],parts[river_id]["validation"],scalers[river_id],seed,args.epochs)
            source_states[river_id,seed] = copy.deepcopy(model.state_dict())
            checkpoint = f"experiments/observed_rivers/models/{river_id}_gru_seed{seed}.pt"
            save_checkpoint(ROOT/checkpoint,model,scalers[river_id],river_id,river_id,"gru",seed,protocol,history,updates,[],source_hashes)
            predicted = predict(model,test["x"],scalers[river_id]).reshape(-1)
            forecasted = recursive_forecast(model,final_history,scalers[river_id],6)
            seed_backtests[seed], seed_forecasts[seed] = predicted,forecasted
            backtest_path = f"experiments/observed_rivers/backtests/{river_id}_seed{seed}.csv"
            forecast_path = f"experiments/observed_rivers/forecasts/{river_id}_seed{seed}.csv"
            write_csv(ROOT/backtest_path,prediction_rows(test["dates"],truth,predicted,seed,"gru"))
            write_csv(ROOT/forecast_path,prediction_rows(forecast_dates,None,forecasted,seed,"gru",protocol["forecast_origin"]))
            river["runs"].append({"seed":seed,"metrics":metrics(predicted,truth),"checkpoint":checkpoint,"backtest":backtest_path,"forecast":forecast_path,"optimizer_updates":updates})
            river["downloads"]["models"].append({"seed":seed,"path":checkpoint})
        ensemble = np.mean(list(seed_backtests.values()),axis=0)
        ensemble_forecast = np.mean(list(seed_forecasts.values()),axis=0)
        river["metrics"] = {"gru":summarize_metrics(river["runs"]),"persistence":aggregate_baseline(persistent,truth,seeds)["summary"]}
        river["ensemble_metrics"] = metrics(ensemble,truth)
        river["backtest"] = [{"date":day,"observed":float(truth[i]),"prediction":float(ensemble[i]),"persistence":float(persistent[i]),"predictions_by_seed":{str(seed):float(seed_backtests[seed][i]) for seed in seeds}} for i,day in enumerate(test["dates"])]
        river["forecast"] = [{"date":day,"prediction":float(ensemble_forecast[i]),"observed":None,"origin_date":protocol["forecast_origin"],"lead_days":i+1,"predictions_by_seed":{str(seed):float(seed_forecasts[seed][i]) for seed in seeds}} for i,day in enumerate(forecast_dates)]
        write_csv(ROOT/river["downloads"]["backtest"],[{"date":row["date"],"observed":row["observed"],"prediction":row["prediction"],"persistence":row["persistence"],"unit":"m³/s","evaluation":"rolling_one_day_test_backtest",**{f"seed{seed}":row["predictions_by_seed"][str(seed)] for seed in seeds}} for row in river["backtest"]])
        write_csv(ROOT/river["downloads"]["forecast"],[{"date":row["date"],"origin_date":row["origin_date"],"lead_days":row["lead_days"],"prediction":row["prediction"],"observed":"","unit":"m³/s","evaluation":"recursive_historical_extension",**{f"seed{seed}":row["predictions_by_seed"][str(seed)] for seed in seeds}} for row in river["forecast"]])
        result["rivers"][river_id] = river
    for source,target in [("yellow_river","mississippi"),("mississippi","yellow_river")]:
        direction_id = source+"_to_"+target
        target_train,validation,test = parts[target]["train"],parts[target]["validation"],parts[target]["test"]
        indices = np.arange(0,len(target_train["dates"]),5,dtype=int)
        labeled = {"x":target_train["x"][indices],"y":target_train["y"][indices]}
        truth = test["y"].reshape(-1)
        direction = {"source":source,"target":target,"backtest_seed":42,"target_label_indices":indices.tolist(),"target_label_dates":[target_train["dates"][i] for i in indices],"n_target_labeled_windows":len(indices),"n_target_train_windows":len(target_train["dates"]),"target_label_fraction":len(indices)/len(target_train["dates"]),"methods":{},"backtest":[],"downloads":{"backtest":f"experiments/observed_rivers/transfer/{direction_id}_seed42.csv"}}
        by_method = {}
        for method in ["target_only","zero_shot","fine_tuned","persistence"]:
            runs = []
            for seed in seeds:
                print(f"Transfer {direction_id} {method} seed {seed}",flush=True)
                checkpoint = None
                if method=="persistence":
                    predicted = test["x"][:,-1,0].astype(np.float64)
                    updates = 0
                else:
                    initial = source_states[source,seed] if method in ("zero_shot","fine_tuned") else None
                    if method=="zero_shot":
                        seed_everything(seed)
                        model = DailyDischargeGRU(**MODEL_CONFIG)
                        model.load_state_dict(initial)
                        history,updates = [],0
                    else:
                        model,history,updates = train(labeled,validation,scalers[target],seed,args.target_epochs,initial)
                    predicted = predict(model,test["x"],scalers[target]).reshape(-1)
                    checkpoint = f"experiments/observed_rivers/models/{direction_id}_{method}_seed{seed}.pt"
                    save_checkpoint(ROOT/checkpoint,model,scalers[target],source,target,method,seed,protocol,history,updates,[] if method=="zero_shot" else indices,source_hashes)
                csv_path = f"experiments/observed_rivers/transfer/{direction_id}_{method}_seed{seed}.csv"
                write_csv(ROOT/csv_path,prediction_rows(test["dates"],truth,predicted,seed,method))
                runs.append({"seed":seed,"metrics":metrics(predicted,truth),"checkpoint":checkpoint,"backtest":csv_path,"optimizer_updates":updates,"n_target_training_labels":0 if method in ("zero_shot","persistence") else len(indices)})
                if seed==42:
                    by_method[method] = predicted
            direction["methods"][method] = {"summary":summarize_metrics(runs),"runs":runs}
        direction["backtest"] = [{"date":day,"observed":float(truth[i]),**{method:float(prediction[i]) for method,prediction in by_method.items()}} for i,day in enumerate(test["dates"])]
        target_error = direction["methods"]["target_only"]["summary"]["rmse"]["mean"]
        fine_error = direction["methods"]["fine_tuned"]["summary"]["rmse"]["mean"]
        direction["error_change_pct"] = 100*(fine_error/target_error-1) if target_error>0 else None
        write_csv(ROOT/direction["downloads"]["backtest"],direction["backtest"])
        result["transfer"][direction_id] = direction
    write_json(OUTPUT/"protocol.json",protocol)
    write_json(OUTPUT/"provenance.json",published_provenance)
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name not in ("results.json","website_data.json","manifest.json"):
            kind = "checkpoint" if path.suffix==".pt" else "raw_source" if path.parent.name=="raw" else "data" if path.parent.name=="data" else "protocol" if path.name=="protocol.json" else "provenance" if path.name=="provenance.json" else "prediction_csv"
            result["artifacts"].append({"path":path.relative_to(ROOT).as_posix(),"type":kind})
    result["artifacts"] += [{"path":"experiments/observed_rivers/results.json","type":"results"},{"path":"experiments/observed_rivers/manifest.json","type":"manifest"}]
    write_json(OUTPUT/"results.json",result)
    write_json(OUTPUT/"website_data.json",result)
    asset = ROOT/"website/assets/observed_river_data.js"
    asset.parent.mkdir(parents=True,exist_ok=True)
    asset.write_text("// Trained observed daily discharge models; no synthetic fallback.\nwindow.OBSERVED_RIVERS_DATA = "+json.dumps(result,ensure_ascii=False,separators=(",",":"),allow_nan=False)+";\n",encoding="utf-8")
    files = [{"path":path.relative_to(ROOT).as_posix(),"sha256":sha256(path),"bytes":path.stat().st_size} for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name!="manifest.json"]
    write_json(OUTPUT/"manifest.json",{"schema_version":2,"data_status":"observed","model_status":"trained_observed_daily","files":files,"source_hashes":source_hashes})
    print("Training complete: 24 checkpoints and genuine observed daily forecasts/transfer results.",flush=True)
    for direction_id,direction in result["transfer"].items():
        print(direction_id,"fine-tuned RMSE change %:",direction["error_change_pct"],flush=True)


if __name__ == "__main__":
    main()
