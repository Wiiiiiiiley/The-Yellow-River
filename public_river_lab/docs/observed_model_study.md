# Yellow–Mississippi model and comparative study

This release models daily mean discharge at two stations: Yellow River at Lijin and Mississippi River at St. Louis. It combines next-day prediction, seasonal comparison, and bidirectional transfer learning. The model code and weights are downloadable under the scoped MIT license in `LICENSE.observed-models`; dataset licenses and attribution are documented separately in [the source note](website_data_status.md).

## Forecasting task

A small residual GRU takes the previous 30 daily discharge values and predicts the following day's discharge. Its output is a learned correction to the latest input value. A persistence baseline simply predicts that the next day equals the latest observation. Both are evaluated on the same dates in m³/s.

The common study period is 2015-01-01 through 2021-12-31. The original sources are preserved; processed files sort dates, filter this interval and convert USGS flow units. Normalization uses the training period only. No rainfall, dam-operation, power-output, channel geometry or sediment variable is used in this model.

## Chronological evaluation

Raw dates are divided 70% training, 15% validation and 15% test, without shuffling dates across partitions. A window belongs to the partition of its prediction target; preceding observations can supply its input history. The first target follows the initial 30-day context. Missing or discontinuous windows are excluded, not filled.

Validation selects a training checkpoint; test observations are used only for evaluation. Runs use seeds 42, 43 and 44. The result JSON retains each run, and summary metrics report the mean and sample standard deviation across those three runs. This standard deviation describes initialization variability, not a confidence or forecast interval.

MAE and RMSE use m³/s. NSE compares squared error with variation around the test-period mean. NRMSE divides RMSE by the observed test-period range. Cross-river absolute RMSE is affected by the stations' different flow scales, so seasonal profiles and scale-normalized measures are useful alongside it.

The rolling next-day backtest uses the observed preceding days at each test date. The six-day extension instead feeds predictions back into the model, with an issue date of 2021-12-31. It is a dated historical recursive forecast, not a present-day forecast. No future observation is inserted into that extension; rolling one-day errors do not establish six-day forecast accuracy.

## Comparative and transfer learning

Seasonality is the mean discharge for each calendar month across the common study period, divided by each station's overall mean for the normalized comparison. The coefficient of variation is standard deviation divided by mean. These are descriptive station statistics, not evidence that one river's dams caused a particular observed pattern.

Each transfer direction uses the same model architecture and three seeds:

- **Target only:** train from scratch on every fifth target-training window.
- **Zero shot:** use the source-pretrained weights without supervised target fitting, but with the target training-period scaler.
- **Fine tuned:** initialize with the source-pretrained weights and train on exactly the same sparse target windows as target only.
- **Persistence:** use the latest target observation as a benchmark.

Source pretraining uses the full source-training partition. Target-only and fine-tuned runs have the same target supervision budget and number of optimizer updates. Target training observations are still needed for normalization and window histories; validation labels select fitted checkpoints. Therefore, “zero shot” means no supervised target parameter fitting, not zero access to target records.

Improvement compares fine-tuned RMSE against target-only RMSE within one direction. A negative improvement means transfer worsened that metric. The experiment reports both directions and the baselines without assuming that transfer must help.

## Reproduce and reuse

From the project or portable publication root:

```bash
python3 -m pip install -r requirements_observed.txt
python3 scripts/run_observed_rivers.py
python3 scripts/verify_observed_rivers.py
python3 -m unittest discover -s tests -p 'test_*.py'
```

Default training uses 30 epochs, 30 input days, seeds 42/43/44, a target-window stride of 5, and a six-day recursive extension. Training regenerates the observed experiment outputs; preserve a copy before experimenting with different settings. The archive input files are not rewritten. Exact configuration, split dates, checkpoint metadata, metrics and paths are saved with the results.

For a saved checkpoint and a chronological CSV with the expected daily discharge columns:

```bash
python3 scripts/predict_observed_river.py \
  --checkpoint experiments/observed_rivers/models/yellow_river_gru_seed42.pt \
  --input experiments/observed_rivers/data/yellow_river.csv \
  --steps 6 --output /tmp/yellow_forecast.csv
```

The input's final date defines the issue date. Providing newer observations permits a newer forecast; downloading an old checkpoint alone does not make a current forecast.

## Water-management research

Dam regulation, hydropower, sediment control and water allocation are comparison topics for extending this study. Testing their effects would require aligned reservoir-operation, rainfall, sediment, demand and energy records. The current discharge-only model does not estimate causal dam effects, hydropower production or sediment forecasts.
