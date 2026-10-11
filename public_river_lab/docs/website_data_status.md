# Data sources and preparation

Snapshot: 2026-10-10. The two stations share a 2015-01-01–2021-12-31 daily-discharge study period, with 2,557 days per station. Original downloads are retained unchanged; processed CSV files contain the study-period values in m³/s. Dates retain each provider's calendar labels rather than being interpreted as simultaneous UTC measurements.

## Yellow River, Lijin

Tao Li (2025), *Daily mean discharge and suspended sediment concentration from 2015 to 2021 at Lijin station*, [Mendeley Data, version 1](https://data.mendeley.com/datasets/vskktt3v3k/1), DOI [10.17632/vskktt3v3k.1](https://doi.org/10.17632/vskktt3v3k.1). The archive author attributes the original records to the Chinese Hydrological Yearbook. This project downloaded the public archive, not directly from the Yellow River Conservancy Commission.

License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please retain author, dataset title, DOI, license link and modification description when redistributing. Processing filters the common study period and selects discharge; it does not alter the original workbook.

The original XLSX contains 2,558 consecutive dates, including one extra day, 2022-01-01. Discharge is in m³/s; suspended sediment concentration is present but is not an input or prediction target for this model. The workbook does not specify its civil-day timezone convention or supply per-record instrument-quality flags.

## Mississippi River, St. Louis

[U.S. Geological Survey station USGS-07010000](https://waterdata.usgs.gov/monitoring-location/USGS-07010000), accessed through the [USGS Water Data API](https://api.waterdata.usgs.gov/ogcapi/v1/?f=html). The daily collection is filtered to discharge parameter `00060` and daily-mean statistic `00003`, for 2015–2021.

The API identifies these U.S. Government records as public-domain data; retain USGS attribution. The original JSON has 2,557 unique daily values in ft³/s. Processing sorts dates and multiplies discharge by `0.028316846592` to obtain m³/s; the original JSON keeps its native values and units.

All records are Approved. Provider flags cover 41 days: 39 ESTIMATED, including 15 also EQUIP, plus 2 FORCEINTERPOLATION. Values and flags are retained. No project interpolation, imputation or generated observation is substituted for a source value.

## Reproducibility

`experiments/observed_rivers/provenance.json` records original URLs, licenses, coverage and SHA-256 hashes. The model study describes preprocessing, chronological splits, training and evaluation. Sources do not license the project model code: the new model release has its own MIT license, while data retain the licenses above.

The website uses these two stations only. Legacy simulated experiments remain separate project records and are excluded from the static publication bundle.
