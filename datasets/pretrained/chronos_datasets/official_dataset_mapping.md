# Official Chronos-2 Dataset Mapping

The table below maps the official Chronos-2 training dataset names shown in the report image to configs available in `autogluon/chronos_datasets`.

## Found configs

| Official dataset name | `autogluon/chronos_datasets` config(s) |
| --- | --- |
| Electricity | `electricity_15min`, `monash_electricity_hourly`, `monash_electricity_weekly` |
| KDD Cup (2018) | `monash_kdd_cup_2018` |
| M4 (Daily) | `m4_daily` |
| M4 (Hourly) | `m4_hourly` |
| M4 (Monthly) | `m4_monthly` |
| M4 (Weekly) | `m4_weekly` |
| Mexico City Bikes | `mexico_city_bikes` |
| Pedestrian Counts | `monash_pedestrian_counts` |
| Solar | `solar`, `solar_1h` |
| Taxi | `taxi_30min`, `taxi_1h` |
| Uber TLC | `uber_tlc_hourly`, `uber_tlc_daily` |
| USHCN | `ushcn_daily` |
| Weatherbench | `weatherbench_daily`, `weatherbench_weekly`, `weatherbench_hourly_*` configs |
| Wiki | `wiki_daily_100k` |
| Wind Farms | `wind_farms_daily`, `wind_farms_hourly` |
| Temperature-Rain | `monash_temperature_rain` |
| London Smart Meters | `monash_london_smart_meters` |

## Not found in `autogluon/chronos_datasets`

- Alibaba Cluster Trace (2018)
- Azure VM Traces (2017)
- Borg Cluster Data (2011)
- LargeST (2017)
- Q-Traffic
- Buildings 900K

## Partial frequency coverage

- Electricity: the official table lists `15min`, `1H`, `1W`, and `1D`; no clearly matching daily Electricity config was found.
- USHCN: the official table lists `1D` and `1W`; only `ushcn_daily` was found.
- Wiki: the official table lists `1H`, `1D`, and `1W`; only `wiki_daily_100k` was found.
