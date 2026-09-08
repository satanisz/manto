# Monthly data input

Upload UTF-8 long-form CSV with these required columns:

```csv
series_id,period,available_at,value,title,unit
sales,2024-01-01,2024-02-10T09:00:00Z,120,Monthly sales,thousand PLN
sales,2024-01-01,2024-03-15T09:00:00Z,122,Monthly sales,thousand PLN
inflation,2024-01-01,2024-02-15T09:00:00Z,3.9,Inflation,percent
```

`period` is the month's first day at midnight UTC. `available_at` is the actual
instant the value became usable, not the period it describes. Times without a
timezone are interpreted as UTC. A later revision of the same period is a new
row with a distinct `available_at`. Duplicate series/period/publication triples
are rejected, even if their values agree. Releases before the observation month,
missing required cells, and infinite values are rejected.

`title` and `unit` are optional, but must be consistent within a series. The first
series is initially marked as a target for display; the analysis request can select
any series as Y. Sales, margins, volumes, and financial indices use the same format.
Missing months are retained as gaps; importing never interpolates or backfills.
User-supplied publication dates are an assertion by the uploader and must be checked
against the source to make historical forecasts credible.

Snapshots store all vintages, the series catalog, and provenance in canonical JSON.
The SHA-256 filename identifies exact content. Reload checks both hash and schema;
snapshots cannot be edited in place or renamed without invalidating the digest.

## Synthetic demo

The built-in dataset contains 156 months of fictional positive sales and ten
fictional explanatory variables. Monthly changes in sales depend on the preceding
month's simulated inflation, demand, and marketing activity. All observations are
available at the end of their month. This is a reproducible software demonstration,
not evidence of economic relationships. No API key or internet access is required.

## NBP connector

`fetch_nbp_monthly(code, start, end)` retrieves table-A daily mid FX rates and computes
their arithmetic mean for each completed calendar month fully inside the requested
date range. Units are PLN per foreign currency unit. Daily rates are not weighted
by calendar days. An incomplete start or end month is omitted; the current month
is also omitted. The monthly aggregate is available at that month's final instant.

The connector uses HTTPS with bounded timeouts and batches of at most 93 inclusive
days. A 404 denotes absent observations. Missing entire months, malformed responses,
and network errors produce explicit errors. It returns an FX predictor dataset,
not inflation, GDP, or WIG20 data. The current public archive does not establish
historical revision vintages. See the [official NBP API documentation](https://api.nbp.pl/en.html).
