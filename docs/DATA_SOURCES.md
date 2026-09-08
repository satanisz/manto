# Data-Source Assessment

Reviewed 2026-09-08 using official documentation. This is a sourcing proposal,
not a populated catalog or a completed connector validation. Exact dataset IDs,
dimensions, coverage, release timestamps, revision history, and reuse conditions
must be verified for each selected series in sprint 1.

## Recommended starting point

Start with Eurostat for monthly European/Polish macro series, NBP Web API for FX,
and an explicitly sourced WIG20 file import. Assess GUS DBW/SDP for national
indicators when Eurostat definitions do not match the question. Implement one macro
connector first and a second after the analytical core works. Avoid making the
first vertical slice depend on every provider.

| Source | Verified capability | Proposed use and unresolved work |
| --- | --- | --- |
| Eurostat | Public Statistics and SDMX APIs with metadata and dataset discovery | First macro connector; verify Polish monthly price/activity/labor series and exact adjustment/units |
| GUS DBW / SDP | Official access to statistical domains and published indicators | Assess national CPI, activity, wages and labor data; exact endpoint/series coverage is not yet checked |
| GUS BDL | API covering local/regional and national statistics, annual and short-term data | Useful when catalog coverage fits; do not assume every national monthly macro series is present |
| NBP Web API | Historical/current exchange rates and gold prices | EUR/PLN and USD/PLN; not a general inflation, policy-rate, or monetary-aggregate API |
| GPW Benchmark / approved market-data provider | Official WIG20 index page and historical-data service information | Verify downloadable history, refresh route, price-index definition and permitted use; no working history API established in this task |
| FRED / ALFRED | Economic series and historical vintages for supported series | Later global factors or a vintage-aware methodological fixture; not a presumed replacement for Polish source coverage |

Evidence: [Eurostat API guide](https://ec.europa.eu/eurostat/web/user-guides/data-browser/api-data-access/api-getting-started),
[GUS DBW](https://api.stat.gov.pl/Home/DBWApi),
[GUS SDP](https://api.stat.gov.pl/Home/SDPApi),
[GUS BDL](https://api.stat.gov.pl/Home/BdlApi),
[NBP Web API](https://api.nbp.pl/),
[WIG20 index page](https://gpwbenchmark.pl/karta-indeksu?isin=PL9999999987),
[GPW historical-data page](https://gpwbenchmark.pl/dane-historyczne),
and [St. Louis Fed's ALFRED explanation](https://www.stlouisfed.org/open-vault/2021/august/using-the-alfred-database).

## The main data limitation: historical availability

Eurostat states that its database contains the latest dataset version without
past-version history. Consequently, a historical forecast evaluated on downloaded
Eurostat data is not automatically a point-in-time backtest.
See [Eurostat web services](https://ec.europa.eu/eurostat/data/web-services).

For each source classify both publication-time fidelity and vintage fidelity:

1. `verified_point_in_time`: the values and their availability at historical origins
   can be reconstructed from release/vintage records.
2. `release_lag_proxy_latest_vintage`: publication timing is conservatively modeled,
   but revised historical values may remain. Label the analysis exploratory.
3. `availability_unknown`: suitable only for labeled historical association until
   a defensible availability contract exists.

Retain immutable downloads and newly observed revisions going forward. ALFRED can
provide vintage evidence for supported series, but coverage must be checked per
series; do not infer that all FRED or Polish series have complete release histories.
Future snapshots do not retroactively repair missing old vintages.

## Initial candidate concepts for review

These are hypotheses for a catalog, not verified series IDs or claims of predictive
power. Relevance is to the target `Y`, with pins respected and redundancy examined.

| Concept | Likely source to assess | Modeling note |
| --- | --- | --- |
| Polish consumer inflation | GUS or Eurostat | National CPI and HICP differ; distinguish rate from index level |
| Industrial production | Eurostat / GUS | Preserve volume, seasonal adjustment, and release delay |
| Retail sales volume | Eurostat / GUS | Do not mix nominal sales with a volume index |
| Unemployment | Eurostat / GUS | Harmonized and registered unemployment are different measures |
| Wages | GUS | Nominal/real and population definitions matter |
| Producer prices | Eurostat / GUS | Index level versus annual/monthly rate affects integration order |
| EUR/PLN | NBP Web API | Declare month-end or monthly-average aggregation |
| USD/PLN | NBP Web API | Potential redundancy with EUR/PLN; assess VIF within model |
| Euro-area industrial activity | Eurostat | External-demand hypothesis, not established causation |
| Euro-area inflation | Eurostat | Potential external price channel and overlap with domestic inflation |

Policy rates, money supply, confidence measures, commodities, and global equity
indices can extend the catalog after exact access routes are verified. The basic
NBP FX/gold API is not evidence that those other series are available there.
Quarterly GDP belongs in a later quarterly or explicitly mixed-frequency experiment.

Inflation-rate pinning does not authorize substituting CPI levels for cointegration.
Any semantic change is a visible user choice. A plausible I(1) level relationship
requires its own candidate-specific evidence, not a list chosen to pass a test.

## Series and snapshot contract

Every admitted series needs:

- Stable internal ID, provider ID/dimensions, English name, description and geography.
- Unit, frequency, calendar/timezone, seasonal-adjustment status, price basis if applicable.
- Observation period, availability timestamp or explicit timing assumption, retrieval timestamp.
- Vintage/revision identifier where supported, raw snapshot hash, provider provenance.
- Earliest/latest usable date, missingness, observed breaks, revision/freshness policy.
- Permitted transformations/aggregation, publication lag, source attribution and reuse notes.

Differentiate observation month, release date, and date the application retrieved
the record. Specify the forecast origin precisely, for example after a declared
month-end market close; an aggregate is unavailable until all required inputs
have been published. Align cointegration input periods separately from operational
forecast feature availability. Do not hide either timing choice in a join.

## Sprint 1 sourcing acceptance

Resolve one target and at least ten candidate concepts into verified catalog
entries or explicit blocked statuses. Fetch a small real sample, inspect its
dimensions and historical coverage, record publication/vintage limitations, and
establish the common monthly window. Do not invent identifiers for missing series.

If WIG20 history is not available, use a user-provided file with provenance or a
clearly labeled synthetic fixture to develop the workflow. Switching the business
target is a separate product choice. Production-like claims wait for actual target
data, suitable rights, and a defensible availability contract.
