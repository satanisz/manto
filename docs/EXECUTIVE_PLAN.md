# Manto: Executive Plan

Version 0.1, 2026-09-08. Product roadmap. The sprint 6 demo is now implemented;
see [actual scope and limitations](DEMO_GUIDE.md). Later milestones remain proposed.
Numerical thresholds are initial product policies to validate, not universal statistical rules.

## 1. Product thesis

Manto turns a conversation about a time series into a reproducible analytical
experiment. A user describes a target, supplies preferred explanatory variables,
and controls the search breadth. Manto prepares data, compares small linear
models and justified extensions, identifies Pareto-efficient alternatives,
diagnoses weaknesses, and recommends a champion with evidence. Users can save
the analysis, inspect alternatives, replace a variable, and monitor an activated model.

The primary user understands the business question but may not understand
econometrics. The product must explain both the findings and their limitations.
It must support the honest result that no candidate improves on a simple baseline.
Association, predictive usefulness, and causation are distinct claims.

The system is agentic because it interprets an underspecified request, asks for
missing information, proposes constrained experiments, branches on evidence, and
revises recommendations with user input. Numerical calculations remain testable
Python functions. Agent count is not a product success criterion.

## 2. Confirmed requirements and terminology

The owner confirmed forecasting plus historical relationship analysis, monthly
data, and a one-month forecast horizon. Linear regression is the core method.
Cointegration is in scope. Seasonal and decomposition extensions must appear
explicitly in the model-building decision graph. All project artifacts are English;
conversation and prompt content may be Polish. Gemini is the chosen LLM provider.

| User concept | Precise interpretation |
| --- | --- |
| WIG20 or another series to explain | One dependent variable / target `Y` per experiment |
| Two, three, or at most four variables | Explanatory variables `X`; maximum four distinct external source series per model |
| A fixed variable, such as inflation | Mandatory inclusion, not a constant input value or frozen coefficient |
| Ten, twenty, or all variables | Candidate catalog subset from which models are generated |
| Preliminary training | Fitting and time-ordered validation of feasible specifications |
| Pareto best models | Non-dominated trade-offs across declared objectives, not one automatic winner |
| Champion | A policy-eligible model recommended using recorded evidence |
| Replace a variable | A new experiment branch linked to an immutable parent |
| Continuous monitoring | A persistent service checking data, scoring matured forecasts, and raising alerts |

Pinned variables count toward requested model size. From ten candidates including
inflation, exactly three variables give `C(10,3) = 120` combinations; pinning inflation
gives `C(9,2) = 36`. Generally the count is `C(n-p,k-p)` for `n` candidates, `p` pins,
and exactly `k` variables. "Up to three" searches sizes one through three instead.

The four-variable limit alone does not bound model complexity. Show external
series count, design-matrix columns, lag orders, seasonal terms, and estimated
parameter count separately. The initial OLS uses one term per external series.
ECM and seasonal extensions have a separate explicit parameter/sample budget;
they cannot hide dozens of parameters behind four variable names. Baselines are
exempt from pin/count constraints and are labeled as reference models.

Lag selection and transformations toward stationarity are explicitly required.
They are versioned modeling decisions, not hidden preprocessing defaults.

## 3. Working assumptions and unresolved decisions

| Decision | Working assumption | Resolution point |
| --- | --- | --- |
| First use case | Poland, with WIG20 as the example target | Sprint 1 |
| Target scale | Original user-facing Y; training-only transformations and inverse forecasts are internal, explicit decisions | Confirmed by the owner; optional overrides remain visible |
| Data | Public macro sources plus a target file import; verify WIG20 history access | Sprint 1 |
| Deployment | Local single-user Python application, modular monolith | Before UI implementation |
| UI | Streamlit and Plotly, separate from the analytical core | Sprint 2 dependency spike |
| LLM | Gemini API with a pinned supported model ID | Sprint 2 integration |
| Alerts | In-app inbox first; external delivery later | Before monitoring implementation |
| Promotion | Explicit user activation; automatic alerts and challenger proposals | Before monitoring implementation |
| Budget and team | One experienced full-time engineer with part-time statistical review | Planning assumption, not a commitment |

Still clarify target scale, available historical data, acceptable data/API costs,
local versus hosted use, and expected user count. Do not block a fixture-based
vertical slice on these choices; do block claims that depend on unavailable data.

## 4. User journey

1. Extract target, objective, frequency, horizon, model size, and pins from the
   conversation. Ask only for missing fields; do not repeat answered questions.
2. Resolve the target series and explain level versus return interpretations.
   Show the proposed experiment specification for review.
3. Query the available catalog. Propose candidates with coverage, source,
   publication timing, and a relevance hypothesis for `Y`. A relationship with
   pinned inflation alone is not a reason to select a variable; it may create VIF problems.
4. Support manual selection, replacement, exclusion, pinning, larger shortlists,
   and all eligible catalog entries. User choices are first-class inputs.
5. Show feasible combination count, data losses, transformations, model families,
   development/holdout dates, and compute estimate. Over-budget requests require
   a scope/budget choice, never silent truncation.
6. Prepare aligned data and evaluate candidates using the declared statistical
   branches. Record every eligible, rejected, failed, and skipped specification.
7. Compute the Pareto front. Run the complete detailed diagnostic suite for every
   Pareto model; paginate the UI instead of silently replacing the front with top-N.
8. Recommend a champion with trade-offs or report insufficient evidence. Keep all
   tested models accessible with their eligibility and diagnostic status.
9. Save/export the analysis, switch the displayed model, or branch a new experiment.
   Switching the view does not activate a different monitored model.
10. Activate a model, issue forecasts, and monitor outcomes. Recalibration produces
    a challenger with its own history and an explicit promotion decision.

## 5. Scope and ambition

This is an advanced analytical PoC, approaching a small decision-support product
with experiment management and MLOps. The hard parts are time-aware data,
statistically honest comparison, and reproducibility, not the chat interface.

Include one target per experiment, monthly one-step forecasts, a curated catalog,
small OLS models, a constrained cointegration/ECM branch, seasonal linear variants,
STL diagnostic decomposition, and an optional STL-based forecasting challenger.
Include persisted conversations, auditable decisions, reports, model comparison,
forecast issuance, monitoring, and controlled recalibration.

Defer unrestricted model search, neural forecasting, trading execution, causal
identification, arbitrary generated-code execution, mixed-frequency models,
multi-target VAR/VECM systems, enterprise tenancy, and automatic promotion.
An ARDL bounds-test branch for mixed integration orders and SARIMAX extensions are
documented future capabilities, not silently included in PoC acceptance.

## 6. Statistical design

### Data availability and leakage

Forecast `Y(t+1)` using only information available at origin `t`. Historical
contemporaneous association models must be labeled separately. Unknown future
predictors cannot enter an unconditional forecast; user-supplied future inputs
produce explicitly conditional scenarios.

Store observation period, publication/availability time, retrieval time, source,
unit, seasonal-adjustment status, and revision/vintage metadata. March inflation
published in April cannot enter a March-origin forecast. Later revisions must not
be presented as information available at an earlier origin.

Without historical vintages, snapshot data going forward and label historical
experiments as latest-vintage exploratory analysis. Conservative publication lags
do not eliminate revision bias. Such models can enter labeled shadow monitoring,
but cannot claim verified point-in-time historical performance.

Avoid quarterly-to-monthly interpolation in the initial PoC. Fit imputation,
scaling, transformations requiring estimation, and decomposition inside training
windows. Never backfill from future observations. Use common evaluation origins,
report missingness, and preserve skipped forecast origins as coverage failures.

### Stationarity and cointegration

The graph separates data characterization from residual diagnostics. For each
candidate specification, inspect the intended level series and permitted
transformations using plots and ADF, with KPSS as a complementary check. Record
trend/intercept specification, lag choice, sample size, and inconclusive results.
Non-rejection of a null is not proof that a series has a particular integration order.
See the [ADF reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)
and [KPSS reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html).

| Training evidence | Allowed branch | Interpretation |
| --- | --- | --- |
| Stationary target and regressors under the declared deterministic terms | OLS / lagged linear regression | Conditional association and forecast on that scale |
| A candidate long-run relation among plausibly I(1) level series | Engle-Granger cointegration test for that candidate subset | Test a specified long-run relation, not arbitrary correlation |
| Cointegration evidence and valid operational timing | Linear error-correction model (ECM), with short-run dynamics | Long-run relation plus adjustment; still not causality |
| No sufficient cointegration evidence | Regression in supported stationary changes/returns | Short-run relationship; no claimed equilibrium |
| Mixed I(0)/I(1) inputs | Conservative stationary transformation route in PoC | ARDL/bounds testing is a later explicit extension |
| I(2), inconsistent evidence, structural breaks, or insufficient sample | Review, exclude an unsupported specification, or return insufficient evidence | Do not force an ordinary levels regression |

Engle-Granger uses cointegration-specific critical values and assumes the tested
level variables are I(1). An ordinary ADF p-value on fitted residuals is not a
substitute. Use a dedicated [cointegration implementation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html).
Inflation as a rate and a price-index level are different series; never replace a
pinned inflation rate with a price index simply to obtain cointegration.

Cointegration is tested on economically aligned observation periods, with units
and release metadata retained. Operational ECM features must also be observable
at forecast time. If a contemporaneous equilibrium error cannot be observed,
use a predeclared available lag or mark the specification historical-only. Do not
manufacture a long-run relationship by forward-filling incompatible releases.

Estimate the long-run relation and short-run equation anew within each training
fold. An illustrative ECM is `delta_Y(t+1) = a + lambda * ECT(t) + short_run_terms + e`,
where `ECT(t) = Y(t) - b0 - sum(bj * Xj(t))` is usable only if all components are
available at that origin. Report adjustment-coefficient sign, magnitude,
uncertainty, and full dynamic stability rather than a simplistic sign-only pass.
The [statsmodels UECM documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.ardl.UECM.html)
provides a related implementation reference; the bounded two-step ECM is the initial design.

Freeze the routing algorithm before development evaluation. At each origin it
uses training evidence only. If a family becomes ineligible, record that outcome;
do not silently score it on an easier subset. The candidate is a versioned
forecasting procedure, and any approved fallback is part of its specification.

### Transformations toward stationarity and lag selection

D20 governs transformations and D19 governs lag construction/selection in the
decision graph. Preserve the original series, transformation recipe, and transformed
series separately. Candidate recipes include unchanged levels when justified,
log levels for strictly positive values, first differences, log differences/returns,
and seasonal differences when evidence supports a seasonal unit-root treatment.
A logarithm may stabilize scale/variance but does not by itself ensure stationarity.
Standardization does not remove a unit root either.

Choose a small permitted recipe set before scoring. Within each training window,
use ADF/KPSS, trend/seasonal metadata, and visual summaries to evaluate the selected
recipe. Record pre/post evidence. Do not repeatedly difference until a preferred
p-value appears: initially bound ordinary differencing to order one and seasonal
differencing to order one, and disallow their automatic combination. Conflicting
tests or apparent breaks yield an inconclusive/review state. A deterministic trend
and a stochastic trend need not imply the same remedy.

For a cointegrated ECM, retain levels for the long-run relation and build stationary
differences for the short-run component. Do not difference all series before testing
cointegration. For mixed integration orders, transform each series as appropriate;
do not difference already stationary inputs without a declared reason. If no supported
recipe gives a defensible specification, return an unsupported/inconclusive result.

For monthly forecasts, the initial external-series lag menu is 0, 1, 2, 3, 6, and
12 months, subject to data availability and sample/parameter budgets. Lag zero
means the same-origin observation and is allowed only when it is genuinely
available at that origin; it never means the target month's future observation.
The first OLS searches one selected lag per external series. Multiple distributed
lags and lagged `Y` terms are bounded extensions with visible parameter counts.
Use explicit timestamp formulas: an external lag `L` refers to `X(t-L)` at origin
`t`, while a conventional response lag `j` for `Y(t+1)` refers to `Y(t+1-j)`.
Thus response lag one is the last observed target `Y(t)`, not a future value.

Keep economic lag and publication lag distinct. A lag of two months describes
the modeled relationship; a release delay describes what was knowable. Persist
both rather than adding an unexplained shift. A feature record includes series ID,
transformation ID, economic lag, availability rule, and source snapshot. User-pinned
series remain mandatory, while pinning an exact lag/transform is an optional
stronger constraint. Infeasible pins trigger a visible choice, not silent substitution.

Use training-only cross-correlation/ACF/PACF as exploratory lag evidence, after
appropriate detrending/transformation; raw trend correlations are not a causal
lag detector. Rank lag recipes by inner chronological validation within the current
development training prefix, or evaluate a fixed predeclared grid as separate
candidate specifications. Record which mode is used. Never tune on that origin's
validation outcome or on the final holdout. Count the inner fits against the budget.

Trim the effective sample after lags, differencing, and seasonal operations before
checking adequacy. Compare candidates on common forecast origins. For a level
target, invert predicted changes using the known origin level; invert seasonal
differences using the required known seasonal anchor. A log inversion must state
whether it estimates a median or a mean; any bias correction is training-derived.
Persist inverse-transform rules and apply them to uncertainty estimates as well.
If inversion needs an unavailable anchor, the forecast cannot be issued.

### Seasonality and decomposition

Monthly data suggests checking period 12, not assuming seasonality exists. Inspect
training ACF, seasonal plots, periodograms, adjustment metadata, and stability
across training subwindows. Use at least three annual cycles as an initial
diagnostic minimum, with sufficient training data after any transformations.

The explicit branches are: no seasonal extension; linear regression with a small
predeclared Fourier basis; or STL decomposition for descriptive analysis with an
optional STL-plus-linear/AR forecasting challenger. Show why a branch was enabled
or skipped and charge additional terms against the complexity budget. Avoid
automatic eleven-dummy expansions or double adjustment of already adjusted data.

[STL](https://www.statsmodels.org/stable/generated/statsmodels.tsa.seasonal.STL.html)
decomposes a series; it does not itself define the future forecast. A challenger
must specify how the adjusted component is forecast and how the seasonal component
is extended, as illustrated by [STLForecast](https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.stl.STLForecast.html).
Every fold refits decomposition using training data only. Whole-history plots are
allowed as labeled retrospective illustrations, never as backtest features.

### Validation and metrics

Use chronological expanding/rolling training windows and an untouched final
holdout. Respect publication timing and any overlapping-label gap. Compare the
same origins and forecast target scale. This ordering is supported by the
[TimeSeriesSplit design](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

A provisional monthly design requires 120 usable observations, an initial
60-period training window, at least 24 one-step development outcomes, and a final
12-period holdout. Longer history expands development evaluation. Reassess adequacy
against estimated parameters and structural changes; shorter data enters an
explicit reduced-evidence mode or blocks the requested specification.

Use last-value/random-walk and seasonal-naive baselines for levels, and zero-return
for return targets. Report training R-squared/adjusted R-squared, out-of-sample
R-squared with reference definition, MAE, RMSE, baseline-relative error, VIF where
applicable, coverage, and block error variability. Undefined metrics remain
undefined. Constant targets and singular design matrices require explicit handling.

MAPE is optional: disable it for returns, zero-crossing targets, and near-zero
values under a declared threshold. Its instability is documented in the
[MAPE reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html).
Do not rank level and return errors together. When an internal difference model
predicts the same level target, reconstruct the level using values known at the
origin and score on that common scale. Persist the reconstruction method.

Freeze candidates, family routing, objectives, and policies before evaluation.
Choose the specification from development results, then open the holdout once to
audit that choice against the baseline. A failed holdout does not authorize
selecting a runner-up on the same data. Subsequent user edits are exploratory
branches needing fresh evidence for confirmatory claims; record holdout exposure.

### Pareto and champion policy

Within the OLS comparison group, minimize development MAE, maximum training-fold
predictor VIF, and standard deviation of development block MAE. VIF excludes the
intercept and concerns coefficient stability, not predictive accuracy. Keep
R-squared/RMSE visible without adding every correlated metric to the front.

For cross-family comparison, use a separate front minimizing development MAE,
block-error variability, and effective parameter count under family-specific
counting rules. VIF is family-specific supporting evidence: never replace
unavailable VIF with zero. Count the long-run and short-run estimated coefficients
in ECM complexity; for adaptive STL smoothers where comparable effective degrees
of freedom are unavailable, show the challenger separately until a documented
comparable complexity measure is implemented. It can be compared on forecast
errors, but is not silently inserted into this three-objective front.

Apply hard validity gates before Pareto. Dominance means no worse on all objectives
and better on at least one, with recorded numerical tolerance. Preserve tied
models. Every front member receives diagnostics. If diagnostic hard gates remove
members, recompute the front over the remaining candidates and diagnose newly
exposed members until stable; bound the work by the finite candidate set.

The rule engine proposes a default among eligible front members within 5% of best
development MAE, preferring lower error variability, then lower complexity, then
stable model ID. Record exact-zero handling. Diagnostic warnings influence claims
and promotion restrictions; do not simply maximize the number of p-values above 0.05.

Gemini proposes an eligible champion with evidence references. Preserve the rule
default and the LLM proposal separately when they differ; the user resolves the
choice before activation. Invalid IDs/claims get a bounded repair then a rule-based
template fallback. No eligible model or failure against baseline yields
`no_qualified_champion`; users can still inspect exploratory candidates.

### Detailed diagnostic report

For every Pareto model include coefficient interpretation, uncertainty assumptions,
actual-versus-predicted paths, forecast errors, residual timeline, residual-versus-
fitted plot, histogram, Q-Q plot, ACF/PACF, periodogram with units, and rolling
performance. ECM adds equilibrium-error and adjustment views; seasonal variants
add component and seasonal stability plots.

| Check | Meaning and limit |
| --- | --- |
| Shapiro-Wilk | Evidence against normal residuals; non-rejection does not establish normality |
| Residual ADF | Evidence against a unit root for declared settings; distinct from cointegration inference |
| Durbin-Watson | Descriptive first-order autocorrelation statistic; not a universal p-value gate |
| Breusch-Pagan | Evidence against constant residual variance under a declared implementation |
| Ljung-Box / Breusch-Godfrey | Supplemental serial-correlation checks, including when lagged targets limit DW interpretation |

Persist statistics, p-values when applicable, null hypotheses, settings, sample
sizes, validity warnings, and implementation versions. Separate training residuals
from out-of-sample errors. Post-search tests are diagnostic, not confirmatory
significance claims. Normality is not a prerequisite for useful point forecasts.
Robust standard errors alone do not fix forecast dynamics or prediction intervals.
Report prediction-interval method and observed coverage separately from coefficient
confidence intervals; unsupported intervals are explicitly unavailable.

Implementation references: [Shapiro-Wilk](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html),
[Durbin-Watson](https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html),
[Breusch-Pagan](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html),
and [VIF](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html).

## 7. Architecture and evidence

| Component | Responsibility |
| --- | --- |
| Python numerical core | Data validation, features, fitting, forecasting, backtesting, metrics, diagnostics, Pareto |
| LangGraph | Typed workflow state, conditional branches, bounded retries, user interrupts, checkpoints |
| Gemini API | Intent extraction, catalog-grounded proposals, eligible champion proposal, evidence-grounded narrative |
| Versioned repository policy | Decision IDs, routing rules, thresholds, budgets, monitoring policy |
| Langfuse | Runtime observations, decisions/evidence links, prompt versions, costs, latency, evaluation |
| SQLite and local artifacts | Sessions/checkpoints, experiments, decisions, snapshots, forecasts, model registry, alerts, reports |
| Streamlit / Plotly | Chat, candidate review, comparison, diagnostics, decision history, alerts |
| Independent scheduled worker | Forecast issuance, source refresh, delayed-outcome scoring, recalibration jobs |

LangGraph supports [checkpoints](https://docs.langchain.com/oss/python/langgraph/persistence)
and [human interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).
Langfuse offers [LangGraph tracing through LangChain](https://langfuse.com/integrations/frameworks/langchain)
and [prompt-to-trace links](https://langfuse.com/docs/prompt-management/features/link-to-traces).
The workflow and its rules live in the repository; Langfuse records their actual
execution. Neither tracing nor prompt management replaces the rule engine.

Use Gemini [structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
for typed decisions and [function calling](https://ai.google.dev/gemini-api/docs/function-calling)
for allowlisted tools. Record a pinned supported model ID. Tools supply all
numbers; the LLM interprets them. Keep API credentials in ignored environment
files or a secret manager; redact before traces, reports, and error logs.

Plan package areas under `src/manto`: `domain`, `data`, `analysis`, `selection`,
`agents`, `observability`, `storage`, `reporting`, `monitoring`, and `ui`.
Keep framework state outside statistical functions. Use one backend and a worker;
add FastAPI/PostgreSQL only when deployment requires them.

Every experiment stores parent ID, resolved request, snapshot hashes, feature and
family specs, split manifest, policy/prompt versions, LLM ID, code revision,
dependency lock, decisions, metrics, and artifact references. Numeric replay uses
frozen inputs; persist accepted LLM outputs because identical regeneration is not
guaranteed. [Agent workflow](AGENT_WORKFLOW.md) defines the proposed decision contract.

## 8. Monitoring and recalibration

Continuous service does not mean continuously arriving statistical evidence.
A one-month forecast can be scored only when the target observation matures.
Persist predictions before outcomes; distinguish first releases from revisions.

| Monitor | Evidence | Response |
| --- | --- | --- |
| Data freshness and schema | Missed release, changed unit, connector failure | Data alert; stale status or blocked forecast |
| Input change | Training/reference versus recent distribution with enough data | Drift warning; not proof of forecast degradation |
| Forecast quality | Matched matured predictions, rolling MAE, contemporaneous baseline | Refit/search proposal |
| Interval calibration | Observed coverage with sufficient matured outcomes | Uncertainty warning and challenger evaluation |
| ECM/seasonal stability | Equilibrium-error behavior, adjustment dynamics, seasonal changes | Method-specific review with its own rule ID |
| Operations | Failed jobs, duplicate work, latency, API budget | Technical alert and bounded retry |

A provisional monthly performance rule uses 12 matured forecasts and flags MAE
above 1.25 times its development reference for three consecutive newly scored
origins. Also flag sustained underperformance against a matched baseline.
Calibrate these heuristic thresholds by historical replay; they are not statistical
control limits. Define absolute tolerance for zero reference error.

Before enough outcomes exist, show `insufficient_evidence`, not healthy status.
Twelve monthly errors require roughly a year of live evidence. PoC acceptance
therefore uses historical replay with a simulated clock and labels it as replay.

Deduplicate alerts by model/rule/origin, preserve acknowledgements and resolutions,
and count each outcome once. Respect release calendars, cooldowns, and restart
recovery. A local worker requires the machine to run; always-on monitoring needs
an always-on deployment. A closed UI must not stop a running worker.

Coefficient refitting preserves the specification and changes the training cutoff;
specification search changes predictors, transformations, or family. Both create
versioned challengers under the same validation contract. Keep the incumbent,
record promotion, and support rollback. Diagnostic deterioration can motivate
review; it does not uniquely identify a variable to replace.

## 9. Fourteen-sprint delivery roadmap

Use outcome-based sprints, provisionally 1-2 weeks each: approximately 14-28 weeks
for the assumed staffing. This is an estimate, not a deadline. Re-estimate after
source verification and the first vertical slice. A first demo precedes the full
monitored PoC; user and statistical review are needed throughout.

| Sprint | Outcome | Acceptance evidence | Depends on |
| --- | --- | --- | --- |
| 1 | Freeze first target/data contract, target scale, release timing, catalog and baseline | Target plus ten candidates have verified identifiers/coverage or explicit blocked status; fixture fallback is documented | Product/data choices |
| 2 | Observable skeleton: Gemini extraction, LangGraph interrupt/resume, baseline tool, persisted session, Langfuse trace | Request survives restart; malformed LLM output and provider failure have explicit states; no credentials in traces | 1 |
| 3 | Target file import plus first macro connector; snapshots, as-of joins, revision labels | Later publications cannot enter earlier forecast origins; snapshots reproduce aligned inputs | 1-2 |
| 4 | OLS, D19 lag recipes, D20 stationarity transformations and inverses, baseline forecasts, chronological validation and holdout separation | Reproducible one-step forecasts; future/publication-lag leakage, sample trimming and inverse-transform fixtures pass | 3 |
| 5 | Candidate proposal/review, pin/exclude/manual/all modes, combinations and work budget | 120 combinations without pins and 36 with one pin; impossible/over-budget requests become explicit decisions | 2-4 |
| 6 | Metrics, initial OLS Pareto, thin comparison UI and provisional explanation | End-to-end linear analysis works, including ties/undefined scores/no-qualified-model | 4-5 |
| 7 | ADF/KPSS routing, transformation review, candidate-specific Engle-Granger and bounded inner lag selection | Cointegrated and unrelated random-walk fixtures exercise routes without promising perfect test power; inconclusive status exists; inner selection never sees outer outcomes | 4-6 |
| 8 | Linear ECM, operational feature timing, long/short-run interpretation and forecast reconstruction | Each fold re-estimates both stages; ECM cannot use unreleased inputs; common-scale comparison is reproducible | 7 |
| 9 | Seasonality decision, STL diagnostic views, compact Fourier regression; optional STL forecasting challenger | No full-history decomposition leakage; enabled/skipped branches have evidence; extra parameters are visible | 4, 7-8 |
| 10 | Complete residual/ECM/seasonal diagnostics, cross-family comparison, champion rules, Gemini evidence validation, holdout audit | Every Pareto member has the required suite; invalid claims fall back; failed holdout does not select a runner-up | 6-9 |
| 11 | Usable chat workspace, model switching, one-variable branches, side-by-side comparisons | Parent results remain immutable; incomparable scales/dates are flagged; view switch is separate from activation | 10 |
| 12 | Saved/reopened sessions, HTML report and machine-readable export, second macro connector | User can reopen and export provenance; frozen numeric outputs reproduce | 3, 11 |
| 13 | Persistent forecast worker, alert inbox, data/performance/method monitors and historical replay | Deterioration emits one actionable alert; repeated polling does not amplify evidence; restart preserves forecasts/alerts | 10-12 |
| 14 | Challenger refit/search, activation/rollback, failure handling, onboarding, PoC acceptance | Replay covers forecast, deterioration, challenger comparison, explicit promotion and rollback; user walkthrough completed | 13 |

Milestones: sprint 2 = observable skeleton; sprint 6 = linear analytical demo;
sprint 10 = extended methodological PoC; sprint 12 = usable analytical PoC;
sprint 14 = monitored PoC covering the broader request. Optional STL forecasting
must be labeled deferred if not delivered; STL diagnostics and the seasonal
decision branch remain required.

## 10. Acceptance and principal risks

Accept the full PoC when a non-specialist can define a problem, edit candidates,
complete an honest experiment, compare alternatives, inspect decision evidence,
save/reopen/export the analysis, and observe the monitoring/recalibration cycle.
Every number must come from stored computation and every recommendation must
reference evidence. A model need not beat the baseline for the software to pass.

| Risk | Delivery response |
| --- | --- |
| Weak WIG20/macro predictive relationship | Baseline/no-champion is a useful result; do not promise predictive skill |
| Publication/revision leakage | As-of contract, vintages where available, explicit exploratory/shadow labeling |
| Many combinations and small samples | Frozen search budget, development folds, untouched holdout, exposure tracking |
| Spurious level regression | Integration/cointegration branches, constrained ECM, inconclusive states |
| Seasonal overfitting | Training-only analysis, compact predefined variants, common out-of-sample comparison |
| LLM invents evidence | Catalog IDs, tool-supplied numbers, typed decisions, validation and fallback |
| Too little live monitoring evidence | Historical replay acceptance plus visible live insufficient-evidence status |
| Platform-sized scope | One target/frequency, modular monolith, explicit deferred extensions |
