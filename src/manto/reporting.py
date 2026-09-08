"""Charts and a portable, escaped HTML summary built from stored evidence."""

from html import escape

import pandas as pd
import plotly.graph_objects as go

from manto.domain import AnalysisResult, ModelResult


def model_table(result: AnalysisResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": model.model_id,
                "Variables": ", ".join(
                    f"{name} (lag {model.lags[name]})" for name in model.features
                ),
                "Status": model.status,
                "Pareto": model.is_pareto,
                "MAE": model.metrics.get("development_mae"),
                "RMSE": model.metrics.get("development_rmse"),
                "R² (validation)": model.metrics.get("development_r2"),
                "Max VIF": model.metrics.get("max_vif"),
                "Error variability": model.metrics.get("block_mae_std"),
                "Reason": model.reason,
            }
            for model in result.models
        ]
    )


def prediction_chart(model: ModelResult) -> go.Figure:
    figure = go.Figure()
    frame = pd.DataFrame(model.predictions)
    if not frame.empty:
        for column, label, color in [
            ("actual", "Observed", "#24384b"),
            ("predicted", "One-month prediction", "#13876f"),
            ("baseline", "Last-value baseline", "#bd8e45"),
        ]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(
                        x=frame["period"],
                        y=frame[column],
                        name=label,
                        line={"color": color, "width": 2},
                    )
                )
        if "split" in frame and (frame["split"] == "holdout").any():
            holdout_start = frame.loc[frame["split"] == "holdout", "period"].iloc[0]
            figure.add_vrect(
                x0=holdout_start,
                x1=frame["period"].iloc[-1],
                fillcolor="#d6e6e2",
                opacity=0.3,
                line_width=0,
                annotation_text="Holdout audit",
                annotation_position="top left",
            )
    figure.update_layout(
        template="plotly_white",
        height=390,
        margin={"t": 35, "b": 30},
        yaxis_title="Original target units",
        xaxis_title="Outcome month",
        legend={"orientation": "h", "y": 1.16},
        hovermode="x unified",
    )
    return figure


def render_report(result: AnalysisResult) -> str:
    """No untrusted HTML or remote scripts; report opens without a network."""
    chosen = next((m for m in result.models if m.model_id == result.recommended_id), None)
    chart = (
        prediction_chart(chosen).to_html(full_html=False, include_plotlyjs=True) if chosen else ""
    )
    table = model_table(result).to_html(index=False, escape=True, na_rep="Unavailable")
    decisions = "".join(
        f"<li><strong>{escape(d.rule_id)} · {escape(d.outcome)}</strong><p>{escape(d.explanation)}</p></li>"
        for d in result.decisions
    )
    warnings = "".join(f"<li>{escape(w)}</li>" for w in result.warnings)
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Manto · {escape(result.experiment_id)}</title>
<style>body{{font:16px system-ui;color:#24384b;max-width:1200px;margin:40px auto;padding:0 24px}}
h1{{color:#13876f}}table{{border-collapse:collapse;font-size:13px;width:100%}}
td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.meta{{color:#5d6a77}}
section{{margin:32px 0}}li p{{margin-top:4px}}</style>
<h1>Manto / Analytical report</h1><p class="meta">{escape(result.dataset_name)} ·
{escape(result.provenance)} · {escape(result.status)}</p>
<p>Experiment: {escape(result.experiment_id)}. Policy: {escape(result.policy_version)}.</p>
<p>{escape(result.explanation)}</p><ul>{warnings}</ul>
<section><h2>Original-scale predictions</h2>{chart}</section>
<section><h2>Development comparison</h2>{table}</section>
<section><h2>Decision history</h2><ol>{decisions}</ol></section>
<p class="meta">Sprint 6 demo: relationships are not causal evidence. ECM,
full residual diagnostics and continuous monitoring are later milestones.</p></html>"""
