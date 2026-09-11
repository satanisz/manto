"""Local conversational workbench for the first Manto demo."""

import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from manto.agent_tools import render_specification
from manto.data import demo_dataset, load_csv
from manto.dialogue import AgentConversation
from manto.domain import AnalysisResult
from manto.policy import load_policy
from manto.reporting import (
    METRIC_LABELS,
    candidate_metrics,
    model_table,
    prediction_chart,
    render_report,
)
from manto.storage import ResultStore
from manto.workflow import Conversation


@st.cache_data
def _demo():
    return demo_dataset()


def _conversation_call(directory, dataset, method, *args):
    with closing(Conversation(directory / "conversations.sqlite", dataset)) as conversation:
        return getattr(conversation, method)(*args)


def _accept_state(state, directory, dataset):
    st.session_state["conversation_state"] = state
    if state.get("result"):
        result = AnalysisResult.model_validate(state["result"])
        ResultStore(directory / "experiments").save(result, dataset)
        st.session_state["analysis_result"] = result.model_dump(mode="json")
    else:
        st.session_state.pop("analysis_result", None)


def _dataset_key(dataset):
    content = dataset.observations.to_csv(index=False)
    metadata = json.dumps([item.model_dump() for item in dataset.catalog], sort_keys=True)
    return hashlib.sha256(
        f"{content}:{metadata}:{dataset.name}:{dataset.provenance}".encode()
    ).hexdigest()


def _open_saved(directory, experiment_id):
    try:
        store = ResultStore(Path(directory) / "experiments")
        result = store.load(experiment_id)
        dataset = store.load_dataset(experiment_id)
        for key in ("conversation_state", "thread_id", "user_message", "target_choice"):
            st.session_state.pop(key, None)
        st.session_state["source_choice"] = "Saved input snapshot"
        st.session_state["saved_dataset"] = dataset
        st.session_state["dataset_key"] = _dataset_key(dataset)
        st.session_state["request_defaults"] = result.request.model_dump()
        st.session_state["analysis_result"] = result.model_dump(mode="json")
        thread_id = str(uuid4())
        with AgentConversation(directory, dataset) as chat:
            dialogue = chat.import_result(thread_id, experiment_id)
        st.session_state["thread_id"] = thread_id
        st.session_state["dialogue_state"] = dialogue.model_dump()
        st.session_state.pop("load_error", None)
    except (ValueError, OSError):
        st.session_state["load_error"] = (
            "The saved analysis or input snapshot failed its integrity check."
        )


def _start(directory, dataset, message):
    st.session_state.pop("target_choice", None)
    st.session_state.pop("request_defaults", None)
    thread_id = str(uuid4())
    st.session_state["thread_id"] = thread_id
    state = _conversation_call(directory, dataset, "start", thread_id, message)
    st.session_state["user_message"] = message
    st.session_state.pop("analysis_result", None)
    _accept_state(state, directory, dataset)


def _chat_send(directory, dataset, message):
    thread_id = st.session_state.setdefault("thread_id", str(uuid4()))
    with AgentConversation(directory, dataset) as chat:
        state = chat.send(thread_id, message)
    st.session_state["dialogue_state"] = state.model_dump()
    if state.result:
        st.session_state["analysis_result"] = state.result


def _render_result(result: AnalysisResult):
    st.markdown("### Results in your original units")
    st.caption(f"{result.dataset_name} · {result.provenance} · experiment {result.experiment_id}")
    if result.champion_id:
        st.success(result.explanation or "A candidate passed the declared baseline audit.")
    else:
        st.info(
            result.explanation
            or "No qualified champion. Development results remain available for exploration."
        )
    if result.warnings:
        with st.expander(f"Assumptions and limitations ({len(result.warnings)})"):
            for warning in result.warnings:
                st.write(f"• {warning}")
    table = model_table(result)
    eligible = [model for model in result.models if model.predictions]
    first, second, third = st.columns(3)
    first.metric("Models evaluated", len(result.models))
    second.metric("Pareto alternatives", sum(model.is_pareto for model in result.models))
    third.metric("Horizon", "1 month")
    if result.next_forecast:
        forecast = result.next_forecast
        period = pd.Timestamp(forecast["period"]).strftime("%B %Y")
        st.metric(
            f"Next forecast · {period}",
            f"{forecast['predicted']:,.2f} {forecast.get('unit', 'units')}",
        )
        st.caption(
            "Experimental one-month forecast in the original target units. Prediction intervals are not available in this milestone."
        )
        with st.expander("Forecast calculation and transformation evidence"):
            st.json(forecast, expanded=True)
    if eligible:
        model_ids = [model.model_id for model in eligible]
        preferred = result.recommended_id if result.recommended_id in model_ids else model_ids[0]
        chosen_id = st.selectbox(
            "Inspect a model",
            model_ids,
            index=model_ids.index(preferred),
            key=f"model_{result.experiment_id}",
        )
        chosen = next(model for model in eligible if model.model_id == chosen_id)
        st.caption(" + ".join(f"{name} (lag {chosen.lags[name]})" for name in chosen.features))
        st.plotly_chart(prediction_chart(chosen), width="stretch", key=f"forecast_{chosen_id}")
        st.caption(
            "Validation is out of sample. Holdout results exist only for the frozen recommended model."
        )
        with st.expander("Transformations, coefficients and metric definitions"):
            st.json(chosen.transformations)
            st.json(chosen.coefficients)
            st.json(chosen.metrics)
            st.caption(
                "Training R² describes fit; validation MAE/RMSE measure original-scale forecast errors. "
                "VIF concerns predictor collinearity. Undefined metrics remain unavailable."
            )
    if not table.empty:
        st.markdown("#### Compare all candidates")
        display_columns = [
            "Pareto",
            "MAE",
            "RMSE",
            "Max VIF",
            "R² (validation)",
            "Error variability",
            "Variables",
            "Model",
            "Status",
            "Reason",
        ]
        st.dataframe(table[display_columns], hide_index=True, width="stretch")
        metrics = candidate_metrics(result)
        options = list(metrics.columns)
        if options:
            left_axis, right_axis = st.columns(2)
            x_metric = left_axis.selectbox(
                "X axis",
                options,
                index=options.index("development_mae") if "development_mae" in options else 0,
                format_func=lambda key: METRIC_LABELS.get(key, key),
                key=f"comparison_x_{result.experiment_id}",
            )
            y_metric = right_axis.selectbox(
                "Y axis",
                options,
                index=options.index("max_vif") if "max_vif" in options else 0,
                format_func=lambda key: METRIC_LABELS.get(key, key),
                key=f"comparison_y_{result.experiment_id}",
            )
            finite = metrics.dropna(subset=[x_metric, y_metric]).join(
                table[["Model", "Variables", "Pareto"]]
            )
            st.caption(
                f"Showing {len(finite)} of {len(table)} candidates with finite values on both axes. "
                "Axis selection does not change the original Pareto membership or recommendation. "
                "Higher R² is better; lower forecast errors and VIF are better. "
                "Training fit and out-of-sample scores have different interpretations. "
                "Holdout metrics exist only for the frozen recommended model."
            )
        else:
            finite = pd.DataFrame()
        if not finite.empty:
            scatter = px.scatter(
                finite,
                x=x_metric,
                y=y_metric,
                labels=METRIC_LABELS,
                color="Pareto",
                hover_data=["Model", "Variables"],
                color_discrete_map={True: "#13876f", False: "#a8b4bf"},
            )
            scatter.update_layout(
                template="plotly_white",
                height=330,
                title="Candidate metric comparison",
            )
            st.plotly_chart(scatter, width="stretch", key=f"comparison_{result.experiment_id}")
        else:
            st.info("No candidates have finite values for the selected metrics.")
    with st.expander("Validation dates and evidence"):
        st.json(result.split)
    left, right = st.columns(2)
    left.download_button(
        "Download result JSON",
        result.model_dump_json(indent=2),
        file_name=f"manto-{result.experiment_id}.json",
        mime="application/json",
    )
    right.download_button(
        "Download HTML report",
        render_report(result),
        file_name=f"manto-{result.experiment_id}.html",
        mime="text/html",
    )


def main():
    load_dotenv()
    st.set_page_config(page_title="Manto · Time-series workbench", page_icon="◒", layout="wide")
    directory = Path(os.getenv("MANTO_DATA_DIR", ".manto")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with st.sidebar:
        st.markdown("## ◒ manto")
        st.caption("Time-series workbench / Sprint 6")
        source = st.radio(
            "Data source",
            ["Synthetic sales demo", "Upload your data", "Saved input snapshot"],
            key="source_choice",
        )
        if source == "Upload your data":
            uploaded = st.file_uploader("Monthly long-form CSV", type=["csv"])
            st.caption(
                "Required: series_id, period, available_at, value. The target can be any numeric business series."
            )
            if uploaded is None:
                st.info("Upload a file or select the synthetic demo.")
                st.stop()
            try:
                dataset = load_csv(uploaded)
            except ValueError as error:
                st.error(str(error))
                st.stop()
        elif source == "Saved input snapshot":
            dataset = st.session_state.get("saved_dataset")
            if dataset is None:
                st.info(
                    "Select the synthetic demo, then use Saved analyses to reopen a result and its input snapshot."
                )
                st.stop()
        else:
            dataset = _demo()
        st.divider()
        gemini = os.getenv("MANTO_ENABLE_GEMINI", "false").lower() == "true"
        st.caption(
            "Gemini requested (fallback available)"
            if gemini
            else "Guided offline mode · no API key needed"
        )
        st.caption(
            "Your target stays in its natural units. Transformations are internal and recorded."
        )
        if st.button("New conversation", width="stretch"):
            for key in (
                "conversation_state",
                "dialogue_state",
                "analysis_result",
                "thread_id",
                "user_message",
                "target_choice",
                "request_defaults",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        if st.session_state.get("thread_id"):
            st.caption("Conversation ID")
            st.code(st.session_state["thread_id"], language=None)
        with st.expander("Resume a saved conversation"):
            thread = st.text_input("Conversation ID to resume")
            if st.button("Resume") and thread.strip():
                try:
                    with AgentConversation(directory, dataset) as chat:
                        state = chat.state(thread.strip())
                    st.session_state["thread_id"] = thread.strip()
                    st.session_state["dialogue_state"] = state.model_dump()
                    if state.result:
                        st.session_state["analysis_result"] = state.result
                    else:
                        st.session_state.pop("analysis_result", None)
                    st.rerun()
                except ValueError:
                    st.error("This conversation cannot be resumed with the selected dataset.")
    # Changing data must not silently reuse an experiment's target/candidate context.
    dataset_key = _dataset_key(dataset)
    if st.session_state.get("dataset_key") not in (None, dataset_key):
        for key in (
            "conversation_state",
            "dialogue_state",
            "analysis_result",
            "thread_id",
            "user_message",
            "target_choice",
            "request_defaults",
        ):
            st.session_state.pop(key, None)
    st.session_state["dataset_key"] = dataset_key
    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])
    st.title("Manto · Analytical conversation")
    if "synthetic" in dataset.provenance.lower():
        st.info(
            "Synthetic demonstration data — these are simulated relationships, not real market or business findings."
        )
    workspace, decisions, data_tab, saved_tab = st.tabs(
        ["Workspace", "Decision trail", "Data & sources", "Saved analyses"]
    )
    with workspace:
        dialogue = st.session_state.get("dialogue_state", {})
        st.caption(
            dialogue.get("mode", "Gemini when configured · limited offline recovery otherwise")
        )
        for message in dialogue.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if not dialogue:
            with st.chat_message("assistant"):
                st.write(
                    "What would you like to analyze? Tell me your business question. We can explore the available variables together before approving any calculations."
                )
            if st.button("Try: forecast sales with 3 variables, including inflation"):
                _chat_send(
                    directory, dataset, "Forecast sales with 3 variables, including inflation"
                )
                st.rerun()
        if dialogue.get("draft"):
            with st.expander("Current analysis setup · read only"):
                st.json(dialogue["draft"])
        if dialogue.get("report"):
            report = dialogue["report"]
            left, right = st.columns(2)
            left.download_button(
                "Download specification",
                render_specification(report),
                file_name=f"manto-spec-{report['report_id'][:12]}.md",
            )
            right.download_button(
                "Download specification JSON",
                json.dumps(report, indent=2),
                file_name=f"manto-spec-{report['report_id'][:12]}.json",
            )
        prompt = st.chat_input("Reply, ask a question, or change an analysis setting…")
        if prompt:
            try:
                with st.spinner("Working on your message…"):
                    _chat_send(directory, dataset, prompt)
                st.rerun()
            except (ValueError, OSError):
                st.error(
                    "The conversation could not be updated. Check the input snapshot and retry; your saved results remain available."
                )
        if st.session_state.get("analysis_result"):
            _render_result(AnalysisResult.model_validate(st.session_state["analysis_result"]))
            st.caption(
                "These are the last saved results. Chat changes prepare a new experiment; they do not overwrite these results."
            )
    with decisions:
        st.markdown("### What happened, and why")
        st.caption(
            "Rules and tool evidence are inspectable. This is an execution record, not private model reasoning."
        )
        result_data = st.session_state.get("analysis_result")
        if result_data:
            for decision in AnalysisResult.model_validate(result_data).decisions:
                with st.expander(f"{decision.rule_id} · {decision.outcome}"):
                    st.write(decision.explanation)
                    st.json(decision.evidence)
        else:
            st.info("Run a comparison to populate the numerical decision trail.")
        for event in st.session_state.get("dialogue_state", {}).get("events", []):
            with st.expander(f"{event['rule_id']} · {event['outcome']} · {event['turn_id'][:8]}"):
                st.json(event)
        with AgentConversation(directory, dataset) as conversation:
            graph = conversation.graph.get_graph().draw_mermaid()
            topology = conversation.graph.get_graph()
            dot = [
                'digraph workflow { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef6f3", color="#13876f", fontname="Arial"];'
            ]
            for node in topology.nodes:
                label = {
                    "__start__": "Start",
                    "__end__": "Finish",
                    "review": "User review",
                    "analysis": "Numerical analysis",
                }.get(node, node.title())
                dot.append(f"{json.dumps(node)} [label={json.dumps(label)}];")
            for edge in topology.edges:
                style = " [style=dashed]" if edge.conditional else ""
                dot.append(f"{json.dumps(edge.source)} -> {json.dumps(edge.target)}{style};")
            dot.append("}")
        st.markdown("#### Executable conversation graph")
        st.graphviz_chart("\n".join(dot), width="stretch")
        st.caption(
            "Dashed arrows are conditional routes. Detailed numerical choices appear in the decision trail above."
        )
        with st.expander("Mermaid graph source"):
            st.code(graph, language="mermaid")
        st.download_button("Download graph", graph, file_name="manto-demo-workflow.mmd")
        with st.expander("Active demo policy"):
            st.json(load_policy())
        st.caption(
            "Cointegration, ECM, advanced seasonality and monitoring remain explicit later-sprint branches."
        )
    with data_tab:
        st.markdown("### Catalog and availability")
        st.dataframe(
            pd.DataFrame([entry.model_dump() for entry in dataset.catalog]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Provenance: {dataset.provenance}. period is the observation month; "
            "available_at is when that value could be used. Revisions remain separate records."
        )
        st.dataframe(dataset.observations.head(40), hide_index=True, width="stretch")
        st.download_button(
            "Download data / CSV template",
            dataset.observations.to_csv(index=False),
            file_name="manto-monthly-data.csv",
            mime="text/csv",
        )
        st.markdown(
            "Public FX connector: `manto fetch-nbp --code EUR --start 2015-01-01 --end 2025-12-31`. "
            "Macro catalog sourcing and WIG20 access are documented under `docs/DATA_SOURCES.md`."
        )
    with saved_tab:
        st.markdown("### Reopen your evidence")
        store = ResultStore(directory / "experiments")
        saved = store.list_results()
        if not saved:
            st.info("Completed runs are saved automatically on this computer.")
        else:
            ids = [item["experiment_id"] for item in saved]
            selected = st.selectbox(
                "Saved experiment",
                ids,
                format_func=lambda value: next(
                    f"{i['dataset_name']} · {i['status']} · {value[:8]}"
                    for i in saved
                    if i["experiment_id"] == value
                ),
            )
            st.button("Open saved analysis", on_click=_open_saved, args=(str(directory), selected))
            st.caption(
                "Reopening also restores the original input snapshot and request for a reproducible new comparison."
            )
    st.divider()
    st.caption(
        "Manto / Sprint 6 demo · Original-scale forecasts · Past-only evaluation · Explicit decisions"
    )


if __name__ == "__main__":
    main()
