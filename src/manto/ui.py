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
from pydantic import ValidationError

from manto.data import demo_dataset, load_csv
from manto.domain import AnalysisRequest, AnalysisResult
from manto.policy import load_policy
from manto.reporting import model_table, prediction_chart, render_report
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
        st.session_state.pop("load_error", None)
    except (ValueError, OSError):
        st.session_state["load_error"] = (
            "The saved analysis or input snapshot failed its integrity check."
        )


def _start(directory, dataset, message):
    thread_id = str(uuid4())
    st.session_state["thread_id"] = thread_id
    state = _conversation_call(directory, dataset, "start", thread_id, message)
    st.session_state["user_message"] = message
    st.session_state.pop("analysis_result", None)
    _accept_state(state, directory, dataset)


def _review(directory, dataset):
    policy = load_policy()
    state = st.session_state.get("conversation_state", {})
    question = state.get("question") or {}
    request = (
        state.get("request")
        or question.get("request")
        or st.session_state.get("request_defaults")
        or {}
    )
    if not isinstance(request, dict):
        request = {}
    if question:
        st.info(question.get("message", "Review the experiment before running the comparison."))
        for error in question.get("errors", []):
            st.warning(str(error))
    ids = [entry.id for entry in dataset.catalog]
    labels = {entry.id: f"{entry.title} · {entry.id}" for entry in dataset.catalog}
    target_default = request.get("target_id")
    if target_default not in ids:
        target_default = next(
            (entry.id for entry in dataset.catalog if entry.role == "target"), ids[0]
        )
    # Target is outside the form so changing it immediately updates candidate options.
    target = st.selectbox(
        "What do you want to forecast?",
        ids,
        index=ids.index(target_default),
        format_func=lambda value: labels[value],
        key="target_choice",
    )
    available = [entry for entry in ids if entry != target]
    candidates = [entry for entry in request.get("candidate_ids", available) if entry in available]
    if not candidates:
        candidates = available
    pins = [entry for entry in request.get("pinned_ids", []) if entry in available]
    with st.form("experiment_review"):
        st.markdown("#### Your experiment")
        selected = st.multiselect(
            "Candidate variables",
            available,
            default=candidates,
            format_func=lambda value: labels[value],
        )
        pinned = st.multiselect(
            "Always include",
            available,
            default=pins,
            format_func=lambda value: labels[value],
            help="Pinned variables count toward model size. Include them in the candidate list.",
        )
        left, right = st.columns(2)
        size = left.number_input(
            "Variables per model", min_value=1, max_value=4, value=int(request.get("model_size", 3))
        )
        size_mode = right.selectbox(
            "Search size",
            ["exact", "up_to"],
            index=0 if request.get("size_mode", "exact") == "exact" else 1,
            format_func=lambda value: (
                "Exactly this many" if value == "exact" else "Up to this many"
            ),
        )
        lags = st.multiselect(
            "Monthly predictor lags",
            list(range(13)),
            default=request.get("lag_menu", policy["default_lag_menu"]),
            help="0 uses the origin month only if already published. Each extra lag expands the search.",
        )
        with st.expander("Transformation and validation settings"):
            options = ["auto", "identity", "difference", "log_difference"]
            target_transform = st.selectbox(
                "Internal target transformation",
                options,
                index=options.index(request.get("target_transform", "auto")),
                help="Auto checks only the initial training prefix. Forecasts return to original units.",
            )
            feature_options = ["auto", "identity", "difference"]
            feature_transform = st.selectbox(
                "Predictor transformation",
                feature_options,
                index=feature_options.index(request.get("feature_transform", "auto")),
            )
            st.caption(
                "Monthly · one month ahead · expanding training window · final holdout kept separate."
            )
            train = st.number_input(
                "Initial training months",
                min_value=24,
                max_value=600,
                value=int(request.get("initial_train", policy["initial_train"])),
            )
            holdout = st.number_input(
                "Final holdout months",
                min_value=3,
                max_value=60,
                value=int(request.get("holdout_periods", policy["holdout_periods"])),
            )
        submitted = st.form_submit_button("Run model comparison", type="primary", width="stretch")
    if submitted:
        try:
            spec = AnalysisRequest(
                target_id=target,
                candidate_ids=selected,
                pinned_ids=pinned,
                model_size=int(size),
                size_mode=size_mode,
                lag_menu=lags,
                target_transform=target_transform,
                feature_transform=feature_transform,
                initial_train=int(train),
                holdout_periods=int(holdout),
                min_development=policy["min_development"],
                max_models=policy["max_models"],
                max_fits=policy["max_fits"],
            )
            from manto.analysis import estimate_work

            work = estimate_work(dataset, spec)
            if not work.get("within_budget", True):
                st.error(
                    work.get("reason")
                    or "This search exceeds the work budget. Reduce candidates or lags."
                )
                return
            with st.spinner(
                f"Evaluating {work.get('model_count', 'candidate')} models on past-only windows…"
            ):
                if not state.get("question"):
                    _start(directory, dataset, f"Analyze {target} with {size} variables.")
                response = _conversation_call(
                    directory,
                    dataset,
                    "resume",
                    st.session_state["thread_id"],
                    {"approved": True, "request": spec.model_dump()},
                )
                _accept_state(response, directory, dataset)
            st.rerun()
        except (ValidationError, ValueError) as error:
            st.error(str(error))


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
        st.dataframe(table, hide_index=True, width="stretch")
        finite = table.dropna(subset=["MAE", "Max VIF"])
        if len(finite) > 1:
            scatter = px.scatter(
                finite,
                x="MAE",
                y="Max VIF",
                color="Pareto",
                hover_data=["Model", "Variables", "Error variability"],
                color_discrete_map={True: "#13876f", False: "#a8b4bf"},
            )
            scatter.update_layout(
                template="plotly_white",
                height=330,
                title="Forecast error vs. collinearity — lower is better",
            )
            st.plotly_chart(scatter, width="stretch")
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
                    state = _conversation_call(directory, dataset, "state", thread.strip())
                    st.session_state["thread_id"] = thread.strip()
                    _accept_state(state, directory, dataset)
                    st.rerun()
                except ValueError:
                    st.error("This conversation cannot be resumed with the selected dataset.")
    # Changing data must not silently reuse an experiment's target/candidate context.
    dataset_key = _dataset_key(dataset)
    if st.session_state.get("dataset_key") not in (None, dataset_key):
        for key in (
            "conversation_state",
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
    st.title("From a question to an explainable forecast.")
    st.markdown(
        "Explore a monthly target, compare small linear models, and see the evidence behind every choice."
    )
    if "synthetic" in dataset.provenance.lower():
        st.info(
            "Synthetic demonstration data — these are simulated relationships, not real market or business findings."
        )
    workspace, decisions, data_tab, saved_tab = st.tabs(
        ["Workspace", "Decision trail", "Data & sources", "Saved analyses"]
    )
    with workspace:
        prompt = st.chat_input("Describe your target and any variables you want to include…")
        if prompt:
            with st.spinner("Preparing your experiment…"):
                _start(directory, dataset, prompt)
            st.rerun()
        if st.session_state.get("user_message"):
            with st.chat_message("user"):
                st.write(st.session_state["user_message"])
        conversation_state = st.session_state.get("conversation_state", {})
        assistant_messages = [
            item
            for item in conversation_state.get("messages", [])
            if item.get("role") == "assistant"
        ]
        if assistant_messages:
            with st.chat_message("assistant"):
                st.write(assistant_messages[-1].get("content", ""))
        if conversation_state.get("status") == "failed":
            st.error("This run failed. Review the message above and start a new comparison.")
        elif conversation_state.get("status") == "cancelled":
            st.info("This conversation was cancelled. Start a new comparison when ready.")
        if not st.session_state.get("conversation_state"):
            st.caption(
                "Start with a question, or configure the experiment below. "
                "In offline mode the assistant uses a bounded catalog parser, not an LLM."
            )
            if st.button("Try: forecast sales with 3 variables, including inflation"):
                _start(directory, dataset, "Forecast sales with 3 variables, including inflation")
                st.rerun()
        if st.session_state.get("analysis_result"):
            _render_result(AnalysisResult.model_validate(st.session_state["analysis_result"]))
            with st.expander("Change a variable and start a new comparison"):
                _review(directory, dataset)
        else:
            _review(directory, dataset)
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
        with closing(Conversation(directory / "conversations.sqlite", dataset)) as conversation:
            graph = conversation.graph_mermaid()
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
