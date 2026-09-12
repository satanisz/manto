"""Local, versioned explanations and read-only conversation detours."""

import json
import re
from difflib import get_close_matches

from manto.analysis import estimate_work
from manto.domain import AnalysisRequest
from manto.providers import _plain

# English product explanations, grounded in the actual execution contract.
TOPICS = {
    "initial_train": "Minimum usable monthly training pairs for the first model fit. 60 means at least five years of monthly training pairs. Lags, missing releases and transformations can require more calendar history. The training window then expands. Increasing it leaves fewer development evaluation months; it does not guarantee better forecasts.",
    "holdout_periods": "Number of final monthly outcomes reserved for the final audit of one frozen recommendation. Preliminary runs do not access them. A larger holdout leaves less development data. Reusing viewed outcomes is exploratory.",
    "min_development": "Minimum number of monthly out-of-sample development outcomes needed to compare models. This is separate from initial training and final holdout. A larger requirement can make a short dataset infeasible.",
    "lag_menu": "Allowed predictor delays in months (0–12). Lag L uses published X(t-L) to predict Y(t+1). Zero is the origin month, not the future outcome month. The engine tests one lag per predictor; adding options multiplies model combinations. Publication delay remains a separate constraint.",
    "model_size": "Number of distinct external predictors in each model, from 1 to 4. Pinned predictors count toward it. This is not the number of candidates or models. More predictors increase the search and may increase overfitting or collinearity.",
    "size_mode": "exact tests exactly model_size predictors; up_to also tests smaller feasible models, retaining all pins. The latter expands the search.",
    "candidate_ids": "Catalog variables allowed in the search, excluding the target. A shortlist of five candidates does not mean five predictors in each model. Candidates are hypotheses requiring validation.",
    "pinned_ids": "Predictors that must appear in every model. Their values and coefficients are not held constant. They must be candidates and count toward model_size.",
    "target_id": "The one catalog series Y to forecast, such as sales or margin. Changing it requires reviewing dependent candidates and transformation choices. Forecasts remain in its original units.",
    "target_transform": "Internal target recipe: auto, identity, difference or log_difference. Auto uses an initial-training ADF check, otherwise first differences. Forecasts are inverted to original units. Log differences require positive values and the inverse gives a conditional median. No recipe guarantees stationarity.",
    "feature_transform": "Predictor transformation: auto, unchanged identity, or first difference. Auto uses initial-training evidence. Differencing expresses changes rather than levels; already stationary data need not be differenced.",
    "max_models": "Safety cap on the complete model/lag grid. Over-budget searches are blocked, not silently truncated. Raising it permits more candidates but does not improve their quality by itself.",
    "max_fits": "Safety cap on total fits across models and chronological origins. One candidate needs many fits. The preflight estimate also conservatively reserves the final audit/forecast work.",
    "run_mode": "preliminary saves a development-only comparison without holdout, champion or next forecast. full also audits the frozen recommendation and attempts the next forecast. Both currently use the linear engine; full does not enable future ECM/monitoring features.",
    "share_vectors": "Whether explicitly confirmed training-prefix numerical vectors may be sent to Gemini. False still allows configured metadata and bounded summaries. It does not send the entire series or authorize analysis; Langfuse receives metadata only.",
    "r2": "R² compares squared model errors to variation around the sample mean. Higher is better on the same data/scale; out-of-sample R² can be negative. Training R² is not forecast accuracy.",
    "vif": "VIF measures predictor collinearity, not forecast accuracy. Larger values indicate inflated coefficient variance; compare it alongside out-of-sample errors, not as a standalone winner rule.",
    "mae": "MAE is the average absolute forecast error, in original target units. Lower is better on matching outcomes. RMSE weights large errors more heavily; MAPE expresses percentage errors and can be undefined near zero.",
    "pareto": "A Pareto model is not dominated on all declared objectives by another candidate. Here those objectives are development MAE, maximum VIF and error variability. There may be several trade-offs, not one universal winner.",
}

ALIASES = {
    "initial train": "initial_train",
    "poczatkowe okno": "initial_train",
    "holdout": "holdout_periods",
    "lag": "lag_menu",
    "lagi": "lag_menu",
    "r²": "r2",
    "r2": "r2",
    "max vif": "vif",
    "rmse": "mae",
    "mape": "mae",
}


def topic_mentions(message):
    plain = _plain(message)
    found = []
    for term, key in {**{key: key for key in TOPICS}, **ALIASES}.items():
        if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", plain) and key not in found:
            found.append(key)
    return found


def question_intent(message):
    plain = _plain(message).strip()
    return bool(
        "?" in plain
        or re.match(
            r"^(co (robi|to|oznacza|daje|jesli)|czym |jak dziala|po co |dlaczego |wyjasnij |"
            r"what (is|does|if)|how does|why |explain |a (gdy|jesli)|czy )",
            plain,
        )
    )


def help_action(message, state):
    """Return a bounded action dictionary, or defer to Gemini/general routing."""
    plain = _plain(message).strip(" .!")
    if plain.startswith("set {"):
        return None  # Preserve the existing typed, multi-field JSON command.
    topics = topic_mentions(message)
    if re.search(
        r"\b(selected|recommended|champion|result|wybran\w*|rekomendowan\w*|wynik\w*)\b", plain
    ) and not re.match(r"^(explain|what is|co to|co oznacza|wyjasnij)\b", plain):
        return None  # Inspect stored results, not the meaning of their metric/lag labels.
    if plain in {"help", "pomoc", "co moge zrobic", "co moge zrobic?"}:
        return {"action": "help"}
    if plain in {
        "back to setup",
        "continue setup",
        "wroc do konfiguracji",
        "wrocmy do konfiguracji",
    }:
        return {"action": "resume_setup"}
    if len(topics) > 1 and question_intent(message):
        return {"action": "explain_setting", "reference": ",".join(topics)}
    if len(topics) > 1 and re.match(r"^(ustaw|zmien|set|change)\s", plain):
        return {"action": "clarify", "text": "multiple_settings"}
    is_followup = bool(
        re.match(r"^(a (gdy|jesli|dlaczego)|what if|why\b|dlaczego\b|ustaw\b|set\b)", plain)
    )
    context_topic = (state.inquiry or {}).get("topic") if is_followup else None
    topic = topics[0] if topics else context_topic
    hypothetical = bool(
        re.search(r"\b(what if|co jesli|a gdy|a jesli|czy .*\d|would .*\d)\b", plain)
    )
    if topic and (question_intent(message) or plain in {"a dlaczego", "dlaczego", "why"}):
        if hypothetical and topic in state.draft.settings:
            # Digits in setting names (e.g. R2) cannot become hypothetical values.
            values = re.findall(r"(?<![\w.])-?\d+(?![\w.])", plain)
            value = (
                [int(item) for item in values]
                if topic == "lag_menu"
                else int(values[-1])
                if len(values) == 1
                else None
            )
            return {"action": "what_if", "reference": topic, "hypothetical_value": value}
        return {"action": "explain_setting", "reference": topic}
    if (
        topic in state.draft.settings
        and len(topics) <= 1
        and re.match(r"^(ustaw|zmien|set|change)\s", plain)
        and not question_intent(message)
    ):
        numbers = re.findall(r"(?<![\w.])-?\d+(?![\w.])", plain)
        scalar = {
            "initial_train",
            "holdout_periods",
            "min_development",
            "max_models",
            "max_fits",
            "model_size",
        }
        if len(numbers) == 1 and topic in scalar:
            return {"action": "patch", "changes": {topic: int(numbers[0])}}
        if numbers and topic == "lag_menu":
            return {"action": "patch", "changes": {topic: [int(item) for item in numbers]}}
    if re.fullmatch(r"\d+", plain):
        return {"action": "clarify", "text": "number"}
    if question_intent(message):
        unknown = re.findall(r"\b[a-z]+_[a-z_]+\b", plain)
        if unknown:
            suggestions = get_close_matches(
                unknown[0], list(state.draft.settings), n=1, cutoff=0.65
            )
            return {"action": "clarify", "reference": suggestions[0] if suggestions else None}
    return None


def explain_topics(state, topics):
    blocks = []
    for topic in topics:
        if topic not in TOPICS:
            raise ValueError("Unknown explanation topic")
        text = f"**{topic}** — {TOPICS[topic]}"
        if topic in state.draft.settings:
            setting = state.draft.settings[topic]
            text += (
                "\n\nCurrent value: "
                + f"`{json.dumps(setting.value, ensure_ascii=False)}` ({setting.status})."
            )
            field = AnalysisRequest.model_fields.get(topic)
            bounds = [
                f"{symbol} {getattr(constraint, attribute)}"
                for constraint in (field.metadata if field else [])
                for attribute, symbol in (("ge", "≥"), ("le", "≤"))
                if getattr(constraint, attribute, None) is not None
            ]
            if bounds:
                text += " Constraints: " + ", ".join(bounds) + "."
        blocks.append(text)
    blocks.append(
        "No settings changed and no analysis ran. Ask a follow-up or say 'back to setup'."
    )
    return "\n\n".join(blocks)


def preview_change(dataset, draft, changes):
    """Validate a temporary draft and estimate work; never fit, approve, or persist it."""
    from manto.agent_tools import validate_catalog

    alternative = draft.patch(changes, "hypothetical", proposed=True)
    validate_catalog(dataset, alternative)
    estimates = []
    for option in (draft, alternative):
        values = {key: option.values[key] for key in AnalysisRequest.model_fields}
        try:
            request = AnalysisRequest.model_validate(values)
        except ValueError:
            return {"changes": changes, "estimates": None, "reason": "incomplete_specification"}
        estimates.append(estimate_work(dataset, request))
    return {"changes": changes, "estimates": estimates, "reason": estimates[1]["reason"]}
