"""Local, versioned explanations and read-only conversation detours."""

import json
import re
from difflib import get_close_matches

from manto.analysis import estimate_work
from manto.domain import AnalysisRequest
from manto.providers import _plain

# English/Polish product explanations, grounded in the actual execution contract.
TOPICS = {
    "initial_train": (
        "Minimum usable monthly training pairs for the first model fit. 60 means at least five years of monthly training pairs. Lags, missing releases and transformations can require more calendar history. The training window then expands. Increasing it leaves fewer development evaluation months; it does not guarantee better forecasts.",
        "Minimalna liczba użytecznych miesięcznych par danych do pierwszego dopasowania modelu. 60 oznacza co najmniej pięć lat miesięcznych danych treningowych. Lagi, brakujące publikacje i transformacje mogą wymagać dłuższej historii kalendarzowej. Później okno treningowe rośnie. Zwiększenie tej wartości zostawia mniej miesięcy do walidacji — nie gwarantuje lepszych prognoz.",
    ),
    "holdout_periods": (
        "Number of final monthly outcomes reserved for the final audit of one frozen recommendation. Preliminary runs do not access them. A larger holdout leaves less development data. Reusing viewed outcomes is exploratory.",
        "Liczba ostatnich miesięcznych wyników odłożonych do końcowego audytu jednego wcześniej wybranego modelu. Analiza wstępna ich nie odczytuje. Większy holdout zostawia mniej danych do walidacji. Ponowne użycie obejrzanych wyników ma charakter eksploracyjny.",
    ),
    "min_development": (
        "Minimum number of monthly out-of-sample development outcomes needed to compare models. This is separate from initial training and final holdout. A larger requirement can make a short dataset infeasible.",
        "Minimalna liczba miesięcznych wyników walidacji poza próbą treningową potrzebna do porównania modeli. To osobny etap od treningu początkowego i końcowego holdoutu. Wyższy wymóg może uniemożliwić analizę krótkiego szeregu.",
    ),
    "lag_menu": (
        "Allowed predictor delays in months (0–12). Lag L uses published X(t-L) to predict Y(t+1). Zero is the origin month, not the future outcome month. The engine tests one lag per predictor; adding options multiplies model combinations. Publication delay remains a separate constraint.",
        "Dozwolone opóźnienia zmiennych objaśniających w miesiącach (0–12). Lag L używa opublikowanego X(t-L) do prognozy Y(t+1). Zero oznacza miesiąc początkowy prognozy, nie przyszły miesiąc wyniku. Silnik sprawdza jeden lag na zmienną; więcej opcji mnoży liczbę modeli. Opóźnienie publikacji pozostaje osobnym ograniczeniem.",
    ),
    "model_size": (
        "Number of distinct external predictors in each model, from 1 to 4. Pinned predictors count toward it. This is not the number of candidates or models. More predictors increase the search and may increase overfitting or collinearity.",
        "Liczba różnych zewnętrznych zmiennych objaśniających w modelu, od 1 do 4. Zmienne obowiązkowe wliczają się do limitu. To nie liczba kandydatów ani wszystkich modeli. Więcej zmiennych powiększa wyszukiwanie i może zwiększać przeuczenie lub współliniowość.",
    ),
    "size_mode": (
        "exact tests exactly model_size predictors; up_to also tests smaller feasible models, retaining all pins. The latter expands the search.",
        "exact oznacza dokładnie model_size zmiennych; up_to sprawdza też mniejsze dopuszczalne modele, zachowując wszystkie zmienne obowiązkowe. Drugi wariant poszerza wyszukiwanie.",
    ),
    "candidate_ids": (
        "Catalog variables allowed in the search, excluding the target. A shortlist of five candidates does not mean five predictors in each model. Candidates are hypotheses requiring validation.",
        "Zmienne z katalogu dopuszczone do wyszukiwania, bez zmiennej prognozowanej. Lista pięciu kandydatów nie oznacza pięciu zmiennych w każdym modelu. Kandydatury są hipotezami wymagającymi walidacji.",
    ),
    "pinned_ids": (
        "Predictors that must appear in every model. Their values and coefficients are not held constant. They must be candidates and count toward model_size.",
        "Zmienne, które muszą wystąpić w każdym modelu. Ich wartości ani współczynniki nie są zamrażane. Muszą należeć do kandydatów i wliczają się do model_size.",
    ),
    "target_id": (
        "The one catalog series Y to forecast, such as sales or margin. Changing it requires reviewing dependent candidates and transformation choices. Forecasts remain in its original units.",
        "Jeden szereg Y z katalogu, który prognozujemy, np. sprzedaż lub marża. Zmiana celu wymaga przeglądu kandydatów i transformacji. Prognozy pozostają w oryginalnych jednostkach tego szeregu.",
    ),
    "target_transform": (
        "Internal target recipe: auto, identity, difference or log_difference. Auto uses an initial-training ADF check, otherwise first differences. Forecasts are inverted to original units. Log differences require positive values and the inverse gives a conditional median. No recipe guarantees stationarity.",
        "Wewnętrzna transformacja Y: auto, identity, difference lub log_difference. Auto korzysta z ADF na początkowych danych treningowych, a przy braku potwierdzenia stacjonarności wybiera pierwsze różnice. Prognozy wracają do oryginalnych jednostek. Różnice logarytmów wymagają wartości dodatnich, a odwrócenie daje medianę warunkową. Żadna receptura nie gwarantuje stacjonarności.",
    ),
    "feature_transform": (
        "Predictor transformation: auto, unchanged identity, or first difference. Auto uses initial-training evidence. Differencing expresses changes rather than levels; already stationary data need not be differenced.",
        "Transformacja zmiennych objaśniających: auto, niezmienione wartości (identity) albo pierwsza różnica (difference). Auto korzysta z początkowej próby treningowej. Różnice opisują zmiany zamiast poziomów; danych już stacjonarnych nie trzeba automatycznie różnicować.",
    ),
    "max_models": (
        "Safety cap on the complete model/lag grid. Over-budget searches are blocked, not silently truncated. Raising it permits more candidates but does not improve their quality by itself.",
        "Limit bezpieczeństwa dla pełnej siatki modeli i lagów. Przekroczenie blokuje wyszukiwanie, zamiast potajemnie ucinać listę. Zwiększenie pozwala sprawdzić więcej wariantów, ale samo nie poprawia ich jakości.",
    ),
    "max_fits": (
        "Safety cap on total fits across models and chronological origins. One candidate needs many fits. The preflight estimate also conservatively reserves the final audit/forecast work.",
        "Limit wszystkich dopasowań dla modeli i kolejnych momentów prognozy. Jeden kandydat wymaga wielu dopasowań. Wstępne oszacowanie konserwatywnie rezerwuje też pracę na końcowy audyt i prognozę.",
    ),
    "run_mode": (
        "preliminary saves a development-only comparison without holdout, champion or next forecast. full also audits the frozen recommendation and attempts the next forecast. Both currently use the linear engine; full does not enable future ECM/monitoring features.",
        "preliminary zapisuje porównanie na walidacji, bez holdoutu, championa i następnej prognozy. full dodaje audyt wybranego modelu i próbę prognozy na kolejny miesiąc. Oba tryby używają obecnego silnika liniowego; full nie włącza przyszłych funkcji ECM ani monitoringu.",
    ),
    "share_vectors": (
        "Whether explicitly confirmed training-prefix numerical vectors may be sent to Gemini. False still allows configured metadata and bounded summaries. It does not send the entire series or authorize analysis; Langfuse receives metadata only.",
        "Czy po wyraźnym potwierdzeniu wolno wysłać do Gemini wektory liczbowe z początkowego fragmentu treningowego. False nadal dopuszcza skonfigurowane metadane i ograniczone podsumowania. Nie oznacza wysłania całego szeregu ani zgody na analizę; Langfuse dostaje tylko metadane.",
    ),
    "r2": (
        "R² compares squared model errors to variation around the sample mean. Higher is better on the same data/scale; out-of-sample R² can be negative. Training R² is not forecast accuracy.",
        "R² odnosi błędy kwadratowe modelu do zmienności wokół średniej z próby. Wyższe jest lepsze na tych samych danych i skali; R² poza próbą może być ujemne. R² treningowe nie jest miarą skuteczności przyszłej prognozy.",
    ),
    "vif": (
        "VIF measures predictor collinearity, not forecast accuracy. Larger values indicate inflated coefficient variance; compare it alongside out-of-sample errors, not as a standalone winner rule.",
        "VIF mierzy współliniowość zmiennych, nie skuteczność prognozy. Wyższe wartości oznaczają zwiększoną wariancję współczynników; porównuj je razem z błędami walidacji, nie jako samodzielną regułę zwycięstwa.",
    ),
    "mae": (
        "MAE is the average absolute forecast error, in original target units. Lower is better on matching outcomes. RMSE weights large errors more heavily; MAPE expresses percentage errors and can be undefined near zero.",
        "MAE to średni bezwzględny błąd prognozy w oryginalnych jednostkach Y. Mniejsze jest lepsze przy porównaniu tych samych wyników. RMSE mocniej waży duże błędy, a MAPE opisuje błędy procentowe i może być nieokreślone blisko zera.",
    ),
    "pareto": (
        "A Pareto model is not dominated on all declared objectives by another candidate. Here those objectives are development MAE, maximum VIF and error variability. There may be several trade-offs, not one universal winner.",
        "Model Pareto nie jest zdominowany przez innego kandydata we wszystkich zadeklarowanych kryteriach. Tutaj są to MAE na walidacji, maksymalny VIF i zmienność błędu. Może istnieć kilka kompromisów, nie jeden bezwzględny zwycięzca.",
    ),
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
    pl = state.language == "pl"
    blocks = []
    for topic in topics:
        if topic not in TOPICS:
            raise ValueError("Unknown explanation topic")
        text = f"**{topic}** — {TOPICS[topic][int(pl)]}"
        if topic in state.draft.settings:
            setting = state.draft.settings[topic]
            status = (
                {
                    "undiscussed": "nieomówione",
                    "proposed": "propozycja",
                    "confirmed": "potwierdzone",
                }.get(setting.status)
                if pl
                else setting.status
            )
            text += (
                "\n\nAktualna wartość: " if pl else "\n\nCurrent value: "
            ) + f"`{json.dumps(setting.value, ensure_ascii=False)}` ({status})."
            field = AnalysisRequest.model_fields.get(topic)
            bounds = [
                f"{symbol} {getattr(constraint, attribute)}"
                for constraint in (field.metadata if field else [])
                for attribute, symbol in (("ge", "≥"), ("le", "≤"))
                if getattr(constraint, attribute, None) is not None
            ]
            if bounds:
                text += (" Ograniczenia: " if pl else " Constraints: ") + ", ".join(bounds) + "."
        blocks.append(text)
    blocks.append(
        "Nie zmieniłem ustawień ani nie uruchomiłem analizy. Możesz dopytać albo napisać „wróć do konfiguracji”."
        if pl
        else "No settings changed and no analysis ran. Ask a follow-up or say 'back to setup'."
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
