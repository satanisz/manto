"""Bounded conversational intents; spelling recovery never grants execution permission."""

import re

from manto.providers import _plain

NUMBER_WORDS = dict(
    zip(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ],
        range(21),
        strict=True,
    )
)
NUMBER_WORDS.update({"dwa": 2, "trzy": 3, "cztery": 4, "piec": 5})


def setup_intent(message):
    """Resolve common setup requests without guessing identifiers or mutating a draft."""
    plain = re.sub(r"\s+", " ", _plain(message)).strip(" .!?")
    if plain.startswith("set {"):
        return None
    if re.search(r"\b(don't|do not|not|never|nie)\b", plain):
        return {"action": "recover"}
    # Near-miss execution requests lead to review, never directly to a numerical run.
    if re.fullmatch(
        r"(?:please )?(?:perform|performe|do|start|run|execute|begin) "
        r"(?:the )?(?:analysis|analys|analyis|analisis)(?: please)?",
        plain,
    ):
        return {"action": "prepare_analysis"}
    target = bool(re.search(r"\b(targets?|target_id|cel)\b", plain))
    suggestion = bool(
        re.match(
            r"^(?:(?:can|could|would) you |please )?(?:choose|select|pick|suggest|recommend|propose|zaproponuj|przygotuj|wybierz)\b",
            plain,
        )
    )
    if target and suggestion:
        return {"action": "propose_target"}
    if target and (
        re.match(r"^(what|which|how|show|list|explain|jaki|jakie|pokaz|co)\b", plain)
        or plain in {"target", "targets", "target options"}
    ):
        return {"action": "targets"}
    if suggestion and re.search(r"\b(lags?|lagi|opoznienia)\b", plain):
        return {"action": "lags"}
    if suggestion and (
        re.search(
            r"\b(features?|predictors?|variables?|candidates?|kandydat\w*|zmienn\w*)\b", plain
        )
        or re.match(r"^(propose|zaproponuj|przygotuj)\b", plain)
    ):
        if "per model" in plain or "all " in plain:
            return {
                "action": "recover",
                "text": "Candidate count and predictors per model are different. Use 'all candidates' for the full search pool, or 'set model_size 3' for three predictors per model.",
            }
        tokens = re.findall(r"(?<!\w)-?\d+(?!\w)|[a-z]+", plain)
        counts = [
            int(token) if re.fullmatch(r"-?\d+", token) else NUMBER_WORDS[token]
            for token in tokens
            if re.fullmatch(r"-?\d+", token) or token in NUMBER_WORDS
        ]
        if len(counts) > 1 or (counts and not 1 <= counts[0] <= 20):
            return {"action": "recover", "text": "Choose one candidate count between 1 and 20."}
        return {"action": "propose", "count": counts[0] if counts else 5}
    return None
