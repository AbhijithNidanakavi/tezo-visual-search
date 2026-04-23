from __future__ import annotations

from collections import OrderedDict

from app.taxonomy import COLOR_WORDS


def _match_terms(query: str, labels: list[str], colors: list[str]) -> list[str]:
    lowered = query.lower()
    matches: list[str] = []
    for label in list(OrderedDict.fromkeys(labels + colors)):
        if label in lowered:
            matches.append(label)
    for color in COLOR_WORDS:
        if color in lowered and color not in matches:
            matches.append(color)
    return matches


def build_explanation(query: str, labels: list[str], colors: list[str], score: float) -> str:
    matches = _match_terms(query, labels, colors)
    label_text = ", ".join(labels[:3]) if labels else "related visual content"
    color_text = ", ".join(colors[:2]) if colors else "balanced tones"
    confidence = "high" if score >= 0.35 else "moderate"

    if matches:
        matched = ", ".join(matches[:3])
        return (
            f"This result aligns with the query through visible cues such as {matched}. "
            f"The image also scores strongly for {label_text} and is visually anchored by {color_text}, "
            f"which makes the match {confidence}-confidence."
        )

    return (
        f"This image is relevant because its overall scene semantics are closest to the query. "
        f"The strongest visual signals are {label_text}, supported by {color_text}; together they "
        f"produce a {confidence}-confidence match."
    )
