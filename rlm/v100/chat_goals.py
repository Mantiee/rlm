"""Bind natural-language goal changes to the current local operator message."""

import re
import unicodedata


def normalized(message: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", message.casefold()) if not unicodedata.combining(c)
    ).replace("ł", "l")


LONG_CUE = re.compile(
    r"\b(?:cel\w*\s+(?:dlugotermin\w*|glown\w*|nadrzedn\w*)|"
    r"(?:dlugotermin\w*|glown\w*|nadrzedn\w*)\s+cel\w*|"
    r"dlugofalowo|docelowo|dlugoterminowo|long[ -]term\s+(?:goal|objective)|"
    r"(?:main|primary)\s+(?:goal|objective))\b"
)


def authorizes_long(message: str | None) -> bool:
    if not isinstance(message, str) or not message.strip() or "?" in message:
        return False
    text = normalized(message.strip())
    cue = LONG_CUE.search(text)
    if not cue or not re.match(
        r"^(?:prosze\s+)?(?:ustaw|zmien|zrob|przyjmij|niech|chce|chcialbym|cel|celem|glowny|glownym|nadrzedny|nadrzednym|moj|moim|nasz|naszym|twoj|twoim|od teraz|dlugofalowo|docelowo|dlugoterminowo|set|change|make|my|our|your|i want|from now|long[ -]term)\b",
        text,
    ):
        return False
    # Negated requests and quoted/reported instructions are not permission.
    prefix = text[: cue.end()]
    if re.search(
        r"\b(?:nie|not|never|dont|don't|cytat|przyklad|example|quoted|dowiedziec|wyjasnij|jak)\b",
        prefix,
    ):
        return False
    if re.match(
        r"\s*(?:(?:to|jest|is|should|ma|byl|bylo)\s+)?(?:nie|not|never)\b", text[cue.end() :]
    ):
        return False
    return True


def direct_plan(message: str) -> tuple[str, str] | None:
    """Simple declarations work without a model; less regular requests use chat parsing."""
    value = message.strip()
    text = normalized(value)
    if "?" in value:
        return None
    patterns = (
        (
            "long",
            r"(?:(?:moj|nasz|twoj) )?(?:cel dlugoterminowy|glowny cel|nadrzedny cel|cel glowny)\s*(?:to\b|jest\b|:)\s*",
        ),
        (
            "long",
            r"(?:moim|naszym|twoim) (?:celem dlugoterminowym|glownym celem|nadrzednym celem) (?:jest|ma byc)\s*",
        ),
        (
            "long",
            r"(?:prosze )?(?:ustaw|zmien|przyjmij) (?:cel dlugoterminowy|glowny cel|nadrzedny cel)(?: na\b)?\s*:?\s*",
        ),
        ("long", r"(?:dlugofalowo|dlugoterminowo|docelowo) chce\s*"),
        ("long", r"(?:my|our|your) (?:long[ -]term|main|primary) (?:goal|objective) (?:is\b|:)\s*"),
        (
            "long",
            r"(?:set|change) (?:the |my |our |your )?(?:long[ -]term|main|primary) (?:goal|objective)(?: to\b)?\s*:?\s*",
        ),
        (
            "mid",
            r"(?:moj |nasz |twoj )?(?:cel srednioterminowy|plan srednioterminowy)\s*(?:to\b|jest\b|:)\s*",
        ),
        (
            "mid",
            r"(?:prosze )?(?:ustaw|zmien) (?:cel srednioterminowy|plan srednioterminowy)(?: na\b)?\s*:?\s*",
        ),
        ("mid", r"(?:my |our |your )?(?:mid[ -]term|medium[ -]term) (?:goal|plan) (?:is\b|:)\s*"),
        (
            "short",
            r"(?:moj |nasz |twoj )?(?:cel krotkoterminowy|plan na dzis|cel na dzis)\s*(?:to\b|jest\b|:)\s*",
        ),
        (
            "short",
            r"(?:prosze )?(?:ustaw|zmien) (?:cel krotkoterminowy|plan na dzis|cel na dzis)(?: na\b)?\s*:?\s*",
        ),
        ("short", r"(?:my |our |your )?short[ -]term (?:goal|plan) (?:is\b|:)\s*"),
    )
    for horizon, pattern in patterns:
        match = re.match(pattern, text)
        if not match:
            continue
        # Preserve offsets even when terminal input uses decomposed accents.
        end = next(
            index
            for index in range(1, len(value) + 1)
            if len(normalized(value[:index])) >= match.end()
        )
        objective = value[end:].strip()
        if not objective or len(objective) > 2000:
            return None
        another_horizon = r"\b(?:na dzis|krotkotermin\w*|sredniotermin\w*|short[ -]term|mid[ -]term|medium[ -]term)\b"
        if horizon == "long" and re.search(another_horizon, normalized(objective)):
            return None
        if re.search(r"[\n;]|\.\s+\S", objective):
            return None  # Several requests need the model's separate structured actions.
        if horizon == "long" and not authorizes_long(value):
            return None
        return horizon, objective
    return None


def validate_long(message: str | None, objective: str) -> None:
    if not authorizes_long(message) or not objective.strip() or objective.strip() not in message:
        raise ValueError(
            "Long-term goal requires a clear current operator request and literal user-supplied objective"
        )
