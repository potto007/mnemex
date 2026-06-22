"""Write-time content guards for the prehend bank.

``is_anti_give_up`` blocks experiences that codify capitulation ("data not
available", "cannot determine") from being learned, UNLESS the text is actually
a protective guard rule (e.g. "retry before concluding"), which is exactly the
kind of negative-polarity lesson worth keeping.

Generalized from FinAcumen's ``finacumen/fm/pruning_rules.py`` (finance-specific
protective patterns about tickers/lookups dropped).
"""
from __future__ import annotations

import re

ANTI_GIVE_UP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bdata\s+not\s+available\b",
        r"\bdata\s+not\s+provided\b",
        r"\bdata\s+is\s+missing\b",
        r"\bdata\s+unavailable\b",
        r"\binsufficient\s+data\b",
        r"\bcannot\s+determine\b",
        r"\bno\s+data\s+available\b",
        r"\bunable\s+to\s+find\b",
        r"\breturn\s+.*\b(unavailable|unknown)\b",
        r"\bconclude\s+.*\b(unavailable|unknown|missing)\b",
        r"\bstate\s+.*\b(unavailable|unknown|missing|absent)\b",
        # "information is missing / not present" family. The old patterns only
        # matched "data ...", so an insight phrased about "information" slipped
        # through and poisoned a v13 multihop bank with capitulation lessons.
        r"\binformation\s+is\s+(missing|absent|unavailable)\b",
        r"\binformation\s+(is\s+)?not\s+(present|available|provided|found)\b",
        # "the context is garbled/ciphertext/nonsensical/unintelligible -> give
        # up" framing. A constructive lesson would say re-read/retry/verify
        # instead, which PROTECTIVE_PATTERNS overrides below.
        r"\b(garbled|ciphertext|gibberish|nonsensical|unintelligible)\b",
        r"\bno\s+human.?readable\b",
    ]
]

PROTECTIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bretry\b.*\bbefore\b.*\bconcluding\b",
        r"\bretry\b.*\bwith\b.*\b(different|alternate|wider|relaxed)\b",
        r"\bverify\b.*\bdata\b.*\b(before|first)\b",
        r"\bdo\s+not\b.*\b(give\s*up|conclude|assume)\b",
        r"\bnever\b.*\b(give\s*up|conclude|assume)\b",
        r"\bre.?read\b",
        r"\bwiden\b.*\b(range|window)\b",
        r"\bproxy\b.*\bmetric\b",
    ]
]


# Behavioral premature-stop guards (ADR-0010 contrastive failure channel). Unlike
# ANTI_GIVE_UP_PATTERNS (capitulation WORDING), these catch shallow-search-then-
# give-up STRATEGIES a failure distiller would emit ("prefer the first and stop
# searching", "return the best available estimate", "answer with the partial
# result"). Applied ONLY to failure-channel content via is_premature_stop -- NOT
# folded into is_anti_give_up, which runs on the success path too and would then
# drop legitimate positive recipes (review note, 2026-06-22 spec).
PREMATURE_STOP_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bstop\s+searching\b",
        r"\bprefer\s+the\s+first\b",
        r"\bbest\s+available\b",
        r"\bavoid\s+exhaustive\b",
        r"\bcandidate\s+is\s+found,?\s+return\b",
        r"\bcommit\s+to\s+the\s+most\s+frequent\b",
        r"\bpartial\s+(result|answer)\b",
        r"\bnarrow(ing)?\s+(the\s+)?scope\b",
        r"\bsimplest\s+interpretation\b",
        r"\bwithout\s+verif(y|ying)\b",
    ]
]


def is_anti_give_up(text: str) -> bool:
    """True if ``text`` encodes a capitulation directive worth blocking.

    A protective directive (retry/verify/re-read before concluding) overrides
    the capitulation match, so genuine guard rules are kept.
    """
    for p in PROTECTIVE_PATTERNS:
        if p.search(text):
            return False
    for p in ANTI_GIVE_UP_PATTERNS:
        if p.search(text):
            return True
    return False


def is_premature_stop(text: str) -> bool:
    """True if ``text`` encodes a shallow-search / premature-stop strategy.

    Failure-channel ONLY (see PREMATURE_STOP_PATTERNS). A constructive directive
    (re-read/retry/verify/widen) overrides via PROTECTIVE_PATTERNS, so guards that
    say to do MORE survive. Heuristic, not complete -- the injection cap is the
    structural backstop.
    """
    for p in PROTECTIVE_PATTERNS:
        if p.search(text):
            return False
    for p in PREMATURE_STOP_PATTERNS:
        if p.search(text):
            return True
    return False
