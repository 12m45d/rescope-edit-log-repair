"""Content words of a question string (lookup rule of PROTOCOL.md, section 6, amendment "Review round 1", item R4).

    content_words(q) -> frozenset of the content words of a question string

DEFINITION (fixed on 2026-11-11 06:27 UTC, before any score, gate decision, coverage or firing rate of the retrieval
reference or of the lookup rule existed; it was written from general knowledge of English and of the visual-question
frame, not from token statistics of the benchmark, and it is never changed afterwards)

  token         a maximal run of Unicode letters of the lower-cased string (digits, punctuation and white space separate
                tokens; "isn't" -> "isn", "t")
  content word  a token of at least two letters that is not in STOP = FUNCTION_WORDS | PICTURE_FRAME

  FUNCTION_WORDS  the English stop-word list of NLTK (its 179 entries, split at the apostrophe as the tokeniser splits
                  them) plus the modal verbs and pronouns that list lacks (could, would, may, might, must, shall,
                  cannot, whose, us)
  PICTURE_FRAME   words that name the picture or the act of depicting it, in all inflections: nearly every visual
                  question carries one ("... shown in the image?"), so without them "one shared content word" would be
                  true of any two questions about a picture.  Nouns: image, picture, photo, photograph, pic, illustration,
                  snapshot.  Verbs: show, depict, display, picture, feature, see, appear, represent, portray, illustrate,
                  present, capture, photograph; and the adjective visible.

Nothing else is removed: generic question nouns such as "name", "type", "kind", "person" ARE content words.

LOOKUP RULE (fixed at the same time; implemented in retr.py: lookup_decide)
  edit i is a candidate on a query  iff  the query has an image whose content (md5 of the file) is the md5 of the
  edit's image  AND  the query's question and the edit's question share at least one content word.
  Released edit (top-1): the candidate with the most shared content words; a tie goes to the lowest edit index.
  No calibration, no threshold, no guard.  An edit whose question has no content word can never be released (counted).
"""
from __future__ import annotations

import re

FIXED_AT = "2026-11-11 06:27 UTC"

FUNCTION_WORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its
itself they them their theirs themselves what which who whom this that these those am is are was were be been being
have has had having do does did doing a an the and but if or because as until while of at by for with about against
between into through during before after above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such no nor not only own same so than too
very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn
needn shan shouldn wasn weren won wouldn
could would may might must shall cannot whose us
""".split())

PICTURE_FRAME = frozenset("""
image images picture pictures photo photos photograph photographs pic pics illustration illustrations snapshot snapshots
show shows showed shown showing
depict depicts depicted depicting
display displays displayed displaying
pictured picturing
feature features featured featuring
see sees seen seeing
appear appears appeared appearing
represent represents represented representing
portray portrays portrayed portraying
illustrate illustrates illustrated illustrating
present presents presented presenting
capture captures captured capturing
photographed photographing
visible
""".split())

STOP = FUNCTION_WORDS | PICTURE_FRAME
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


def tokens(q: str) -> list:
    """Alphabetic tokens of a question string, lower case, in order."""
    return _TOKEN.findall((q or "").lower())


def content_words(q: str) -> frozenset:
    """The content words of a question string (module docstring)."""
    return frozenset(t for t in tokens(q) if len(t) >= 2 and t not in STOP)


def stop_list_sha1() -> str:
    """sha1 of the sorted stop list (recorded with every lookup decision)."""
    import hashlib
    return hashlib.sha1(" ".join(sorted(STOP)).encode()).hexdigest()[:12]
