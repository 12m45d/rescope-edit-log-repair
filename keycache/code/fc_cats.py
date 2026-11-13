"""Question-category lexicon of the existing hard negatives, COPIED VERBATIM from
08_rescope/p11_retrofit/code/hn_build.py (CATS, cats, useed): hn_build.py runs its argument parser at import,
so it cannot be imported.  fc_select.py checks that these functions reproduce the q_cats / edit_cats stored in
the existing rows.jsonl of both backbones (400 rows each)."""
from __future__ import annotations

import hashlib
import re

CATS = {
    "PERSON": "who whose whom person people actor actress musician singer director author character filmmaker composer "
              "producer player artist figure philosopher poet rapper individual cinematographer songwriter celebrity "
              "footballer writer athlete comedian president personality creator man woman girl boy politician designer "
              "star host model dancer scientist leader king queen founder performer guitarist drummer coach novelist "
              "playwright screenwriter journalist entrepreneur businessman businesswoman painter illustrator architect "
              "engineer inventor mathematician physicist chemist economist historian monarch emperor prince princess "
              "saint pope bishop general soldier astronaut wrestler boxer golfer cricketer swimmer skater driver pilot "
              "chef magician illusionist vocalist pianist violinist conductor dj lawyer judge doctor activist voice "
              "he she him her his",
    "SCREEN": "film films movie movies poster scene tv television series show sitcom episode animated anime cartoon "
              "drama documentary sequel cinema cinematic",
    "ORG": "club team company logo label record institution university college school band group organization "
           "organisation studio party network channel order agency brand corporation firm league association society "
           "institute academy department government ministry army navy football soccer basketball hockey cricket "
           "baseball rugby nfl nba sports sport franchise broadcaster publisher newspaper magazine airline bank",
    "PLACE": "city country town state island nation location place region county province district area capital "
             "landmark campus building kingdom empire prefecture territory municipality village borough river lake "
             "mountain resort park continent republic monarchy flag map where situated located taken",
    "EVENT": "event ceremony conflict olympics war battle festival tournament championship election revolution "
             "competition games",
    "AWARD": "award awards prize honor honour accolade medal category",
}
CATS = {k: set(v.split()) for k, v in CATS.items()}


def cats(q):
    w = set(re.findall(r"[a-z]+", q.lower()))
    return sorted(k for k, v in CATS.items() if w & v)


def useed_p11(uid, salt):
    """hn_build.useed: the per-edit seed of the existing rows (not used for new draws, kept for the reproduction check)."""
    return int(hashlib.sha1(f"{uid}|{salt}".encode()).hexdigest()[:8], 16)
