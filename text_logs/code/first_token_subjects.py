"""Base rate of the request shape behind the exploding updates (CPU, tokenizer only; no model).

The audit of the logs finds that every request whose subject ENDS AT THE FIRST TOKEN of the request prompt (the prompt
starts with the subject and the subject is one GPT-2 token, e.g. 'Chicago is a twin city of') gets a division factor
of a few hundredths and an update tens of times larger than the pre-edit weight.  This script counts such requests in
CounterFact with the public index function (rome.repr_tools.get_words_idxs_in_templates, subtoken 'last': the index
the public code calls the lookup index), so that the reader knows how often a stream of T requests meets one.
Writes out/first_token_subjects.json.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp_common as C  # noqa: E402

C.setup_public()
from rome import repr_tools  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(C.MODEL_DIR)
data = C.load_counterfact()
idx = []
B = 2000
for a in range(0, len(data), B):
    chunk = data[a:a + B]
    r = repr_tools.get_words_idxs_in_templates(tok, [d["requested_rewrite"]["prompt"] for d in chunk],
                                               [d["requested_rewrite"]["subject"] for d in chunk], "last")
    idx += [x[0] for x in r]
hits = [i for i, x in enumerate(idx) if x == 0]
n = len(data)
rate = len(hits) / n
stream = [i for i in hits if C.STREAM_START <= i < C.STREAM_START + C.STREAM_T]
dev = [i for i in hits if C.DEV_START <= i < C.DEV_START + C.DEV_T]
pools = C.jload(os.path.join(C.OUT, "pools.json"))
out = {"written": C.now(), "definition": "public lookup index of the request prompt alone == 0 (subject = first token of the prompt)",
       "n_records": n, "n_first_token_subject": len(hits), "rate": rate,
       "in_stream_0_1999": stream, "in_dev_15000_15099": dev,
       "in_repair_pool": [i for i in hits if i in set(pools["repair_records"])],
       "in_heldout_pool": [i for i in hits if i in set(pools["heldout_records"])],
       "first_in_data_order": hits[:10],
       "probability_that_T_consecutive_requests_contain_none": {str(T): (1 - rate) ** T for T in (100, 173, 500, 1000, 2000)},
       "examples": [data[i]["requested_rewrite"]["prompt"].format(data[i]["requested_rewrite"]["subject"]) for i in hits[:8]]}
# cross-check with the logged division factors of the stream
audit = C.jload(os.path.join(C.OUT, "log_audit.json"))["sources"]["stream"]["per_update"]
out["stream_check"] = {"division_factor_of_these_requests": {str(i): audit[i]["division_factor"] for i in stream if i < len(audit)},
                       "update_fro_of_these_requests": {str(i): audit[i]["update_fro"] for i in stream if i < len(audit)},
                       "smallest_division_factor_of_all_other_requests": min(p["division_factor"] for p in audit if p["index"] not in set(stream)),
                       "lookup_index_agrees_with_the_run": all((audit[i]["lookup_idx_request"] == 0) == (i in set(stream)) for i in range(len(audit)))}
C.jdump(out, os.path.join(C.OUT, "first_token_subjects.json"))
print({k: v for k, v in out.items() if k not in ("in_repair_pool", "in_heldout_pool")})
print("pools:", len(out["in_repair_pool"]), len(out["in_heldout_pool"]))
