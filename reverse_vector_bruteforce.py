""" reverse_vector_bruteforce.py

    S. Kurowski, June 2024

    Demo an inefficient yet effective "reversal" (Method 1) of an embedding vector with
    brute-force embedding+LLM iterations and without a trained vector->dictionary model.

    Stops either at a configured OpenAI cost limit or match confidence error limit.
    Requires gpt4o or gpt4 and an embedding model.

    Example vector text target and output in 20928 iterations:
        - in: "Be at one with your power, peace and joy."
        - out: ERROR 0.5900, "Be in each moment with joyous peace and power"

    No guarantees or warranties are made in this demo.
"""

# pylint: disable-msg=C0301,W0718

import time
import logging
import numpy as np
from openai import OpenAI

model_client = OpenAI()

logging.Formatter.default_msec_format = '%s.%03d'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("reverse_vector.log",'w'),
        logging.StreamHandler()
    ]
)

# encode the TARGET "mystery" vector to be reversed
# (otherwise, fetch it from somewhere), and cache it to a file
TARGET = "Be at one with your power, peace and joy."

res = model_client.embeddings.create(
    input = TARGET,
    model = "text-embedding-3-small",
    dimensions = 1536,
)
v_target = np.array(res.data[0].embedding)
np.save('reverse_vector.bin',v_target)


# uncache/load the "mystery" vector to reverse
v_target = np.load('reverse_vector.bin.npy')

# Initial guess text:
# If there's anything known about the target text,
# customize this guess with keywords or a concept
# that is likely closely-related to the target text.
TEXT = "a random guess to start!"

# MATCH_ERROR stop condition selection:
# https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
# https://www.hackmath.net/en/calculator/normal-distribution?mean=0&sd=1
# Because vector space has unit sigma = 1.0,
# a VECTOR_ERROR == |distance| <= 3.0 is a 99.7% confidence
# that the two points are distinct, or 0.3% that they are the same;
#
# Generally:
#   VECTOR_ERROR, Are-the-Same Match Confidence
#       3.0,         0.3%
#       2.0,         4.6%
#       1.0,        31.7%
#       0.667,      50.5%
#       0.6,        55.0%
#       0.5,        61.7%
#       0.333,      73.9%
#       0.2,        84%
#       0.1,        92%
#       0.01,       99.2%

# stop at the first of either:
MATCH_ERROR = 0.6 # 55% confidence or better
COST_LIMIT = 60.0 # $60 budget spent

VECTOR_ERROR = np.inf
CURRENT_BEST_TEXT = TEXT
CURRENT_BEST_ERROR = VECTOR_ERROR
GUESSES_MADE = 0
BEST_GUESSES = []
PRIOR_GUESSES = []
TOTAL_COST = 0.0 # tally $ spent

# LLM system prompt:
# A compact and -very delicate- directive, tuned by instruction coverage
# and text contraction and expansion as a function of text length
prompt = """User input is last iterative guess of an unknown text string and its vector ERROR from the unknown text.
Determine a better text string having a lower vector ERROR and write only that string in English as your entire output.
RECENT_PRIOR_GUESSES rows are ordered oldest to most recent.
Use BEST_GUESSES and RECENT_PRIOR_GUESSES to help determine your output better text string to write.
When the vector ERROR of RECENT_PRIOR_GUESSES is difficult to reduce:
- When best text is over 7 to 9 words long: try fusing words or concepts, or dropping a random word or phrase (keep spaces between words)
- When best text is only a few words long: try adding a new word or concept often related or used in the context of the lowest-error BEST_GUESSES texts meanings
Never write any text already in BEST_GUESSES or RECENT_PRIOR_GUESSES.
"""

while TOTAL_COST < COST_LIMIT:
    GUESSES_MADE += 1
    while True:
        try:
            res = model_client.embeddings.create(
                input = TEXT,
                model = "text-embedding-3-small",
                dimensions = 1536,
            )
            break
        except Exception as e_:
            logging.error("%s",e_)
            time.sleep(7)

    # the guess text embedding cost, text-embedding-3-small
    TOTAL_COST += res.usage.prompt_tokens/1000 * 0.00002

    # VECTOR_ERROR absolute vector-space distance from target
    v_text = np.array(res['data'][0]['embedding'])
    dv = v_target - v_text
    VECTOR_ERROR = np.sqrt((dv*dv).sum())

    # LLM assistant context message
    assist = f"""BEST_GUESSES:\n{str(BEST_GUESSES)}\n\nRECENT_PRIOR_GUESSES:\n{str(PRIOR_GUESSES)}\n"""
    # LLM user message of the error and text of the guess
    m = f"ERROR {VECTOR_ERROR:.4f}, \"{TEXT}\""

    if VECTOR_ERROR < CURRENT_BEST_ERROR:
        CURRENT_BEST_TEXT = TEXT
        CURRENT_BEST_ERROR = VECTOR_ERROR
        logging.info("%s",f">>> New best text: \"{CURRENT_BEST_TEXT}\", error: {CURRENT_BEST_ERROR:.6f}")
        BEST_GUESSES.append(m)
        BEST_GUESSES.sort()
        BEST_GUESSES = BEST_GUESSES[:3] # up to top 3

    if VECTOR_ERROR <= MATCH_ERROR:
        break

    while True:
        try:
            res = model_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "assistant", "content": assist},
                    {"role": "user", "content": m},
                ],
                timeout=5,
            )
            if hasattr(res.choices[0].message,'content'):
                break
        except Exception as e_:
            logging.error(e_)
            time.sleep(5)

    # guess LLM cost, gpt4o
    u = res.usage
    TOTAL_COST += (u.prompt_tokens * 0.005 + u.completion_tokens * 0.015)/1000

    # new text guess
    TEXT = res.choices[0].message.content
    PRIOR_GUESSES.append(m)
    logging.info("%s",f"{GUESSES_MADE:5d} ${TOTAL_COST:4.3f} {m}")
    if len(PRIOR_GUESSES) > 8: # tune me
        # Keep only last 8 guesses as context to control cost.
        # This must be balanced against having too few recent
        # guesses causing repeating of older guesses.
        PRIOR_GUESSES = PRIOR_GUESSES[1:]

logging.info("%s",str(BEST_GUESSES))
