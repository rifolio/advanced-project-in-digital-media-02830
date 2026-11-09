# reveal_v2: training data for reveal detection

`data/reveal_v2.jsonl` holds 5,858 tutor replies, each labelled `yes` (the
reply reveals the solution) or `no` (it guides without revealing). It is
built by `build_dataset.py` and checked by `shortcut_audit.py`. It is for
training and validating judges and prompts. It is not a test set: MRBench
stays the test set.

v1 (`data/gsm8k.jsonl`) was easy to solve without reading the reply: "does
the reply contain the final answer number" scored 0.958 balanced accuracy and
a bag-of-words model 0.997. v2 was built to remove those shortcuts.

## How it is built

```
MathDial train chats (real)        GSM8K train problems
  leak filter, cut before            leak filter, minus MathDial problems
  a tutor turn                       LLM writes a wrong student attempt
        \                                 /
         `------- 5,001 contexts --------'
                       |  context checks (blank turn, answer already in chat, student right)
                 4,599 contexts
                       |  generator writes a minimal pair or two free replies:
                       |  one guide (no) and one reveal (yes), style and length drawn at random
                       |  code checks (answer present, pair closeness, flip, length, key result)
                 3,332 contexts
                       |  every other model family judges both replies (baseline RevealsAnswer prompt)
                 2,929 contexts kept, 5,858 rows      403 disputed -> reveal_v2_disputed.jsonl
```

- Contexts. 1,637 kept contexts are real MathDial train chats cut just
  before a tutor turn (`context_source=mathdial_real`), 1,292 are GSM8K train
  problems with a student attempt written by an LLM
  (`context_source=gsm8k_synthetic`). The attempt model is always a different
  family from the reply model. Both sources pass `without_leaks` from
  `prepare_training_data.py`, so no MRBench problem and no MathDial test
  problem is used. GSM8K problems that are also MathDial train chats are
  skipped.
- Replies. Each context gets one `no` and one `yes` reply from the same model,
  so the label is balanced inside every context. 35% of contexts are minimal
  pairs (`pair_id`): the guide is written first, then copied into the reveal
  with one clause changed. Code checks that the pair is close (word sequence
  similarity at least 0.5) and that the change really flips the answer.
- Styles. Reveals: `final_answer`, `answer_other_form` (words, fraction,
  units), `key_result` (a new intermediate number the student has not
  found), `leading_question` ("So is it 30?"), `worked_step`. Guides:
  `guiding_question`, `conceptual_hint`, `points_to_error`, `recheck_step`,
  `hard_negative` (talks about the right numbers and steps, does not finish).
- Length and ending. Length bucket (short, medium, long) and whether the reply
  ends with a question are drawn independently of the label, then checked.
- Labels follow the strict MRBench rubric: `yes` means the reply states the
  final answer, or a calculation result that settles it. Code flags
  (`answer_in_reply`, `answer_in_words`, `new_numbers`, ...) are stored, not
  used as labels.
- Verification. Each row is judged by every model family except its own
  generator, with the baseline `RevealsAnswer` prompt (not the optimised V5
  prompt). A context is kept only if every vote matches both labels.
- Split by problem: 15% of problems go to `val` (md5 of the problem text), so
  no problem is in both splits.

## Numbers

| | rows | problems | yes share |
|---|---|---|---|
| train | 5,028 | 1,608 | 0.50 |
| val | 830 | 249 | 0.50 |

Problem overlap between train and val: none.

Contexts dropped, one reason each:

| reason | contexts |
|---|---|
| key_result without a new result | 607 |
| verifier disagreed (to disputed file) | 403 |
| answer already in context | 368 |
| pair not close | 149 |
| reveal length off | 125 |
| pair does not flip | 117 |
| key_result states the answer as digits | 106 |
| guide length off | 70 |
| answer in guide | 43 |
| leading_question without the answer | 25 |
| final_answer without the answer | 21 |
| blank turn in context | 19 |
| student already right | 15 |
| answer_other_form states the answer as digits | 4 |

By generator, context source and length (rows, yes share):

| | rows | yes |
|---|---|---|
| claude-haiku-5-5 | 2,766 | 0.50 |
| gemini-3.5-flash-lite | 2,766 | 0.50 |
| gpt-oss-120b | 326 | 0.50 |
| mathdial_real | 3,274 | 0.50 |
| gsm8k_synthetic | 2,584 | 0.50 |
| short (4 to 40 words, median 24) | 1,867 | 0.49 |
| medium (12 to 93, median 43) | 2,457 | 0.50 |
| long (80 to 257, median 134) | 1,534 | 0.52 |

Student attempts (GSM8K contexts) were written by Claude Haiku (1,088 rows),
Gemini (954) and gpt-oss-120b (542).

Styles (rows): answer_other_form 905, final_answer 619, leading_question 596,
worked_step 544, key_result 265; conceptual_hint 617, points_to_error 596,
recheck_step 596, hard_negative 570, guiding_question 550. 866 contexts are
minimal pairs.

Flags by label (share of rows):

| flag | yes | no |
|---|---|---|
| ends with ? | 0.59 | 0.59 |
| answer_in_reply (digits) | 0.57 | 0.01 |
| answer_in_words | 0.31 | 0.00 |
| new_numbers not empty | 0.67 | 0.01 |
| echo_of_student_answer | 0.27 | 0.21 |
| one_step_derivable | 0.08 | 0.19 |

Verifier agreement with the intended label, before filtering: Haiku 0.959 of
3,440 rows, Gemini 0.925 of 3,684. The disputed file has 806 rows (403
contexts); 408 rows are contested (355 `yes`, 53 `no`), mostly `key_result`
(145) and `leading_question` (94). Those are the borderline cases of the
rubric and worth a human look before use.

## Shortcut audit

`uv run shortcut_audit.py data/gsm8k.jsonl data/reveal_v2.jsonl`. Balanced
accuracy for rules, AUC for length and bag-of-words. 0.5 means no signal.

| | rows | number | new# | ends? | len | bow (train to val) | bow to MRBench |
|---|---|---|---|---|---|---|---|
| v1 gsm8k.jsonl | 5,230 | 0.958 | 0.968 | 0.502 | 0.536 | 0.997 | 0.764 |
| v2 reveal_v2.jsonl | 5,858 | 0.783 | 0.827 | 0.499 | 0.483 | 0.917 | 0.751 |
| v2 val | 830 | 0.784 | 0.833 | 0.520 | 0.489 | | |
| MRBench, all labelled | 1,655 | 0.621 | 0.773 | 0.597 | 0.537 | | |
| MRBench, Bridge | 495 | 0.500 | 0.799 | 0.681 | 0.618 | | |
| MRBench, MathDial | 1,160 | 0.894 | 0.701 | 0.581 | 0.773 | | |

- number: `yes` iff the reply contains the final answer. new#: `yes` iff it
  has a number not in the chat. ends?: `no` iff it ends with a question.
  len: longer means `yes`. bow: naive Bayes on unigrams and bigrams.
- The answer-number rule dropped from 0.958 to 0.783, between MRBench Bridge
  and MRBench MathDial. Length and question mark carry no signal.
- Bag-of-words still reaches 0.917 inside v2, but transfers to MRBench at
  only 0.751, about the same as v1. So the words it learns are mostly
  generator phrasing, not reveal behaviour.

Openers that still give the label away: "The total" x99 (97% yes), "The
correct" x75, "The final" x41, "So is" x35, "So since" x26, "Let's walk" x14
(all 100% yes), "How many" x22 (0% yes). Top yes words include "therefore"
and "the answer is", although the prompt asks to avoid stock phrases.

## Sanity check on val

The first 300 val rows (150 yes) of the previous build (before 19 blank-turn contexts were dropped, which reshuffled the file; 100 of those rows are still in the first 300 now) scored by `gemini-3.5-flash` with the
baseline prompt, a model that was not a generator or verifier:

| | balanced acc | recall | precision |
|---|---|---|---|
| answer-number rule | 0.757 | | |
| new-number rule | 0.803 | | |
| gemini-3.5-flash, baseline prompt | 0.960 | 0.973 | 0.948 |

The judge disagreed on 12 rows (8 false positives, 4 false negatives). Read
by eye, they are rubric edge cases, not wrong labels: `hard_negative` guides
that state many intermediate results but not the last one, and `key_result`
or `leading_question` reveals that give one new number. This is an upper
bound on label quality, since kept rows already passed a Gemini or Claude
vote and gemini-3.5-flash is in the Gemini family.

## Cost

Logged spend $9.61: student attempts $1.83, replies $4.61, verification
$3.17 (Gemini $5.46, Claude Haiku $3.96, gpt-oss $0.19). About $0.50 more was
spent in runs that were killed before their spend was flushed, and about
$1.30 of the total went on duplicate calls from runs that overlapped by
mistake.

## Known limits

- Only two families did most of the work. gpt-oss-120b on OpenRouter stopped
  after the account ran out of credits, so it wrote 326 rows and part of the
  GSM8K attempts. Its rows were checked by both other families; Claude and
  Gemini rows had only one verifier each.
- The generator is easy to tell from the words (one vs rest AUC 0.99 for
  Claude and Gemini). Train with both, and do not read too much into
  per-generator results.
- The new-number rule (0.827) is a little stronger than on MRBench (0.773):
  reveals tend to add a number, guides almost never do (0.67 vs 0.01).
  `hard_negative` guides help but do not fully close this.
- Reveal-style openers ("The total", "The correct") are a residual cue.
- `key_result` was the most wasteful style (713 contexts dropped by its two
  checks) and the most contested one.
- "One step left" replies are not used as positives: verifiers split on them
  and MRBench gold calls them `no`. They show up only as the
  `one_step_derivable` flag.
- `echo_of_student_answer` compares with the student's initial wrong answer,
  so it is a weak flag.
- The bag-of-words probe is naive Bayes, not TF-IDF logistic regression, to
  stay pure Python.
- Real MathDial chats are messy. Some tutors switch to a side problem ("Let's
  try something different", about 60 kept contexts) while the reply is still
  written for the original problem, 116 contexts have two tutor turns in a
  row, and 73 conversations appear twice (two MathDial chats with the same
  problem and the same first turns, different replies). Contexts with a blank
  turn are dropped.
- Hand read after the build (80 random rows): about 4% looked wrong, mostly
  `key_result` or `hard_negative` rows at the edge of the rubric.
- Labels come from the generator's intent plus code checks plus model votes.
  No human labelled these rows. About 70 pilot rows were read by eye.

## Columns

| column | meaning |
|---|---|
| id | `<context id>_guide` or `_reveal` |
| conversation | chat so far, ending on the student |
| response | the tutor reply being judged |
| label | `yes` reveals, `no` guides |
| split | `train` or `val`, by problem |
| solution, answer | reference solution and final answer |
| student_answer | the student's (wrong) answer in the context |
| style, length, words, ends_with_question | drawn reply type, length bucket, word count, ending |
| problem_id, pair_id | problem key; set for minimal pairs |
| context_source | `mathdial_real` or `gsm8k_synthetic` |
| context_generator | `human` or the model that wrote the student attempt |
| generator, tutor | model family that wrote the reply |
| answer_in_reply, answer_in_words, echo_of_student_answer, one_step_derivable, new_numbers | code flags |
| votes | verifier family to vote |
| source, label_by | `reveal_v2`; how the label was set |

`data/reveal_v2_disputed.jsonl` has the same columns plus `contested` (this
row's votes did not match its label).

## Reproduce

```bash
uv run build_dataset.py --selftest
uv run build_dataset.py --mathdial 2701 --gsm8k 2300 --threads 32 --budget 10.0
uv run shortcut_audit.py data/gsm8k.jsonl data/reveal_v2.jsonl
uv run evaluate.py --data data/reveal_v2.jsonl --split val --limit 300 --model gemini-3.5-flash
```

Stage files under `data/raw/reveal_v2_*` make reruns resume instead of paying
again. With an empty `data/raw/` and only the two default families, the rows
will differ from this file, since part of it came from gpt-oss-120b.
