# Project report: CPU-only context compression for RAG

Ori Kessous · October to November 2026

## Summary

A RAG system retrieves passages and sends them to a language model. Most of that text does not help answer the question, and it is paid for on every query. This project is a layer that picks the few sentences worth sending. It runs on a CPU in about 55 ms, makes no LLM call, and copies sentences from the source without rewriting them.

On a held-out set of 1,327 questions:

- Sending the best 5 sentences (440 tokens) gives 65.7% correct answers.
- Plain retrieval cut to the same budget gives 57.5%. Compression wins by 7 to 13 points across budgets.
- The full retrieved context (2,463 tokens) gives 70.8%. Compression loses 4 to 5 points for 82% fewer tokens.

The project did not reach the accuracy of the full context. A review of published work and commercial products found none that does at this compression level on single-fact questions with a strong reader. The supported claim is narrower: for a fixed token budget, choosing sentences this way is clearly better than retrieving chunks.

## 1. The goal and the constraints

The starting spec ([original-spec.md](original-spec.md)) set a target of removing 90 to 95% of context tokens, with three fixed constraints:

| Constraint | Meaning |
|---|---|
| CPU first | No GPU at query time, and no LLM call inside the layer |
| Extractive only | Output is made of sentences copied from the documents, so the layer cannot invent text |
| Model agnostic | Works in front of any LLM |

The spec also listed parameter values: 100 to 120-token chunks with 25% overlap, a similarity threshold of 0.65, an empty-context threshold of 0.45, and a U-shaped ordering of the output. Checking them against the four papers they came from showed that most were illustrative and not supported by the papers. Almost all were replaced after measurement.

## 2. The pipeline as built

1. **Ingest.** Documents are converted to text, split into chunks of at most 64 tokens on sentence boundaries, and embedded with `bge-small-en-v1.5` (INT8, ONNX Runtime). Sentence embeddings are computed here, once.
2. **Retrieve.** The question is embedded and the top chunks are found by inner product.
3. **Shortlist.** Sentences in those chunks are ranked by embedding score, and the top 20 go on.
4. **Rerank.** A MiniLM-L6 cross-encoder scores each shortlisted sentence, reading it with the sentence before and after it.
5. **Select and lay out.** The best 5 sentences are kept, in document order, grouped under a short title line per document.
6. **Empty context.** If the best embedding score is 0.64 or less, the layer returns nothing.

An optional second reranker (fine-tuned Ettin, 68M parameters) rescores the best 8 sentences.

The trained reranker is published at [huggingface.co/Ori5147/context-reranker-MiniLM-L6](https://huggingface.co/Ori5147/context-reranker-MiniLM-L6).

## 3. Challenges, and what was done about each

### 3.1 The first end-to-end run was poor, for two different reasons

On real PDFs the output was mostly broken text: line fragments, table debris, missing spaces. Swapping the PDF extractor for a cleaner one fixed the text and made compression worse, with 66 to 83% of tokens kept. Clean on-topic sentences all scored between 0.65 and 0.80, so a fixed threshold could not tell an answer from a related sentence.

**Decision:** separate the two problems. Evaluate selection on clean text with known answers, and treat PDF extraction as its own issue.

### 3.2 There was no way to tell whether a change helped

Four hand-picked queries were the only test. An early single-question check also pointed the wrong way: a reading-comprehension model looked excellent on one example and was worse than the baseline over 500.

**What was built:** a two-level instrument.

| Level | What it measures | Cost |
|---|---|---|
| 1 | Does the answer sentence survive each stage: retrieved, shortlisted, ranked in the top 1, 3, 5 | Free, seconds |
| 2 | Does a reader model answer correctly from the context, graded by a judge model as correct, abstained or wrong | About $1.40 per full run on Haiku |

The dataset is Natural Questions in a multi-document form: each question comes with 20 passages, one of which contains the answer. It was split into a development half and a test half that stayed untouched until the end. Two kinds of negative questions were added: off-topic ones, and ones whose passages are on topic but lack the answer.

### 3.3 Ranking was the bottleneck

Level 1 showed where answers were lost. Retrieval was fine, with the answer retrieved for 90% of questions. Ranking was weak: the embedding model put the answer sentence first for only 24 to 26% of questions.

| Selector | Answer sentence in top 1 | In top 3 |
|---|---:|---:|
| Embedding score only | 24% | 43% |
| Off-the-shelf cross-encoder (ms-marco MiniLM-L6) | 34% | 53% |
| Distilled, 6,500 teacher rankings | 39% | 57% |
| Distilled, with neighbouring sentences as input | 41% | 59% |
| **Distilled, 35,900 rankings, with neighbours (adopted)** | **43%** | **61%** |
| Adopted model plus fine-tuned Ettin second stage | 47% | 66% |

*Development half, 1,327 questions.*

**Finding the ceiling first.** Before training anything, 300 shortlists were given to Claude models, which were asked to pick the best sentences. Haiku put the answer first for 68% of questions where the small cross-encoder managed 44%, and Sonnet reached 78%. So the task allowed much better ranking; the small model was the limit.

**Distillation.** Haiku ranked the shortlists for 35,900 questions from a separate dataset (DPR), for about $2.40 through the batch API. A MiniLM-L6 was trained on those rankings on a CPU in 5.5 hours. Bigger off-the-shelf rerankers did not help: `bge-reranker-base` scored the same as the small model and ran 8 times slower. Teacher labels mattered and size did not.

**Where it stopped.** Going from 6,500 to 35,900 labelled questions added only 2 points. Asking the teacher to rank the same shortlist twice gave the same first choice only 53% of the time, and the student's agreement with the teacher had reached that same level. Further training on these labels had nothing left to learn from.

### 3.4 Scores changed depending on batch contents

With dynamic INT8 quantization, the same sentence got a slightly different score depending on which other sentences shared its batch. The top-3 set changed for 23% of questions between runs, and the library chose different sentences than the evaluation for 17% of them.

**Fix:** run each text as its own inference on parallel single-thread workers. Results became identical between the library and the evaluation on 400 of 400 questions. It was also faster: 31 ms against 65 ms for 20 sentences.

### 3.5 Readers understood bare sentences worse than passages

On questions where both contexts contained the answer, the reader answered correctly more often from plain chunks than from selected sentences (92% against 85%). The selected sentences had lost their document titles and their order.

**Fix:** group the sentences by document under a title line, in document order. On 200 questions this gave +3 to +4 points on Haiku and +8 on a local 4B model (p < 0.01).

### 3.6 An early conclusion was noise

On 200 questions, 7 sentences appeared to match the full context (62% against 60%). A calculation of the sample size needed showed that 200 questions cannot detect a 3-point difference. The run was repeated on 1,327 questions with an LLM judge, and the result was a 3-point deficit with an interval that excluded zero.

Every later decision used the larger set and paired statistics.

### 3.7 The first comparison flattered the product

Comparing compressed context only with the full context shows "5 times fewer tokens" and hides the obvious alternative of retrieving fewer chunks. An equal-budget baseline was added: plain retrieval cut to the same number of tokens. On the first small run the two were tied at 5 and 7 sentences. Only after the titled layout and the better reranker did the product pull clearly ahead.

### 3.8 The reader already knew most of the answers

Haiku answers 61% of Natural Questions with no context at all, which blurs what the context contributes. Three things were done:

- Results are also reported for the subset the reader does not know without context.
- A new set was built from Wikipedia articles about events after the model's training cutoff, with questions written by Haiku. It turned out too easy for ranking, because written questions echo the wording of their passage (answer sentence in the top 3 for 95%, against 62% on Natural Questions). Rewording the questions lowered that to 88%. The set was kept as a side instrument and not used for conclusions.
- TriviaQA was added as a second established set that nothing had been tuned on. Its ranking difficulty matched Natural Questions.

### 3.9 Things that were tried and did not work

| Idea | Result |
|---|---|
| Chunk overlap | Hurt retrieval by about 3 points at every chunk size |
| U-shaped ordering of the output | No measured benefit; replaced by document order with titles |
| Similarity thresholds for selection | Barely affect the outcome; the cap on the number of sentences is what sets it |
| Reading-comprehension model as ranker | Worse than the cross-encoder, and worse the more candidates it saw |
| BM25 and hybrid retrieval | Neutral to worse on this data |
| Query rewriting | Within noise |
| Smoothing scores across neighbouring sentences | Worse |
| Adding neighbour sentences to the output | At equal tokens, more top-ranked sentences did better |
| A trained "is the answer here?" classifier | Could not separate answerable questions from on-topic ones without an answer (AUC 0.60) |
| Sending more sentences when the ranker is unsure | Same curve as a fixed number of sentences |
| Wider shortlist (all sentences, not 20) | +1 to +1.7 points for 2.3 times the CPU time |
| Fine-tuning the second reranker | +1.3 points over using it as published, not significant |

## 4. Final results

All Level 2 numbers: Claude Haiku as reader and judge, 1,327 answerable questions per set, paired differences with 95% intervals.

### 4.1 Natural Questions, held-out half

Run once, with the configuration frozen beforehand.

| Context | Tokens | Correct |
|---|---:|---:|
| No context | 41 | 61.5% |
| Full retrieved context | 2,463 | 70.8% |
| Plain retrieval, same budget as 3 / 5 / 7 sentences | 248 / 373 / 495 | 48.8 / 57.5 / 59.8% |
| **Compressed: 3 / 5 / 7 sentences** | 306 / 440 / 566 | **61.7 / 65.7 / 66.6%** |
| Compressed with second reranker: 3 / 5 / 7 | 305 / 438 / 567 | 65.1 / 65.6 / 67.5% |

| Comparison | 3 sentences | 5 sentences | 7 sentences |
|---|---:|---:|---:|
| Compressed minus plain retrieval at equal budget | +12.9 [10.2, 15.5] | +8.0 [5.5, 10.5] | +6.8 [4.5, 9.1] |
| Compressed minus full context | −9.1 | −5.2 [−7.4, −3.0] | −4.2 [−6.5, −2.0] |
| Second reranker minus none | +3.5 [1.5, 5.5] | +0.2 | +0.9 |

The development half had shown a gap to full context of about 3 points; the held-out half shows 4 to 5. The win over equal-budget retrieval held in full.

### 4.2 TriviaQA, no tuning

| Context | Tokens | Correct |
|---|---:|---:|
| No context | 56 | 86.8% |
| Full retrieved context | 2,517 | 75.4% |
| Plain retrieval, same budget: 3 / 5 / 7 | 263 / 382 / 497 | 56.1 / 62.5 / 67.8% |
| Compressed with second reranker: 3 / 5 / 7 | 314 / 438 / 554 | 68.7 / 70.8 / 71.9% |

Compressed minus equal-budget retrieval: +12.6, +8.2, +4.1. Compressed minus full context: −4.7 at 5 sentences, −3.5 at 7.

The reader scores higher with no context than with any context here. It knows most TriviaQA answers, and the prompt tells it to answer only from the passages, so it withholds answers it knows. This is a property of the test prompt and affects every context condition alike.

### 4.3 Questions with no answer in the passages

| Context | Wrong answers, Natural Questions | Wrong answers, TriviaQA |
|---|---:|---:|
| Full retrieved context | 37% | 9.5% |
| Compressed, 5 sentences | 37% | 7.5% |

Compression neither causes nor reduces this. The empty-context rule catches off-topic questions (398 of 400) and not on-topic ones.

### 4.4 A small local reader

The same questions as the development half, with a local 4B model (Qwen 3.5, run through Ollama) as the reader and Haiku as the judge. Published work suggested compression helps weak readers most, so this was a test of whether the gap to the full context would close.

| Context | Tokens | Correct |
|---|---:|---:|
| No context | 36 | 29.8% |
| Full retrieved context | 1,683 | 65.8% |
| Plain retrieval, same budget: 3 / 5 / 7 | 177 / 262 / 348 | 44.1 / 50.0 / 54.0% |
| Compressed: 3 / 5 / 7 sentences | 217 / 308 / 395 | 57.6 / 58.9 / 61.0% |

| Comparison | 3 sentences | 5 sentences | 7 sentences |
|---|---:|---:|---:|
| Compressed minus plain retrieval at equal budget | +13.6 [10.7, 16.4] | +8.9 [6.3, 11.5] | +7.0 [4.5, 9.5] |
| Compressed minus full context | −8.2 | −6.9 [−9.4, −4.4] | −4.7 [−7.1, −2.4] |

The gap did not close. It is somewhat larger than with Haiku on the same questions (−4.4 and −2.9 at 5 and 7 sentences). The win over equal-budget retrieval is the same size as with Haiku. What does change is how much any context is worth: this reader goes from 30% with no context to 61% with 395 tokens of compressed context.

### 4.5 Speed

Laptop CPU with 20 logical cores, INT8 ONNX models, 20 passages per question:

| Step | Time |
|---|---:|
| Embed the question | about 5 ms |
| Rerank 20 sentences | about 31 ms |
| Whole selection step, median | about 55 ms |
| With the second reranker | roughly 20 to 40 ms more |

### 4.6 Where the remaining gap comes from

When the answer reaches the reader, the compressed context does better than the full one (81.5% against 77.8% correct on the development half). The whole deficit comes from answers lost before the reader: sentences that were retrieved but did not make the shortlist of 20, and sentences in the shortlist that were not chosen. Matching the full context would need the answer sentence to arrive for roughly 79% of questions; it arrives for 69%.

## 5. How this compares with other work

A review in November 2026 of published papers and commercial products found:

- No one reports full-context accuracy at about 80% compression on single-fact questions with a strong reader. Vendor figures for a comparable commercial feature released in October 2026 show a loss of about 7 points at 75% of the context removed. Published extractive pruners report losses of 1 to 2 points, measured with different readers and metrics.
- No published CPU-budget method promises more than a few points of improvement on this kind of data.
- Published evidence does show compression helping accuracy in three settings: small local reader models, contexts of tens of thousands of tokens, and retrieval with many misleading passages. This project tested the first and did not see it (section 4.4). The other two have not been measured here.
- The equal-budget comparison and CPU latency figures reported here are not commonly published by others.

These numbers come from different datasets and readers and are not directly comparable with each other or with this project. A head-to-head run on the same questions has not been done.

## 6. Future directions

In order of evidence and cost:

1. **Fallback to the full context when the reader abstains.** Computed from the existing answer logs, with no new model calls: if the reader says the compressed context lacks the answer, ask again with the full context. This matches full-context accuracy (71.3% against 70.9% on the held-out half) at an average of 774 tokens, with a second call on 14% of queries. It also raises wrong answers by 1 to 2.4 points, and by 6 to 8 points on questions with no answer in the passages. It is an after-the-fact analysis and needs confirmation on fresh questions.
2. **Longer contexts.** All measurements used about 2,500 tokens. The case for compression should be stronger at 10 to 50 times that, and it has not been tested.
3. **Head-to-head comparison** with a commercial pruning API and an open-source pruner on the same questions.
4. **Document extraction.** Poor PDF text was the first failure seen and remains outside what was evaluated.

## 7. Cost and tools

| Item | Amount |
|---|---|
| Total API spend, all experiments | about $13 |
| Teacher labels for the adopted reranker | about $2.40 |
| One full evaluation run (reader and judge) | about $1.40 |
| Reranker training | 5.5 hours on a laptop CPU |
| Query-time hardware | CPU only |

The code is Python with strict typing (`mypy --strict`), 168 unit tests, and four runtime dependencies. It was written with Claude Code as a pair programmer. I set the constraints, chose the experiments, and made the decisions recorded above.
