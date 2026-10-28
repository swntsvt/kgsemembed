# Response-letter paragraph: efficiency and scalability (#181)

Use this if a reviewer of this paper raises efficiency or scalability, or
adapt it for the cover letter, where it can point out that the Phase 1
concern has been addressed in advance. Fill the bracketed numbers from
`cost_D1.md` once the D1 timing run is in.

---

> *Comment: the paper lacks an efficiency and scalability analysis.*

We thank the reviewer. The study is designed as an effectiveness study: it
measures how the choice of embedding model and entity verbalisation changes
the re-ranking of a fixed candidate set, under one controlled protocol, and
it makes no efficiency or scalability claim. We agree, however, that a gain is
only interpretable alongside its cost, especially since the models compared
range from about 23M to 1.5B parameters and several conditions do not beat
the stage-1 order B0. We have therefore added a section on computational cost
(Section [X], Table [X], Figure [X]) that reports:

1. **The cost structure.** Stage 2 encodes each source and each distinct
   candidate once and compares each source with at most 20 candidates, so
   its cost is linear in the number of entities encoded. The size of the
   target graph affects it only through stage 1, which is the same for
   every condition.
2. **The encoded input size** of every condition on every dataset, in tokens
   under each model's own tokenizer, which does not depend on hardware.
3. **The wall-clock cost** of every condition on D1, our largest biomedical
   dataset, from one controlled run in which every condition reproduced its
   reported metrics exactly. Stage 2 ranges from [X] to [X] minutes, against
   [X] seconds for stage 1. [One sentence on the trade-off, e.g. the largest
   model's lead on D1 costs about [X]× the time of the biomedical model for
   [X] MRR; the best condition is [C4], at [X] minutes.]

A scalability study (larger graphs, memory, distributed or GPU-server
settings) is outside this paper's scope, and we say so in the limitations.
Our timings come from one machine and one implementation, so we present them
as relative costs between conditions rather than as throughput claims, and
we do not compare them with published systems, which are evaluated under
different protocols.
