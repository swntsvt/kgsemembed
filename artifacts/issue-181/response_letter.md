# Response-letter paragraph: efficiency and scalability (#181)

Use this if a reviewer of this paper raises efficiency or scalability, or
adapt it for the cover letter, where it can point out that the Phase 1
concern has been addressed in advance. Numbers from `cost_D1.md` (D1 run at
bf8d7f6); the section, table and figure numbers depend on the final paper
and are left as `[sec]`, `[tab]` and `[fig]`.

---

> *Comment: the paper lacks an efficiency and scalability analysis.*

We thank the reviewer. The study is designed as an effectiveness study: it
measures how the choice of embedding model and entity verbalisation changes
the re-ranking of a fixed candidate set, under one controlled protocol, and
it makes no efficiency or scalability claim. We agree, however, that a gain is
only interpretable alongside its cost, especially since the models compared
range from about 23M to 1.5B parameters and several conditions do not beat
the stage-1 order B0. We have therefore added a section on computational cost
(Section [sec], Table [tab], Figure [fig]) that reports:

1. **The cost structure.** Stage 2 encodes each source and each distinct
   candidate once and compares each source with at most 20 candidates, so
   its cost is linear in the number of entities encoded. The size of the
   target graph affects it only through stage 1, which is the same for
   every condition.
2. **The encoded input size** of every condition on every dataset, in tokens
   under each model's own tokenizer, which does not depend on hardware.
3. **The wall-clock cost** of every condition on D1, our largest biomedical
   dataset, from one controlled run in which every condition reproduced its
   reported metrics exactly. Stage 2 ranges from 40 seconds to 54 minutes,
   against 22 seconds for stage 1. The cost is set mainly by the model and,
   for a fixed model, by the length of the verbalisation: with the same
   text, the largest model (1.5B parameters) takes 13.7 times as long as the
   biomedical one (109M) for 0.026 more MRR. The best condition on D1 (C4)
   takes 31 minutes; the cheapest condition that also beats the stage-1
   order significantly (C17) takes 3.7 minutes, at 0.021 less MRR than C4.

A scalability study (larger graphs, memory, distributed or GPU-server
settings) is outside this paper's scope, and we say so in the limitations.
Our timings come from one machine and one implementation, so we present them
as relative costs between conditions rather than as throughput claims, and
we do not compare them with published systems, which are evaluated under
different protocols.
