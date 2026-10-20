# #145 V8 annotation-predicate exclusion: text inventory

- Before: `6c71421202794ca2b9dba71edb308cdef43867c4`; after: `8b7b3eeb04647bff19a766ecf6e79cfe407ae99d`.
- Entities: each pair's source entities and the union of its candidate targets, as the main pipeline verbalises them. D3 sums its 21 pairs.
- Words: mean whitespace-separated words per V8 text.

| dataset | side | entities | V8 text changed | share | words before | words after |
|---|---|---|---|---|---|---|
| D1 | source | 24,182 | 0 | 0.0% | 8.3 | 8.3 |
| D1 | candidate | 50,103 | 50,103 | 100.0% | 16.6 | 11.1 |
| D2 | source | 2,744 | 316 | 11.5% | 9.1 | 8.7 |
| D2 | candidate | 3,218 | 3,205 | 99.6% | 12.3 | 8.8 |
| D3 | source | 2,291 | 0 | 0.0% | 14.5 | 14.5 |
| D3 | candidate | 2,309 | 0 | 0.0% | 12.4 | 12.4 |
| D4_instance | source | 156,581 | 156,578 | 100.0% | 12.5 | 9.4 |
| D4_instance | candidate | 37,966 | 37,965 | 100.0% | 15.6 | 11.3 |
| D4_schema | source | 4,131 | 0 | 0.0% | 7.7 | 7.7 |
| D4_schema | candidate | 1,879 | 0 | 0.0% | 8.8 | 8.8 |
| D5 | source | 14,762 | 1,531 | 10.4% | 16.9 | 16.7 |
| D5 | candidate | 13,656 | 9,962 | 72.9% | 20.9 | 20.2 |
