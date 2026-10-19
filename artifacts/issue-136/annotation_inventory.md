# #136 annotation predicate fix: V1/V2 text inventory

- Before: `706d4e1083a79bbd7bfd3f2aa3950268c174ac8e`; after: `0b99aef2c1e8954396b9532cb7a5debae2c1c3b6`.
- Entities: each pair's source entities and the union of its candidate targets, as the main pipeline verbalises them. D3 sums its 21 pairs.
- Words: mean whitespace-separated words per text.

| dataset | side | strategy | entities | text changed | share | words before | words after |
|---|---|---|---|---|---|---|---|
| D1 | source | V1 | 24,182 | 153 | 0.6% | 8.1 | 8.2 |
| D1 | source | V2 | 24,182 | 162 | 0.7% | 9.0 | 9.1 |
| D1 | candidate | V1 | 50,103 | 20,272 | 40.5% | 5.9 | 9.7 |
| D1 | candidate | V2 | 50,103 | 20,533 | 41.0% | 7.0 | 10.8 |
| D2 | source | V1 | 2,744 | 305 | 11.1% | 2.6 | 3.0 |
| D2 | source | V2 | 2,744 | 305 | 11.1% | 3.6 | 4.0 |
| D2 | candidate | V1 | 3,218 | 3,205 | 99.6% | 1.0 | 6.6 |
| D2 | candidate | V2 | 3,218 | 3,205 | 99.6% | 2.0 | 14.7 |
| D3 | source | V1 | 2,291 | 0 | 0.0% | 2.3 | 2.3 |
| D3 | source | V2 | 2,291 | 0 | 0.0% | 4.2 | 4.2 |
| D3 | candidate | V1 | 2,309 | 0 | 0.0% | 2.2 | 2.2 |
| D3 | candidate | V2 | 2,309 | 0 | 0.0% | 3.4 | 3.4 |
| D4_instance | source | V1 | 156,581 | 0 | 0.0% | 4.9 | 4.9 |
| D4_instance | source | V2 | 156,581 | 0 | 0.0% | 22.9 | 22.9 |
| D4_instance | candidate | V1 | 37,966 | 0 | 0.0% | 3.8 | 3.8 |
| D4_instance | candidate | V2 | 37,966 | 0 | 0.0% | 21.7 | 21.7 |
| D4_schema | source | V1 | 4,131 | 0 | 0.0% | 4.6 | 4.6 |
| D4_schema | source | V2 | 4,131 | 0 | 0.0% | 5.6 | 5.6 |
| D4_schema | candidate | V1 | 1,879 | 0 | 0.0% | 2.6 | 2.6 |
| D4_schema | candidate | V2 | 1,879 | 0 | 0.0% | 3.6 | 3.6 |
| D5 | source | V1 | 14,762 | 0 | 0.0% | 1.0 | 1.0 |
| D5 | source | V2 | 14,762 | 0 | 0.0% | 2.0 | 2.0 |
| D5 | candidate | V1 | 13,656 | 0 | 0.0% | 2.3 | 2.3 |
| D5 | candidate | V2 | 13,656 | 0 | 0.0% | 3.2 | 3.2 |
