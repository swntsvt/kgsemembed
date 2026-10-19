# #145 V8 predicate classification

- Exclusion families: `v8_exclusions_sha256` `062039673a92…` (`EXCLUSION_FAMILIES`, `verbalisation/v8.py`).
- Entities: each pair's source entities and the union of its candidate targets; a count is the number of entities the predicate touches in that direction. D3 sums its 21 pairs.
- Every predicate's decision, per dataset, side and direction: `predicate_classification.csv`.
- Declared `owl:AnnotationProperty` but kept by V8: none.

## Excluded (annotation) predicates

`rdf:type` and `rdfs:label` (family `type_and_label`, excluded before #145) are omitted.

| dataset | side | direction | predicate | family | before #145 | entities |
|---|---|---|---|---|---|---|
| D1 | candidate | in | `http://purl.org/sig/ont/fma/homonym_for` | roles_v1_annotation | kept | 22 |
| D1 | candidate | in | `http://purl.org/sig/ont/fma/homonym_of` | roles_v1_annotation | kept | 7 |
| D1 | candidate | in | `http://www.w3.org/2002/07/owl#annotatedSource` | owl_axiom_annotation | kept | 48,882 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/AAL` | roles_v1_annotation | kept | 96 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/CMA_label` | roles_v1_annotation | kept | 28 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/DK__Freesurfer_` | roles_v1_annotation | kept | 85 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/Eponym` | roles_v1_annotation | kept | 2 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/FMAID` | roles_v1_annotation | kept | 50,103 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/JHU_DTI-81` | roles_v1_annotation | kept | 43 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/JHU_White-Matter_Tractography_Atlas` | roles_v1_annotation | kept | 16 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/Neurolex` | roles_v1_annotation | kept | 73 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/RadLex_ID` | roles_v1_annotation | kept | 589 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/Talairach` | roles_v1_annotation | kept | 205 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/definition` | roles_v1_annotation | kept | 735 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/fma0329122` | roles_v1_annotation | kept | 11 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/homonym_for` | roles_v1_annotation | kept | 10 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/homonym_of` | roles_v1_annotation | kept | 15 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/non-English_equivalent` | roles_v1_annotation | kept | 5,527 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/preferred_name` | roles_v1_annotation | kept | 49,844 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/slot_synonym` | roles_v1_annotation | kept | 1 |
| D1 | candidate | out | `http://purl.org/sig/ont/fma/synonym` | roles_v1_annotation | kept | 20,271 |
| D1 | candidate | out | `http://www.w3.org/2000/01/rdf-schema#comment` | roles_v1_annotation | excluded | 39 |
| D1 | source | out | `http://www.w3.org/2004/02/skos/core#altLabel` | roles_v1_annotation | excluded | 10,003 |
| D1 | source | out | `http://www.w3.org/2004/02/skos/core#definition` | roles_v1_annotation | excluded | 86 |
| D1 | source | out | `http://www.w3.org/2004/02/skos/core#prefLabel` | roles_v1_annotation | excluded | 24,182 |
| D2 | candidate | out | `http://www.geneontology.org/formats/oboInOwl#hasDefinition` | roles_v1_annotation | kept | 802 |
| D2 | candidate | out | `http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym` | roles_v1_annotation | kept | 3,205 |
| D2 | source | out | `http://www.geneontology.org/formats/oboInOwl#hasAlternativeId` | roles_v1_annotation | kept | 21 |
| D2 | source | out | `http://www.geneontology.org/formats/oboInOwl#hasDbXref` | roles_v1_annotation | kept | 2 |
| D2 | source | out | `http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym` | roles_v1_annotation | kept | 305 |
| D3 | candidate | out | `http://www.w3.org/2000/01/rdf-schema#comment` | roles_v1_annotation | excluded | 136 |
| D3 | source | out | `http://www.w3.org/2000/01/rdf-schema#comment` | roles_v1_annotation | excluded | 274 |
| D4_instance | candidate | in | `http://dbkwik.webdatacommons.org/ontology/thumbnail` | dbkwik_page | kept | 4,967 |
| D4_instance | candidate | in | `http://xmlns.com/foaf/0.1/depiction` | foaf | kept | 5,275 |
| D4_instance | candidate | in | `http://xmlns.com/foaf/0.1/thumbnail` | foaf | kept | 4,967 |
| D4_instance | candidate | out | `http://dbkwik.webdatacommons.org/ontology/abstract` | dbkwik_page | kept | 12,665 |
| D4_instance | candidate | out | `http://dbkwik.webdatacommons.org/ontology/thumbnail` | dbkwik_page | kept | 5,807 |
| D4_instance | candidate | out | `http://dbkwik.webdatacommons.org/ontology/wikiPageExternalLink` | dbkwik_page | kept | 2,689 |
| D4_instance | candidate | out | `http://dbkwik.webdatacommons.org/ontology/wikiPageWikiLinkText` | dbkwik_page | kept | 25,408 |
| D4_instance | candidate | out | `http://purl.org/dc/elements/1.1/rights` | dublin_core | kept | 10,242 |
| D4_instance | candidate | out | `http://www.w3.org/2000/01/rdf-schema#comment` | roles_v1_annotation | excluded | 12,665 |
| D4_instance | candidate | out | `http://www.w3.org/2004/02/skos/core#altLabel` | roles_v1_annotation | excluded | 13,410 |
| D4_instance | candidate | out | `http://xmlns.com/foaf/0.1/depiction` | foaf | kept | 13,403 |
| D4_instance | candidate | out | `http://xmlns.com/foaf/0.1/thumbnail` | foaf | kept | 5,275 |
| D4_instance | source | in | `http://dbkwik.webdatacommons.org/ontology/thumbnail` | dbkwik_page | kept | 47,192 |
| D4_instance | source | in | `http://xmlns.com/foaf/0.1/depiction` | foaf | kept | 47,192 |
| D4_instance | source | in | `http://xmlns.com/foaf/0.1/thumbnail` | foaf | kept | 47,192 |
| D4_instance | source | out | `http://dbkwik.webdatacommons.org/ontology/abstract` | dbkwik_page | kept | 44,361 |
| D4_instance | source | out | `http://dbkwik.webdatacommons.org/ontology/thumbnail` | dbkwik_page | kept | 24,902 |
| D4_instance | source | out | `http://dbkwik.webdatacommons.org/ontology/wikiPageExternalLink` | dbkwik_page | kept | 7,603 |
| D4_instance | source | out | `http://dbkwik.webdatacommons.org/ontology/wikiPageWikiLinkText` | dbkwik_page | kept | 60,322 |
| D4_instance | source | out | `http://purl.org/dc/elements/1.1/rights` | dublin_core | kept | 94,384 |
| D4_instance | source | out | `http://www.w3.org/2000/01/rdf-schema#comment` | roles_v1_annotation | excluded | 44,361 |
| D4_instance | source | out | `http://www.w3.org/2004/02/skos/core#altLabel` | roles_v1_annotation | excluded | 46,104 |
| D4_instance | source | out | `http://xmlns.com/foaf/0.1/depiction` | foaf | kept | 46,099 |
| D4_instance | source | out | `http://xmlns.com/foaf/0.1/thumbnail` | foaf | kept | 47,192 |
| D4_schema | candidate | out | `http://www.w3.org/2004/02/skos/core#prefLabel` | roles_v1_annotation | excluded | 1,443 |
| D4_schema | source | out | `http://www.w3.org/2004/02/skos/core#prefLabel` | roles_v1_annotation | excluded | 3,625 |
| D5 | candidate | out | `http://schema.org/description` | schema_org | kept | 9,962 |
| D5 | candidate | out | `http://www.w3.org/2004/02/skos/core#altLabel` | roles_v1_annotation | excluded | 2,182 |
| D5 | source | in | `http://www.w3.org/2000/01/rdf-schema#seeAlso` | owl_annotation | kept | 194 |
| D5 | source | out | `http://purl.org/dc/elements/1.1/description` | dublin_core | kept | 941 |
| D5 | source | out | `http://www.w3.org/2000/01/rdf-schema#seeAlso` | owl_annotation | kept | 225 |
| D5 | source | out | `http://xmlns.com/foaf/0.1/givenName` | foaf | kept | 297 |
| D5 | source | out | `http://xmlns.com/foaf/0.1/nick` | foaf | kept | 133 |

## Kept predicates (relations and attributes)

| dataset | side | distinct kept predicates |
|---|---|---|
| D1 | candidate | 6 |
| D1 | source | 4 |
| D2 | candidate | 3 |
| D2 | source | 2 |
| D3 | candidate | 12 |
| D3 | source | 12 |
| D4_instance | candidate | 204 |
| D4_instance | source | 328 |
| D4_schema | candidate | 5 |
| D4_schema | source | 9 |
| D5 | candidate | 561 |
| D5 | source | 337 |
