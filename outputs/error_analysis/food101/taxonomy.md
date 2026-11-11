# Error taxonomy - food101 test set

Seed 42 for every model. Buckets are exclusive, assigned in this order: **consensus** (all pretrained models ['P1', 'P2', 'P3'] predict the same wrong class), **sibling** (true and predicted class in the same culinary group of `src/data/food_groups.py`), **near-miss** (other group, true class in top 5), **gross** (other group, not in top 5). Percentages are of the model's errors.

| ID | Experiment | Errors (rate) | Consensus | Sibling | Near-miss (top-5) | Gross | Confident (> 0.9) | Confident that are consensus/sibling | Near-dup errors |
|---|---|---|---|---|---|---|---|---|---|
| B0 | b0_simple_cnn | 7,229 (28.6 %) | 350 (4.8 %) | 2,461 (34.0 %) | 2,790 (38.6 %) | 1,628 (22.5 %) | 165 (2.3 %) | 58 % | 8 |
| P1 | p1_resnet50 | 3,535 (14.0 %) | 674 (19.1 %) | 1,154 (32.6 %) | 1,265 (35.8 %) | 442 (12.5 %) | 731 (20.7 %) | 71 % | 3 |
| P2 | p2_efficientnet_b0 | 3,817 (15.1 %) | 674 (17.7 %) | 1,202 (31.5 %) | 1,434 (37.6 %) | 507 (13.3 %) | 411 (10.8 %) | 72 % | 4 |
| P3 | p3_convnext_tiny | 2,901 (11.5 %) | 674 (23.2 %) | 957 (33.0 %) | 963 (33.2 %) | 307 (10.6 %) | 601 (20.7 %) | 75 % | 2 |

Share of errors that stay inside the true class's culinary group (diagonal of the group matrix): B0 37 %, P1 44 %, P2 42 %, P3 47 %.

## Error overlap between models (same test images)

| A | B | Errors A | Errors B | Wrong in both | Both / smaller error set |
|---|---|---|---|---|---|
| B0 | P1 | 7,229 | 3,535 | 2,700 | 76 % |
| B0 | P2 | 7,229 | 3,817 | 2,859 | 75 % |
| B0 | P3 | 7,229 | 2,901 | 2,258 | 78 % |
| P1 | P2 | 3,535 | 3,817 | 2,295 | 65 % |
| P1 | P3 | 3,535 | 2,901 | 1,995 | 69 % |
| P2 | P3 | 3,817 | 2,901 | 2,044 | 70 % |

- Wrong for **all** 4 models (B0, P1, P2, P3): 1,421 images (5.63 %).
- Oracle (at least one model right): 94.37 % accuracy - the upper bound of any selection/ensemble among these models.
- All pretrained models (P1, P2, P3) agree on the same wrong class: 674 images (2.67 % of the test set).

## 15 classes with the lowest recall (p3_convnext_tiny)

| Class | Group | Recall (%) | Most frequent wrong predictions | Errors that are sibling/consensus |
|---|---|---|---|---|
| steak | meat_cuts | 62.8 | filet_mignon (27), prime_rib (18) | 82 % |
| filet_mignon | meat_cuts | 69.2 | steak (38), pork_chop (12) | 83 % |
| apple_pie | pastry_breakfast | 71.2 | bread_pudding (18), waffles (6) | 60 % |
| foie_gras | meat_cuts | 73.2 | pork_chop (12), scallops (8) | 37 % |
| chocolate_mousse | cakes_creams | 73.6 | chocolate_cake (27), tiramisu (8) | 86 % |
| pork_chop | meat_cuts | 74.4 | steak (21), filet_mignon (7) | 73 % |
| bread_pudding | pastry_breakfast | 76.8 | french_toast (8), french_onion_soup (6) | 45 % |
| ice_cream | cakes_creams | 77.6 | frozen_yogurt (14), chocolate_mousse (11) | 71 % |
| cheesecake | cakes_creams | 78.8 | strawberry_shortcake (12), chocolate_cake (6) | 77 % |
| ceviche | raw_fish_meat | 79.6 | beet_salad (6), tuna_tartare (6) | 33 % |
| chocolate_cake | cakes_creams | 80.0 | chocolate_mousse (22), red_velvet_cake (7) | 84 % |
| huevos_rancheros | eggs | 80.4 | omelette (11), nachos (5) | 37 % |
| omelette | eggs | 80.4 | breakfast_burrito (7), huevos_rancheros (6) | 37 % |
| lasagna | pasta_rice | 81.2 | ravioli (6), french_onion_soup (6) | 40 % |
| ravioli | pasta_rice | 81.6 | gnocchi (8), escargots (4) | 48 % |
