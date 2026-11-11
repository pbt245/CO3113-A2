# Manual review of P1 test errors

Errors of the M2 p1_resnet50 seed-42 checkpoint (run food101-p1_resnet50-s42-20261029-195643, commit e39b497). One rater per image; Wilson 95 % intervals.

| Stratum | n | label_wrong | ambiguous | model_error |
|---|---|---|---|---|
| random | 75 | 2 (3 %, CI 1-9) | 28 (37 %, CI 27-49) | 45 (60 %, CI 49-70) |
| top_conf | 50 | 18 (36 %, CI 24-50) | 23 (46 %, CI 33-60) | 9 (18 %, CI 10-31) |

## Automatic bucket x manual verdict (both strata)

| Bucket | label_wrong | ambiguous | model_error |
|---|---|---|---|
| consensus | 0 | 0 | 0 |
| sibling | 14 | 29 | 17 |
| near_miss | 2 | 16 | 28 |
| gross | 4 | 6 | 9 |

## Persistence in the final (M3) models, seed 42

How many reviewed images each final model still gets wrong; 'all' = every pretrained model wrong, 'all, same class' = every pretrained model predicts the class the M2 model predicted.

| Verdict | n | P1 wrong | P2 wrong | P3 wrong | all wrong | all, same class |
|---|---|---|---|---|---|---|
| label_wrong | 20 | 20 (100 %) | 18 (90 %) | 19 (95 %) | 18 (90 %) | 18 (90 %) |
| ambiguous | 51 | 49 (96 %) | 41 (80 %) | 36 (71 %) | 29 (57 %) | 21 (41 %) |
| model_error | 54 | 42 (78 %) | 31 (57 %) | 30 (56 %) | 22 (41 %) | 13 (24 %) |

## Label-wrong and ambiguous examples

- `bread_pudding/1306663` (random, label bread_pudding, pred cup_cakes 0.87): **ambiguous** - several desserts incl. a cupcake
- `ceviche/3829333` (random, label ceviche, pred macaroni_and_cheese 0.42): **ambiguous** - dark, blurred, dish not identifiable
- `cheesecake/2398894` (random, label cheesecake, pred beignets 0.33): **ambiguous** - charlotte-style fruit cake
- `cheesecake/2730999` (random, label cheesecake, pred hot_dog 0.47): **ambiguous** - pastry with almonds, not clearly cheesecake
- `chicken_quesadilla/2151196` (random, label chicken_quesadilla, pred breakfast_burrito 1.00): **label_wrong** - rolled wrap = burrito, not a flat quesadilla
- `churros/1097261` (random, label churros, pred beignets 0.41): **ambiguous** - extreme close-up of fried dough
- `escargots/3860524` (random, label escargots, pred pork_chop 0.81): **ambiguous** - mixed plated dish, snails not visible
- `filet_mignon/448746` (random, label filet_mignon, pred steak 0.92): **ambiguous** - grilled steak - cut not identifiable
- `foie_gras/664014` (random, label foie_gras, pred pork_chop 0.99): **ambiguous** - meat stack, could be foie gras on beef
- `french_toast/58096` (random, label french_toast, pred apple_pie 0.49): **ambiguous** - plated dessert, not identifiable
- `greek_salad/321467` (random, label greek_salad, pred caesar_salad 0.97): **ambiguous** - green salad with chicken, no feta/olives
- `grilled_salmon/2528707` (random, label grilled_salmon, pred pork_chop 0.99): **ambiguous** - white grilled fish, not obviously salmon
- `hot_and_sour_soup/2586074` (random, label hot_and_sour_soup, pred french_onion_soup 0.51): **ambiguous** - dark soup in a cup
- `nachos/545659` (random, label nachos, pred huevos_rancheros 0.44): **ambiguous** - chips with eggs and salsa
- `onion_rings/1883549` (random, label onion_rings, pred churros 0.99): **ambiguous** - single long fried ring
- `panna_cotta/1050629` (random, label panna_cotta, pred lasagna 0.80): **ambiguous** - savory dish with tomatoes, not recognisable as panna cotta
- `peking_duck/3088221` (random, label peking_duck, pred frozen_yogurt 0.23): **ambiguous** - wrapped in paper
- `peking_duck/598982` (random, label peking_duck, pred pork_chop 0.33): **ambiguous** - meat and pepper stir-fry
- `pork_chop/2090753` (random, label pork_chop, pred steak 0.82): **ambiguous** - grilled meat with fries - cut not identifiable
- `poutine/1474056` (random, label poutine, pred french_fries 0.97): **ambiguous** - fries, no gravy or curds visible
- `poutine/2401894` (random, label poutine, pred french_fries 0.66): **ambiguous** - cheesy fries, unclear if poutine
- `ravioli/2498269` (random, label ravioli, pred caesar_salad 0.61): **ambiguous** - cluttered close-up
- `ravioli/42313` (random, label ravioli, pred bread_pudding 0.74): **ambiguous** - dark photo
- `red_velvet_cake/3465651` (random, label red_velvet_cake, pred strawberry_shortcake 1.00): **ambiguous** - cake top with red fruit, red velvet not verifiable
- `risotto/312397` (random, label risotto, pred paella 0.78): **ambiguous** - red rice with shrimp - paella-like
- `shrimp_and_grits/1595431` (random, label shrimp_and_grits, pred hummus 0.41): **ambiguous** - grits bowl next to a side dish
- `steak/108310` (random, label steak, pred filet_mignon 0.66): **ambiguous** - steak stack - cut not identifiable
- `steak/2198931` (random, label steak, pred pork_chop 0.85): **ambiguous** - bone-in grilled cut
- `tacos/2914162` (random, label tacos, pred breakfast_burrito 0.99): **label_wrong** - burrito/empanada cross-section, not a taco
- `waffles/2609134` (random, label waffles, pred oysters 0.28): **ambiguous** - banana-covered dessert, waffle hidden
- `garlic_bread/2957596` (top_conf, label garlic_bread, pred bruschetta 1.00): **label_wrong** - tomato-topped toasts = bruschetta
- `beef_tartare/2956362` (top_conf, label beef_tartare, pred beet_salad 1.00): **label_wrong** - beet slices with goat cheese = beet salad
- `scallops/643668` (top_conf, label scallops, pred oysters 1.00): **label_wrong** - oysters in shells
- `pizza/2542065` (top_conf, label pizza, pred caprese_salad 1.00): **ambiguous** - tomato, mozzarella, basil on a crust
- `sashimi/1824222` (top_conf, label sashimi, pred sushi 1.00): **label_wrong** - nigiri = sushi
- `falafel/3786422` (top_conf, label falafel, pred crab_cakes 1.00): **ambiguous** - patties on flatbread
- `guacamole/580887` (top_conf, label guacamole, pred ceviche 1.00): **ambiguous** - chopped seafood-like cubes on guacamole
- `pancakes/1008491` (top_conf, label pancakes, pred omelette 1.00): **label_wrong** - omelette
- `cheesecake/248448` (top_conf, label cheesecake, pred tiramisu 1.00): **label_wrong** - layered cocoa-dusted cake = tiramisu
- `cheesecake/280586` (top_conf, label cheesecake, pred cup_cakes 1.00): **ambiguous** - bakery display case
- `mussels/1407674` (top_conf, label mussels, pred oysters 1.00): **label_wrong** - oysters
- `steak/1449248` (top_conf, label steak, pred baby_back_ribs 1.00): **ambiguous** - rib-like grilled meat
- `steak/6261` (top_conf, label steak, pred prime_rib 1.00): **label_wrong** - prime rib slice
- `miso_soup/312662` (top_conf, label miso_soup, pred hot_and_sour_soup 1.00): **label_wrong** - egg-drop / hot-and-sour soup, not miso
- `nachos/3665796` (top_conf, label nachos, pred huevos_rancheros 1.00): **label_wrong** - fried eggs on tortillas = huevos rancheros
- `guacamole/2333696` (top_conf, label guacamole, pred risotto 1.00): **ambiguous** - yellow creamy dish, not identifiable
- `onion_rings/1225453` (top_conf, label onion_rings, pred hamburger 1.00): **ambiguous** - onion rings next to a burger
- `steak/2138503` (top_conf, label steak, pred foie_gras 1.00): **ambiguous** - dark plated items
- `chicken_curry/84451` (top_conf, label chicken_curry, pred lobster_bisque 1.00): **ambiguous** - orange creamy soup
- `caesar_salad/1256965` (top_conf, label caesar_salad, pred fried_calamari 1.00): **ambiguous** - shrimp/calamari salad with croutons
- `gnocchi/1468288` (top_conf, label gnocchi, pred ravioli 1.00): **ambiguous** - creamy pasta - gnocchi vs ravioli
- `ice_cream/1828947` (top_conf, label ice_cream, pred chocolate_cake 1.00): **ambiguous** - chocolate cake and ice cream on one plate
- `hot_dog/108287` (top_conf, label hot_dog, pred lobster_roll_sandwich 1.00): **label_wrong** - lobster roll
- `tuna_tartare/1628008` (top_conf, label tuna_tartare, pred beef_tartare 1.00): **ambiguous** - raw red tartare with yolk - fish vs beef unclear
- `apple_pie/3635782` (top_conf, label apple_pie, pred tiramisu 1.00): **label_wrong** - cheesecake-like slice, not apple pie
- `omelette/851024` (top_conf, label omelette, pred waffles 1.00): **ambiguous** - waffles and omelette on one plate
- `lobster_roll_sandwich/2325607` (top_conf, label lobster_roll_sandwich, pred hamburger 1.00): **label_wrong** - burger and fries
- `caesar_salad/695904` (top_conf, label caesar_salad, pred seaweed_salad 1.00): **ambiguous** - dark kale-like salad
- `bread_pudding/1548953` (top_conf, label bread_pudding, pred french_onion_soup 1.00): **label_wrong** - cheese-topped crock = french onion soup
- `sushi/283089` (top_conf, label sushi, pred sashimi 1.00): **label_wrong** - sashimi plate
- `beignets/521638` (top_conf, label beignets, pred donuts 1.00): **label_wrong** - sugared round donuts
- `chocolate_mousse/3152850` (top_conf, label chocolate_mousse, pred chocolate_cake 1.00): **ambiguous** - chocolate dessert with shavings
- `paella/2986195` (top_conf, label paella, pred mussels 1.00): **ambiguous** - paella topped with mussels
- `grilled_cheese_sandwich/1800478` (top_conf, label grilled_cheese_sandwich, pred club_sandwich 1.00): **ambiguous** - two sandwich plates
- `tuna_tartare/1334503` (top_conf, label tuna_tartare, pred beef_tartare 1.00): **ambiguous** - dark tartare with quail egg - fish vs beef unclear
- `carrot_cake/1779377` (top_conf, label carrot_cake, pred tiramisu 1.00): **ambiguous** - blurred dessert
- `churros/2270485` (top_conf, label churros, pred onion_rings 1.00): **label_wrong** - onion rings
- `ravioli/2067517` (top_conf, label ravioli, pred gnocchi 1.00): **ambiguous** - round dumplings - gnocchi vs ravioli
- `fried_calamari/831978` (top_conf, label fried_calamari, pred fish_and_chips 1.00): **label_wrong** - fried fish with wedges = fish and chips
- `caprese_salad/1365923` (top_conf, label caprese_salad, pred beet_salad 1.00): **ambiguous** - salad with salami, no mozzarella/tomato slices
- `tiramisu/3840092` (top_conf, label tiramisu, pred chocolate_mousse 1.00): **ambiguous** - cocoa-topped dessert in a jar
