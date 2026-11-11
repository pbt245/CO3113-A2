"""Coarse culinary groups of the 101 Food-101 classes, used by the error taxonomy (scripts/error_analysis.py).

An error whose true and predicted class share a group is counted as a "fine-grained sibling" confusion.
Groups follow dish type / preparation only (written from the class list, not from model confusions),
so the taxonomy does not reuse the test results it is meant to explain. Every class is in exactly one group.
"""
FOOD_GROUPS = {
    "meat_cuts": ["steak", "filet_mignon", "prime_rib", "pork_chop", "baby_back_ribs", "peking_duck", "foie_gras"],
    "cakes_creams": ["chocolate_cake", "chocolate_mousse", "tiramisu", "red_velvet_cake", "carrot_cake", "cheesecake",
                     "strawberry_shortcake", "cup_cakes", "panna_cotta", "creme_brulee", "ice_cream", "frozen_yogurt"],
    "pastry_breakfast": ["apple_pie", "bread_pudding", "baklava", "cannoli", "churros", "beignets", "donuts",
                         "french_toast", "waffles", "pancakes", "macarons"],
    "raw_fish_meat": ["beef_tartare", "tuna_tartare", "beef_carpaccio", "sashimi", "sushi", "ceviche"],
    "dumplings_rolls": ["dumplings", "gyoza", "spring_rolls", "samosa", "takoyaki"],
    "soups": ["ramen", "pho", "hot_and_sour_soup", "miso_soup", "french_onion_soup", "clam_chowder", "lobster_bisque"],
    "pasta_rice": ["spaghetti_bolognese", "spaghetti_carbonara", "lasagna", "ravioli", "gnocchi",
                   "macaroni_and_cheese", "pad_thai", "fried_rice", "risotto", "paella", "bibimbap", "chicken_curry"],
    "sandwiches_wraps": ["hamburger", "pulled_pork_sandwich", "club_sandwich", "grilled_cheese_sandwich",
                         "lobster_roll_sandwich", "croque_madame", "hot_dog", "breakfast_burrito",
                         "chicken_quesadilla", "tacos", "nachos"],
    "salads": ["greek_salad", "beet_salad", "caesar_salad", "caprese_salad", "seaweed_salad"],
    "fried": ["french_fries", "poutine", "onion_rings", "fried_calamari", "fish_and_chips", "chicken_wings",
              "falafel", "crab_cakes"],
    "seafood": ["scallops", "mussels", "oysters", "escargots", "grilled_salmon", "shrimp_and_grits"],
    "eggs": ["eggs_benedict", "omelette", "huevos_rancheros", "deviled_eggs"],
    "breads_dips": ["bruschetta", "garlic_bread", "pizza", "hummus", "guacamole", "edamame", "cheese_plate"],
}

CLASS_TO_GROUP = {c: g for g, cs in FOOD_GROUPS.items() for c in cs}
