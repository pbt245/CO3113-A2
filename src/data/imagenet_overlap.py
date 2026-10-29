"""Food-101 classes that have a direct counterpart among the ImageNet-1k classes.

Used to split per-class F1 into "seen during pretraining" vs "not seen" (proposal section 2.3).
Only same-dish matches are listed; ImageNet classes for raw ingredients or live animals
(e.g. "American lobster", "Dungeness crab", "strawberry") are not counted as counterparts.
"""
FOOD101_TO_IMAGENET = {
    "guacamole": "guacamole",
    "hamburger": "cheeseburger",
    "hot_dog": "hotdog, hot dog, red hot",
    "ice_cream": "ice cream, icecream",
    "pizza": "pizza, pizza pie",
    "spaghetti_carbonara": "carbonara",
    "breakfast_burrito": "burrito",
}
