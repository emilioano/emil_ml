"""ImageNet-1k label -> coarse category mapping. Formerly the cascade's
Step 1 category source; superseded by core/detection/yolo_coco's COCO
mapping (see resnet_coarse/__init__.py) but kept working and correct on
its own — this module has no dependency on the cascade being wired to it.

This is deliberately a plain module-level table, not a database row: it's
static taxonomy (which of ~1000 fixed ImageNet labels means "this is
broadly an animal"), the same category of thing as
core/registry_factory.py's own `_TRAINER_FACTORIES`/`_PREDICTOR_FACTORIES`
dispatch dicts — code-level configuration, not a per-component runtime
setting. predictor.py never hardcodes a label->category branch itself; it
only calls `categorize()`.

Matching is keyword/substring-based over decode_predictions()'s
underscore-separated label text (e.g. "sports_car", "Egyptian_cat"),
rather than one entry per exact label — ImageNet-1k has ~120 dog breeds
alone, and a substring match on "terrier"/"spaniel"/"retriever"/... covers
that whole family from a handful of keywords instead of enumerating every
breed name. A label matching no keyword falls into "other" — a normal,
expected outcome (most of ImageNet-1k is neither an animal nor a vehicle),
not a gap to keep patching.

IMPORTANT, DELIBERATE LIMITATION: there is no reliable "human" bucket
here. ImageNet-1k's 1000 classes are overwhelmingly objects and animals;
the closest things to a direct person class are a handful of
occupation/role labels ("bridegroom") and clothing/accessory items
("military_uniform", "bikini", "diaper", "sunglasses", ...) that merely
co-occur with humans in training photos — using the latter as a "human
present" signal would misfire constantly (a photo of sunglasses on a
table is not a person). Verified empirically on a real photo of a person
(scikit-image's bundled astronaut() test image): ResNet-50's top-1
prediction was "bobsled" at ~26% confidence — a plausible scene object,
not the person. Rather than curate a list that only "works" on staged
demo photos, `_HUMAN_KEYWORDS` below is left small and honestly weak,
and this limitation is surfaced in resnet_coarse/__init__.py and
settings.py rather than hidden. core/cascade does not assume the coarse
stage is good at detecting humans specifically — a stronger person-
presence model could be registered as an alternative coarse stage later
(see core/cascade/pipeline.py's module docstring) without any change to
the cascade dispatch itself, which only depends on getting *a* category
string back, never on how it was produced.
"""

from __future__ import annotations

from emil_ml.core.cascade.categories import (
    CATEGORY_ANIMAL,
    CATEGORY_HUMAN,
    CATEGORY_OTHER,
    CATEGORY_UNCERTAIN,
    CATEGORY_VEHICLE,
)

# Ordered so a more specific/earlier match wins if a label could plausibly
# match more than one group (checked in this order).
_HUMAN_KEYWORDS: tuple[str, ...] = (
    # Deliberately small — see module docstring. "bridegroom" is the one
    # ImageNet-1k label that is unambiguously a person as the main subject
    # rather than an object/accessory that merely co-occurs with one.
    "bridegroom",
)

# Explicit overrides checked BEFORE the keyword scan, for the rare cases
# where a keyword below would otherwise misfire on an unrelated label —
# e.g. "dog" (for dog breeds) would also match "hot_dog" (the frankfurter
# food class), and "cat" (for cat breeds) would also match "catamaran"
# (a boat, already in _VEHICLE_KEYWORDS via its full name). Kept as a tiny,
# explicit table rather than trying to make every keyword collision-proof
# with regex word boundaries — ImageNet-1k's WordNet-derived labels mix
# underscore-joined compounds ("Egyptian_cat") with single fused words
# ("bloodhound"), so a plain word-boundary regex would just trade these two
# known false positives for missed matches on the fused-word breed names.
_EXACT_OVERRIDES: dict[str, str] = {
    "hot_dog": CATEGORY_OTHER,
}

_ANIMAL_KEYWORDS: tuple[str, ...] = (
    "_cat", "tabby", "dog", "terrier", "hound", "spaniel", "retriever", "setter", "pointer",
    "collie", "sheepdog", "corgi", "poodle", "puppy", "wolf", "fox", "bear", "lion",
    "tiger", "leopard", "cheetah", "jaguar", "elephant", "zebra", "monkey", "ape",
    "gorilla", "orangutan", "chimpanzee", "baboon", "gibbon", "macaque", "bird",
    "eagle", "owl", "parrot", "hen", "cock", "chicken", "goose", "duck", "peacock",
    "flamingo", "stork", "crane", "pelican", "penguin", "snake", "lizard", "iguana",
    "chameleon", "gecko", "turtle", "tortoise", "crocodile", "alligator", "fish",
    "shark", "whale", "dolphin", "ray", "eel", "trout", "salmon", "goldfish",
    "horse", "zebra", "cattle", "ox", "bison", "buffalo", "sheep", "ram", "goat",
    "pig", "hog", "boar", "rabbit", "hare", "squirrel", "deer", "gazelle", "impala",
    "antelope", "kangaroo", "koala", "panda", "hyena", "otter", "beaver", "skunk",
    "mink", "weasel", "rat", "mouse", "hamster", "porcupine", "hedgehog", "sloth",
    "armadillo", "frog", "toad", "salamander", "newt", "spider", "scorpion", "tick",
    "butterfly", "moth", "beetle", "ant", "bee", "wasp", "grasshopper", "cricket",
    "cockroach", "mantis", "dragonfly", "cicada", "crab", "lobster", "crayfish",
    "snail", "slug", "jellyfish", "starfish", "sea_urchin", "llama", "camel",
)

_VEHICLE_KEYWORDS: tuple[str, ...] = (
    "car", "truck", "bus", "wagon", "cab", "convertible", "jeep", "limousine",
    "ambulance", "minivan", "pickup", "racer", "streetcar", "tram", "trailer",
    "tow_truck", "fire_engine", "garbage_truck", "police_van", "moving_van",
    "half_track", "snowplow", "golfcart", "motor_scooter", "moped", "minibike",
    "motorcycle", "bicycle", "unicycle", "wheelchair", "train", "locomotive",
    "railroad", "airliner", "airship", "warplane", "biplane", "helicopter",
    "space_shuttle", "balloon", "canoe", "gondola", "catamaran", "trimaran",
    "yawl", "schooner", "speedboat", "submarine", "lifeboat", "container_ship",
    "liner", "pirate", "sled", "sleigh", "snowmobile", "forklift", "harvester",
    "tractor", "bobsled", "carousel",
)


def categorize(imagenet_label: str) -> str:
    """Map one decode_predictions() label (e.g. "sports_car") to a coarse
    category. Substring match against the keyword tuples above, checked in
    human -> animal -> vehicle order; unmatched labels are CATEGORY_OTHER.

    Confidence gating (CATEGORY_UNCERTAIN) happens one level up, in
    predictor.py — this function only knows about label text.
    """
    normalized = imagenet_label.lower()
    if normalized in _EXACT_OVERRIDES:
        return _EXACT_OVERRIDES[normalized]
    if any(keyword in normalized for keyword in _HUMAN_KEYWORDS):
        return CATEGORY_HUMAN
    if any(keyword in normalized for keyword in _ANIMAL_KEYWORDS):
        return CATEGORY_ANIMAL
    if any(keyword in normalized for keyword in _VEHICLE_KEYWORDS):
        return CATEGORY_VEHICLE
    return CATEGORY_OTHER
