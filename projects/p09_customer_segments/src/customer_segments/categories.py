"""Product categories from the item descriptions, by keyword rules.

The release has no category column. Each description is matched against the rules in order and the
first match wins, so "CHRISTMAS T-LIGHT HOLDER" is a Christmas item and "WHITE HANGING HEART T-LIGHT
HOLDER" a candle and light item. Descriptions that match no rule fall in "other"; the audit reports
the share of spend that does.
"""

import re

import pandas as pd

OTHER = "other"

RULES: list[tuple[str, list[str]]] = [
    ("christmas", [
        "CHRISTMAS", "XMAS", "ADVENT", "SANTA", "REINDEER", "SNOWFLAKE", "NATIVITY", "SLEIGH"
    ]),
    ("lights_candles", [
        "T-LIGHT", "TLIGHT", "LIGHT", "LIGHTS", "LANTERN", "CANDLE", "CANDLES", "CHOCOLATECANDLE", "LAMP",
        "TORCH"
    ]),
    ("bags", [
        "BAG", "BAGS", "TOTE", "SHOPPER", "CHARLOTTE", "PURSE"
    ]),
    ("kitchen_dining", [
        "MUG", "MUGS", "CUP", "CUPS", "TEA", "TEACUP", "PLATE", "PLATES", "BOWL", "BOWLS", "CAKE",
        "BAKING", "KITCHEN", "JAR", "LUNCH", "BOTTLE", "CUTLERY", "SPOON", "NAPKINS", "COASTER",
        "COASTERS", "APRON", "TRAY", "JUG", "CAKESTAND", "RECIPE", "EGG", "JAM", "POPCORN", "TEAPOT",
        "BREAD", "PANTRY", "MOULDS", "COOKIE", "CUTTERS", "PAN"
    ]),
    ("stationery_wrap", [
        "CARD", "CARDS", "WRAP", "PAPER", "NOTEBOOK", "PEN", "PENS", "PENCIL", "PENCILS", "STICKER",
        "STICKERS", "RIBBON", "RIBBONS", "TAG", "TAGS", "ENVELOPE", "ENVELOPES", "TISSUE", "TISSUES",
        "CHALK", "ERASER", "BOOK", "NOTEBOOKS", "SCISSOR", "MEMO"
    ]),
    ("toys_crafts", [
        "TOY", "TOYS", "GAME", "GAMES", "DOLL", "DOLLY", "CHILDREN", "CHILDRENS", "CHILDS", "KIDS",
        "SPACEBOY", "JIGSAW", "PLAYING", "PUZZLE", "TEDDY", "BABY", "SKIPPING", "CIRCUS", "DINOSAUR",
        "GLIDERS", "COLOURING", "MODELLING", "SKITTLES", "BINGO", "SPINNING", "FELTCRAFT", "SEWING",
        "KNITTING", "PIGGY"
    ]),
    ("party", [
        "PARTY", "BUNTING", "BALLOON", "BALLOONS", "STRAWS", "CONFETTI", "BIRTHDAY", "PIRATE"
    ]),
    ("storage", [
        "BOX", "BOXES", "TIN", "TINS", "BASKET", "CRATE", "DRAWER", "HOOK", "HOOKS", "HANGER", "STORAGE",
        "CABINET", "SHELF", "CASE", "CASES", "RACK", "PEG", "PEGS", "CHEST", "CRATES"
    ]),
    ("garden_outdoor", [
        "GARDEN", "PARASOL", "BIRD", "BIRDHOUSE", "PICNIC", "WATERING", "PLANT", "PLANTER", "FLOWER",
        "POT", "POTS", "UMBRELLA", "HAMMOCK", "WINDMILL"
    ]),
    ("jewellery_accessories", [
        "NECKLACE", "NECKLAC", "BRACELET", "EARRINGS", "RING", "RINGS", "HAIR", "BANGLE", "BROOCH",
        "CHARM", "JEWEL", "DIAMANTE", "SCARF", "HANDBAG", "HAND WARMER", "LIP GLOSS", "CLIPS", "FAN"
    ]),
    ("home_decor", [
        "FRAME", "SIGN", "CLOCK", "MIRROR", "CUSHION", "DOORMAT", "HEART", "WALL", "ORNAMENT",
        "DECORATION", "VINTAGE", "HANGING", "WREATH", "DOORSTOP", "BLANKET", "THROW", "VASE", "PLAQUE",
        "BUNNY", "DOOR MAT", "BLOCK WORD", "CHALKBOARD", "BLACK BOARD", "DUCKS"
    ]),
]  # fmt: skip

CATEGORIES = [name for name, _ in RULES] + [OTHER]


def _pattern(words: list[str]) -> re.Pattern:
    # Whole words only, so that RING does not match SPRING; the longer phrases first.
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"(?<![A-Z])(?:" + "|".join(escaped) + r")(?![A-Z])")


PATTERNS = [(name, _pattern(words)) for name, words in RULES]


def categorize(description: str) -> str:
    text = str(description).upper()
    for name, pattern in PATTERNS:
        if pattern.search(text):
            return name
    return OTHER


def categorize_series(descriptions: pd.Series) -> pd.Series:
    """Category per row; each distinct description is matched once."""
    unique = pd.Series(descriptions.fillna("").astype(str).unique())
    lookup = dict(zip(unique, unique.map(categorize), strict=True))
    return descriptions.fillna("").astype(str).map(lookup)
