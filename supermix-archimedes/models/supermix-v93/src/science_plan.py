"""Strict, deterministic scientific-scenario plans for Supermix.

The module recognises two deliberately small closed-world models:

* constant-acceleration kinematics for final velocity or displacement; and
* the ideal-gas equation for pressure, volume, temperature, or amount.

Natural-language input is admitted only when one complete request supplies an
explicit model assumption, one target, and every required labelled quantity.
The resulting plan is a non-executable JSON-safe object bound to a versioned
formula registry and to SHA-256 digests of its source spans.  Execution uses
only ``Fraction`` and ``Decimal`` conversions: there is no ``eval``, generated
code, network access, mutable cache, or external dependency.

The verifier proves arithmetic consistency *within the stated model*.  It does
not establish that constant acceleration or ideal-gas behaviour describes the
real world, so every successful result remains explicitly model-conditional.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any, Optional


SCIENCE_PLAN_SCHEMA_VERSION = "supermix-science-plan-v1"
SCIENCE_PLAN_ENGINE_VERSION = "supermix-science-plan-engine-v1"
SCIENCE_FORMULA_REGISTRY_VERSION = "supermix-science-formula-registry-v1"
SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION = "supermix-science-plan-receipt-v1"

MAX_QUERY_CHARS = 2_000
MAX_QUANTITIES = 8
MAX_PLAN_STEPS = 4
MAX_LITERAL_DIGITS = 40
MAX_ABS_DECIMAL_EXPONENT = 100
MAX_RESULT_BITS = 4_096


_DIMENSION_ORDER = ("mass", "length", "time", "temperature", "amount")
_DIMENSION_VECTORS = {
    "velocity": (0, 1, -1, 0, 0),
    "acceleration": (0, 1, -2, 0, 0),
    "time": (0, 0, 1, 0, 0),
    "length": (0, 1, 0, 0, 0),
    "pressure": (1, -1, -2, 0, 0),
    "volume": (0, 3, 0, 0, 0),
    "temperature": (0, 0, 0, 1, 0),
    "amount": (0, 0, 0, 0, 1),
    "gas_constant": (1, 2, -2, -1, -1),
}


_FORMULA_REGISTRY_ROWS = (
    {
        "formula_id": "constant_acceleration.final_velocity",
        "scenario": "constant_acceleration",
        "target": "final_velocity",
        "target_symbol": "v",
        "target_dimension": "velocity",
        "target_unit": "m/s",
        "inputs": ("u", "a", "t"),
        "input_dimensions": ("velocity", "acceleration", "time"),
        "required_assumptions": ("constant_acceleration",),
        "equation": "v=u+a*t",
        "domain": "t>0",
    },
    {
        "formula_id": "constant_acceleration.displacement",
        "scenario": "constant_acceleration",
        "target": "displacement",
        "target_symbol": "s",
        "target_dimension": "length",
        "target_unit": "m",
        "inputs": ("u", "a", "t"),
        "input_dimensions": ("velocity", "acceleration", "time"),
        "required_assumptions": ("constant_acceleration",),
        "equation": "s=u*t+(a*t^2)/2",
        "domain": "t>0",
    },
    {
        "formula_id": "ideal_gas.pressure",
        "scenario": "ideal_gas",
        "target": "pressure",
        "target_symbol": "P",
        "target_dimension": "pressure",
        "target_unit": "Pa",
        "inputs": ("V", "n", "T"),
        "input_dimensions": ("volume", "amount", "temperature"),
        "required_assumptions": ("ideal_gas",),
        "equation": "P*V=n*R*T",
        "domain": "P>0,V>0,n>0,T>0",
    },
    {
        "formula_id": "ideal_gas.volume",
        "scenario": "ideal_gas",
        "target": "volume",
        "target_symbol": "V",
        "target_dimension": "volume",
        "target_unit": "m^3",
        "inputs": ("P", "n", "T"),
        "input_dimensions": ("pressure", "amount", "temperature"),
        "required_assumptions": ("ideal_gas",),
        "equation": "P*V=n*R*T",
        "domain": "P>0,V>0,n>0,T>0",
    },
    {
        "formula_id": "ideal_gas.temperature",
        "scenario": "ideal_gas",
        "target": "temperature",
        "target_symbol": "T",
        "target_dimension": "temperature",
        "target_unit": "K",
        "inputs": ("P", "V", "n"),
        "input_dimensions": ("pressure", "volume", "amount"),
        "required_assumptions": ("ideal_gas",),
        "equation": "P*V=n*R*T",
        "domain": "P>0,V>0,n>0,T>0",
    },
    {
        "formula_id": "ideal_gas.amount",
        "scenario": "ideal_gas",
        "target": "amount",
        "target_symbol": "n",
        "target_dimension": "amount",
        "target_unit": "mol",
        "inputs": ("P", "V", "T"),
        "input_dimensions": ("pressure", "volume", "temperature"),
        "required_assumptions": ("ideal_gas",),
        "equation": "P*V=n*R*T",
        "domain": "P>0,V>0,n>0,T>0",
    },
)


def _registry_json_value() -> dict[str, Any]:
    formulas = []
    for raw in _FORMULA_REGISTRY_ROWS:
        formulas.append(
            {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in raw.items()
            }
        )
    return {
        "schema_version": SCIENCE_FORMULA_REGISTRY_VERSION,
        "dimension_order": list(_DIMENSION_ORDER),
        "dimensions": {
            key: list(value) for key, value in sorted(_DIMENSION_VECTORS.items())
        },
        "formulas": formulas,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


SCIENCE_FORMULA_REGISTRY_CANONICAL_JSON = _canonical_json(_registry_json_value())
SCIENCE_FORMULA_REGISTRY_SHA256 = hashlib.sha256(
    SCIENCE_FORMULA_REGISTRY_CANONICAL_JSON.encode("utf-8")
).hexdigest()
FORMULA_REGISTRY_SHA256 = SCIENCE_FORMULA_REGISTRY_SHA256

_FORMULA_BY_ID = {row["formula_id"]: row for row in _FORMULA_REGISTRY_ROWS}
_FORMULA_BY_TARGET = {
    (row["scenario"], row["target"]): row for row in _FORMULA_REGISTRY_ROWS
}

# R is exact because both N_A and k_B are exact in the 2019 SI definition.
_MOLAR_GAS_CONSTANT = Fraction(831446261815324, 100000000000000)

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(_NUM)
_HEX_64_RE = re.compile(r"[0-9a-f]{64}")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")

_VELOCITY_UNIT = (
    r"(?:kilomet(?:er|re)s?\s+per\s+hour|meters?\s+per\s+second|"
    r"metres?\s+per\s+second|centimet(?:er|re)s?\s+per\s+second|"
    r"miles?\s+per\s+hour|km\s*/\s*h|cm\s*/\s*s|m\s*/\s*s|mph)"
)
_ACCELERATION_UNIT = (
    r"(?:meters?\s+per\s+second\s+squared|metres?\s+per\s+second\s+squared|"
    r"kilomet(?:er|re)s?\s+per\s+hour\s+squared|"
    r"centimet(?:er|re)s?\s+per\s+second\s+squared|"
    r"feet\s+per\s+second\s+squared|"
    r"km\s*/\s*h\s*(?:\^\s*2|²)|cm\s*/\s*s\s*(?:\^\s*2|²)|"
    r"ft\s*/\s*s\s*(?:\^\s*2|²)|m\s*/\s*s\s*(?:\^\s*2|²))"
)
_TIME_UNIT = r"(?:seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|hr|h)"
_PRESSURE_UNIT = r"(?:megapascals?|kilopascals?|pascals?|MPa|kPa|Pa|bars?|atmospheres?|atm)"
_VOLUME_UNIT = (
    r"(?:cubic\s+meters?|cubic\s+metres?|cubic\s+centimeters?|"
    r"cubic\s+centimetres?|millilit(?:er|re)s?|lit(?:er|re)s?|"
    r"m\s*(?:\^\s*3|³)|cm\s*(?:\^\s*3|³)|mL|ml|L|l)"
)
_TEMPERATURE_UNIT = r"(?:kelvins?|K|degrees?\s+celsius|celsius|°\s*C)"
_AMOUNT_UNIT = r"(?:kilomoles?|millimoles?|moles?|kmol|mmol|mol)"

_INITIAL_VELOCITY_RE = re.compile(
    rf"\binitial\s+velocity\s*(?:of|is|equals?|=|:)?\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_VELOCITY_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_REST_RE = re.compile(r"\b(?:starts?\s+from\s+rest|initially\s+at\s+rest)\b", re.IGNORECASE)
_ACCELERATION_RE = re.compile(
    rf"\b(?:acceleration\s*(?:of|is|equals?|=|:)?|accelerat(?:es|ing|ed)\s+at)\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_ACCELERATION_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_TIME_RE = re.compile(
    rf"\b(?:for|over|during|time\s*(?:of|is|equals?|=|:)?)\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_TIME_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_PRESSURE_RE = re.compile(
    rf"\b(?:pressure\s*(?:of|is|equals?|=|:)?|at)\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_PRESSURE_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_VOLUME_RE = re.compile(
    rf"\bvolume\s*(?:of|is|equals?|=|:)?\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_VOLUME_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    rf"\b(?:temperature\s*(?:of|is|equals?|=|:)?|at)\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_TEMPERATURE_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(
    rf"\b(?:amount(?:\s+of\s+substance)?\s*(?:of|is|equals?|=|:)?|contains?|has)\s*"
    rf"(?P<value>{_NUM})\s*(?P<unit>{_AMOUNT_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_CONSTANT_ASSUMPTION_RE = re.compile(
    r"\b(?:(?:assuming|under|with)\s+(?:(?:an?|the)\s+)?constant\s+acceleration|"
    r"uniform\s+acceleration|uniformly\s+accelerat(?:es|ing|ed)|"
    r"accelerat(?:es|ing|ed)\s+uniformly)\b",
    re.IGNORECASE,
)
_IDEAL_GAS_ASSUMPTION_RE = re.compile(
    r"\b(?:(?:assuming|under|using)\s+(?:(?:an?|the)\s+)?ideal\s+gas"
    r"(?:\s+(?:law|model|equation))?|according\s+to\s+the\s+ideal\s+gas\s+law)\b",
    re.IGNORECASE,
)

_TARGET_PATTERNS = {
    ("constant_acceleration", "final_velocity"): re.compile(
        r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?final\s+velocity\b",
        re.IGNORECASE,
    ),
    ("constant_acceleration", "displacement"): re.compile(
        r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?displacement\b",
        re.IGNORECASE,
    ),
    ("ideal_gas", "pressure"): re.compile(
        r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?pressure\b",
        re.IGNORECASE,
    ),
    ("ideal_gas", "volume"): re.compile(
        r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?volume\b",
        re.IGNORECASE,
    ),
    ("ideal_gas", "temperature"): re.compile(
        r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?temperature\b",
        re.IGNORECASE,
    ),
    ("ideal_gas", "amount"): re.compile(
        r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?"
        r"(?:amount(?:\s+of\s+substance)?|number\s+of\s+moles)\b",
        re.IGNORECASE,
    ),
}
_REQUEST_CUE_RE = re.compile(r"\b(?:calculate|find|determine)\b|\bwhat\s+is\b", re.IGNORECASE)
_QUESTION_TARGET_PATTERNS = {
    "constant_acceleration": re.compile(
        r"\b(?:final\s+velocity|displacement)\b", re.IGNORECASE
    ),
    "ideal_gas": re.compile(
        r"\b(?:pressure|volume|temperature|amount(?:\s+of\s+substance)?|"
        r"number\s+of\s+moles)\b",
        re.IGNORECASE,
    ),
}

_HIGH_STAKES_OR_OPEN_WORLD_RE = re.compile(
    r"\b(?:patient|clinical|medical|medicine|medication|drug|dose|dosage|"
    r"ventilator|blood|diagnos\w*|emergency|life\s+support|safety-critical|"
    r"reactor|pressure\s+vessel|market|stock|investment|weather|climate|"
    r"forecast|predict(?:ion|ive)?|tomorrow|future\s+outcome|guarantee(?:d)?|"
    r"certainty)\b",
    re.IGNORECASE,
)
_PROMPT_CONTROL_RE = re.compile(
    r"\b(?:ignore|disregard|override|system\s+prompt|developer\s+message|"
    r"instructions?|instead|also\s+(?:write|calculate|find|determine)|"
    r"write\s+(?:code|a\s+poem|an\s+email)|run\s+code|execute|http|https|www)\b",
    re.IGNORECASE,
)
_ALLOWED_QUERY_CHARS_RE = re.compile(r"[A-Za-z0-9\s.,?:=+\-/^°²³·()]+")

_KINEMATICS_ALLOWED_WORDS = frozenset(
    "assuming under with a an the constant uniform uniformly acceleration accelerates "
    "accelerating accelerated motion object particle body car train starts start starting "
    "from rest initially at has have having initial velocity of is equals equal and for "
    "over during time what calculate find determine its final displacement given using "
    "kinematic model scenario second seconds sec secs minute minutes min hour hours hr hrs"
    .split()
)
_IDEAL_GAS_ALLOWED_WORDS = frozenset(
    "assuming under using according to a an the ideal gas law model equation sample contains "
    "contain has have amount of substance is equals equal at with and pressure volume "
    "temperature what calculate find determine its number moles given state scenario"
    .split()
)

_REASON_ALLOWLIST = frozenset(
    {
        "verified_science_plan",
        "empty_query",
        "query_too_long",
        "invalid_query_text",
        "high_stakes_or_open_world",
        "prompt_control_or_mixed_request",
        "missing_explicit_assumption",
        "ambiguous_scenario",
        "multiple_targets",
        "unsupported_target",
        "missing_or_ambiguous_quantity",
        "mixed_or_unconsumed_request",
        "invalid_quantity_domain",
        "invalid_plan",
        "verification_failed",
        "numeric_limit",
    }
)


class _ScienceLimit(ValueError):
    """Raised internally when a numeric or structural bound is exceeded."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX_64_RE.fullmatch(value) is not None


def _guard_fraction(value: Fraction) -> Fraction:
    if (
        abs(value.numerator).bit_length() > MAX_RESULT_BITS
        or value.denominator.bit_length() > MAX_RESULT_BITS
    ):
        raise _ScienceLimit("Science-plan numeric result exceeds its bound.")
    return value


def _parse_fraction(value: Any) -> Fraction:
    if not isinstance(value, str) or not value or len(value) > 96:
        raise ValueError("Science-plan number must be bounded text.")
    if re.fullmatch(_NUM, value) is not None:
        try:
            decimal = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("Science-plan number is invalid.") from exc
        sign, digits, exponent = decimal.as_tuple()
        del sign
        if len(digits) > MAX_LITERAL_DIGITS or abs(exponent) > MAX_ABS_DECIMAL_EXPONENT:
            raise _ScienceLimit("Science-plan number exceeds its literal bound.")
        return _guard_fraction(Fraction(decimal))
    fraction_match = re.fullmatch(r"(?P<n>-?\d+)/(?P<d>\d+)", value)
    if fraction_match is None:
        raise ValueError("Science-plan exact number is not canonical.")
    numerator_text = fraction_match.group("n")
    denominator_text = fraction_match.group("d")
    if len(numerator_text.lstrip("-")) > MAX_LITERAL_DIGITS * 4 or len(denominator_text) > MAX_LITERAL_DIGITS * 4:
        raise _ScienceLimit("Science-plan fraction exceeds its literal bound.")
    denominator = int(denominator_text)
    if denominator == 0:
        raise ValueError("Science-plan fraction denominator is zero.")
    return _guard_fraction(Fraction(int(numerator_text), denominator))


def _fraction_text(value: Fraction) -> str:
    guarded = _guard_fraction(value)
    if guarded.denominator == 1:
        return str(guarded.numerator)
    return f"{guarded.numerator}/{guarded.denominator}"


def _finite_decimal_places(value: Fraction) -> Optional[int]:
    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return None
    return max(twos, fives)


def _display_number(value: Fraction) -> tuple[str, bool]:
    if value.denominator == 1:
        return str(value.numerator), False
    places = _finite_decimal_places(value)
    if places is not None and places <= 16:
        with localcontext() as context:
            context.prec = max(32, places + len(str(abs(value.numerator))) + 4)
            rendered = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered, False
    with localcontext() as context:
        context.prec = 16
        rendered = format(Decimal(value.numerator) / Decimal(value.denominator), ".12g")
    return rendered, True


def _normalise_unit(value: str) -> str:
    cooked = value.casefold().strip()
    cooked = cooked.replace("²", "^2").replace("³", "^3").replace("·", "")
    cooked = re.sub(r"\s+", " ", cooked)
    cooked = cooked.replace("metres", "meters").replace("metre", "meter")
    cooked = cooked.replace("kilometres", "kilometers").replace("kilometre", "kilometer")
    cooked = cooked.replace("centimetres", "centimeters").replace("centimetre", "centimeter")
    cooked = cooked.replace("litres", "liters").replace("litre", "liter")
    cooked = cooked.replace("millilitres", "milliliters").replace("millilitre", "milliliter")
    return cooked


def _unit_conversion(kind: str, raw_unit: str) -> tuple[str, Fraction, Fraction, str]:
    unit = _normalise_unit(raw_unit)
    compact = re.sub(r"\s+", "", unit)
    specs: dict[str, dict[str, tuple[str, Fraction, Fraction, str]]] = {
        "velocity": {
            "m/s": ("m/s", Fraction(1), Fraction(0), "m/s"),
            "meterpersecond": ("m/s", Fraction(1), Fraction(0), "m/s"),
            "meterspersecond": ("m/s", Fraction(1), Fraction(0), "m/s"),
            "km/h": ("km/h", Fraction(5, 18), Fraction(0), "m/s"),
            "kilometerperhour": ("km/h", Fraction(5, 18), Fraction(0), "m/s"),
            "kilometersperhour": ("km/h", Fraction(5, 18), Fraction(0), "m/s"),
            "cm/s": ("cm/s", Fraction(1, 100), Fraction(0), "m/s"),
            "centimeterpersecond": ("cm/s", Fraction(1, 100), Fraction(0), "m/s"),
            "centimeterspersecond": ("cm/s", Fraction(1, 100), Fraction(0), "m/s"),
            "mph": ("mph", Fraction(1397, 3125), Fraction(0), "m/s"),
            "mileperhour": ("mph", Fraction(1397, 3125), Fraction(0), "m/s"),
            "milesperhour": ("mph", Fraction(1397, 3125), Fraction(0), "m/s"),
        },
        "acceleration": {
            "m/s^2": ("m/s^2", Fraction(1), Fraction(0), "m/s^2"),
            "meterpersecondsquared": ("m/s^2", Fraction(1), Fraction(0), "m/s^2"),
            "meterspersecondsquared": ("m/s^2", Fraction(1), Fraction(0), "m/s^2"),
            "km/h^2": ("km/h^2", Fraction(1, 12_960), Fraction(0), "m/s^2"),
            "kilometerperhoursquared": ("km/h^2", Fraction(1, 12_960), Fraction(0), "m/s^2"),
            "kilometersperhoursquared": ("km/h^2", Fraction(1, 12_960), Fraction(0), "m/s^2"),
            "cm/s^2": ("cm/s^2", Fraction(1, 100), Fraction(0), "m/s^2"),
            "centimeterpersecondsquared": ("cm/s^2", Fraction(1, 100), Fraction(0), "m/s^2"),
            "centimeterspersecondsquared": ("cm/s^2", Fraction(1, 100), Fraction(0), "m/s^2"),
            "ft/s^2": ("ft/s^2", Fraction(381, 1250), Fraction(0), "m/s^2"),
            "feetpersecondsquared": ("ft/s^2", Fraction(381, 1250), Fraction(0), "m/s^2"),
        },
        "time": {
            "s": ("s", Fraction(1), Fraction(0), "s"),
            "sec": ("s", Fraction(1), Fraction(0), "s"),
            "secs": ("s", Fraction(1), Fraction(0), "s"),
            "second": ("s", Fraction(1), Fraction(0), "s"),
            "seconds": ("s", Fraction(1), Fraction(0), "s"),
            "min": ("min", Fraction(60), Fraction(0), "s"),
            "mins": ("min", Fraction(60), Fraction(0), "s"),
            "minute": ("min", Fraction(60), Fraction(0), "s"),
            "minutes": ("min", Fraction(60), Fraction(0), "s"),
            "h": ("h", Fraction(3600), Fraction(0), "s"),
            "hr": ("h", Fraction(3600), Fraction(0), "s"),
            "hrs": ("h", Fraction(3600), Fraction(0), "s"),
            "hour": ("h", Fraction(3600), Fraction(0), "s"),
            "hours": ("h", Fraction(3600), Fraction(0), "s"),
        },
        "pressure": {
            "pa": ("Pa", Fraction(1), Fraction(0), "Pa"),
            "pascal": ("Pa", Fraction(1), Fraction(0), "Pa"),
            "pascals": ("Pa", Fraction(1), Fraction(0), "Pa"),
            "kpa": ("kPa", Fraction(1000), Fraction(0), "Pa"),
            "kilopascal": ("kPa", Fraction(1000), Fraction(0), "Pa"),
            "kilopascals": ("kPa", Fraction(1000), Fraction(0), "Pa"),
            "mpa": ("MPa", Fraction(1_000_000), Fraction(0), "Pa"),
            "megapascal": ("MPa", Fraction(1_000_000), Fraction(0), "Pa"),
            "megapascals": ("MPa", Fraction(1_000_000), Fraction(0), "Pa"),
            "bar": ("bar", Fraction(100_000), Fraction(0), "Pa"),
            "bars": ("bar", Fraction(100_000), Fraction(0), "Pa"),
            "atm": ("atm", Fraction(101_325), Fraction(0), "Pa"),
            "atmosphere": ("atm", Fraction(101_325), Fraction(0), "Pa"),
            "atmospheres": ("atm", Fraction(101_325), Fraction(0), "Pa"),
        },
        "volume": {
            "m^3": ("m^3", Fraction(1), Fraction(0), "m^3"),
            "cubicmeter": ("m^3", Fraction(1), Fraction(0), "m^3"),
            "cubicmeters": ("m^3", Fraction(1), Fraction(0), "m^3"),
            "l": ("L", Fraction(1, 1000), Fraction(0), "m^3"),
            "liter": ("L", Fraction(1, 1000), Fraction(0), "m^3"),
            "liters": ("L", Fraction(1, 1000), Fraction(0), "m^3"),
            "ml": ("mL", Fraction(1, 1_000_000), Fraction(0), "m^3"),
            "milliliter": ("mL", Fraction(1, 1_000_000), Fraction(0), "m^3"),
            "milliliters": ("mL", Fraction(1, 1_000_000), Fraction(0), "m^3"),
            "cm^3": ("cm^3", Fraction(1, 1_000_000), Fraction(0), "m^3"),
            "cubiccentimeter": ("cm^3", Fraction(1, 1_000_000), Fraction(0), "m^3"),
            "cubiccentimeters": ("cm^3", Fraction(1, 1_000_000), Fraction(0), "m^3"),
        },
        "temperature": {
            "k": ("K", Fraction(1), Fraction(0), "K"),
            "kelvin": ("K", Fraction(1), Fraction(0), "K"),
            "kelvins": ("K", Fraction(1), Fraction(0), "K"),
            "degc": ("degC", Fraction(1), Fraction(5463, 20), "K"),
            "°c": ("degC", Fraction(1), Fraction(5463, 20), "K"),
            "celsius": ("degC", Fraction(1), Fraction(5463, 20), "K"),
            "degreecelsius": ("degC", Fraction(1), Fraction(5463, 20), "K"),
            "degreescelsius": ("degC", Fraction(1), Fraction(5463, 20), "K"),
        },
        "amount": {
            "mol": ("mol", Fraction(1), Fraction(0), "mol"),
            "mole": ("mol", Fraction(1), Fraction(0), "mol"),
            "moles": ("mol", Fraction(1), Fraction(0), "mol"),
            "mmol": ("mmol", Fraction(1, 1000), Fraction(0), "mol"),
            "millimole": ("mmol", Fraction(1, 1000), Fraction(0), "mol"),
            "millimoles": ("mmol", Fraction(1, 1000), Fraction(0), "mol"),
            "kmol": ("kmol", Fraction(1000), Fraction(0), "mol"),
            "kilomole": ("kmol", Fraction(1000), Fraction(0), "mol"),
            "kilomoles": ("kmol", Fraction(1000), Fraction(0), "mol"),
        },
    }
    spec = specs.get(kind, {}).get(compact)
    if spec is None:
        raise ValueError("Science-plan unit is unsupported.")
    return spec


def _source_span(text: str, start: int, end: int) -> dict[str, Any]:
    if not 0 <= start < end <= len(text):
        raise ValueError("Science-plan source span is invalid.")
    return {"start": start, "end": end, "sha256": _sha256_text(text[start:end])}


def _quantity_from_match(
    text: str,
    symbol: str,
    dimension: str,
    match: re.Match[str],
) -> dict[str, Any]:
    source_value = _parse_fraction(match.group("value"))
    source_unit, factor, offset, si_unit = _unit_conversion(dimension, match.group("unit"))
    si_value = _guard_fraction(source_value * factor + offset)
    return {
        "symbol": symbol,
        "source_value": _fraction_text(source_value),
        "source_unit": source_unit,
        "si_value": _fraction_text(si_value),
        "si_unit": si_unit,
        "dimension": dimension,
        "span": _source_span(text, match.start(), match.end()),
    }


def _rest_quantity(text: str, match: re.Match[str]) -> dict[str, Any]:
    return {
        "symbol": "u",
        "source_value": "0",
        "source_unit": "rest",
        "si_value": "0",
        "si_unit": "m/s",
        "dimension": "velocity",
        "span": _source_span(text, match.start(), match.end()),
    }


def _one_match(pattern: re.Pattern[str], text: str) -> Optional[re.Match[str]]:
    matches = list(pattern.finditer(text))
    return matches[0] if len(matches) == 1 else None


def _target_for_scenario(text: str, scenario: str) -> tuple[Optional[str], str]:
    matches: list[tuple[str, re.Match[str]]] = []
    for (candidate_scenario, target), pattern in _TARGET_PATTERNS.items():
        if candidate_scenario != scenario:
            continue
        matches.extend((target, match) for match in pattern.finditer(text))
    if len(matches) > 1:
        return None, "multiple_targets"
    if not matches:
        return None, "unsupported_target"
    request_cues = list(_REQUEST_CUE_RE.finditer(text))
    if len(request_cues) != 1:
        return None, "multiple_targets"
    question_text = text[request_cues[0].start() :]
    if len(list(_QUESTION_TARGET_PATTERNS[scenario].finditer(question_text))) != 1:
        return None, "multiple_targets"
    return matches[0][0], ""


def _spans_are_disjoint(spans: Sequence[tuple[int, int]]) -> bool:
    ordered = sorted(spans)
    return all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:]))


def _all_numbers_consumed(text: str, spans: Sequence[tuple[int, int]]) -> bool:
    for match in _NUMBER_RE.finditer(text):
        if not any(start <= match.start() and match.end() <= end for start, end in spans):
            return False
    return True


def _all_words_consumed(
    text: str,
    spans: Sequence[tuple[int, int]],
    allowed_words: frozenset[str],
) -> bool:
    remaining = list(text)
    for start, end in spans:
        remaining[start:end] = " " * (end - start)
    words = re.findall(r"[A-Za-z]+", "".join(remaining).casefold())
    return all(word in allowed_words for word in words)


def _plan_sha256(plan_without_digest: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(plan_without_digest))


def _make_plan(
    text: str,
    formula: Mapping[str, Any],
    quantities: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "schema_version": SCIENCE_PLAN_SCHEMA_VERSION,
        "registry_version": SCIENCE_FORMULA_REGISTRY_VERSION,
        "registry_sha256": SCIENCE_FORMULA_REGISTRY_SHA256,
        "scenario": formula["scenario"],
        "target": formula["target"],
        "assumptions": list(formula["required_assumptions"]),
        "quantities": [dict(quantity) for quantity in quantities],
        "steps": [
            {
                "formula_id": formula["formula_id"],
                "solve_for": formula["target_symbol"],
                "inputs": list(formula["inputs"]),
            }
        ],
        "query_sha256": _sha256_text(text),
    }
    plan["plan_sha256"] = _plan_sha256(plan)
    return plan


def _clean_query(query: Any) -> tuple[Optional[str], str]:
    if not isinstance(query, str):
        return None, "invalid_query_text"
    if len(query) > MAX_QUERY_CHARS:
        return None, "query_too_long"
    text = query.strip()
    if not text:
        return None, "empty_query"
    if len(text) > MAX_QUERY_CHARS:
        return None, "query_too_long"
    if _CONTROL_CHAR_RE.search(text) is not None or "`" in text or '"' in text or "'" in text:
        return None, "invalid_query_text"
    if _ALLOWED_QUERY_CHARS_RE.fullmatch(text) is None:
        return None, "invalid_query_text"
    if text.count("?") > 1:
        return None, "multiple_targets"
    if _HIGH_STAKES_OR_OPEN_WORLD_RE.search(text) is not None:
        return None, "high_stakes_or_open_world"
    if _PROMPT_CONTROL_RE.search(text) is not None:
        return None, "prompt_control_or_mixed_request"
    return text, ""


def _parse_kinematics(text: str, target: str) -> tuple[Optional[dict[str, Any]], str]:
    formula = _FORMULA_BY_TARGET[("constant_acceleration", target)]
    velocity_matches = list(_INITIAL_VELOCITY_RE.finditer(text))
    rest_matches = list(_REST_RE.finditer(text))
    acceleration_matches = list(_ACCELERATION_RE.finditer(text))
    time_matches = list(_TIME_RE.finditer(text))

    if len(velocity_matches) + len(rest_matches) != 1:
        return None, "missing_or_ambiguous_quantity"
    if len(acceleration_matches) != 1 or len(time_matches) != 1:
        return None, "missing_or_ambiguous_quantity"

    try:
        initial = (
            _quantity_from_match(text, "u", "velocity", velocity_matches[0])
            if velocity_matches
            else _rest_quantity(text, rest_matches[0])
        )
        acceleration = _quantity_from_match(
            text, "a", "acceleration", acceleration_matches[0]
        )
        elapsed = _quantity_from_match(text, "t", "time", time_matches[0])
    except (_ScienceLimit, ValueError, InvalidOperation):
        return None, "missing_or_ambiguous_quantity"

    quantities = (initial, acceleration, elapsed)
    spans = tuple((item["span"]["start"], item["span"]["end"]) for item in quantities)
    if not _spans_are_disjoint(spans):
        return None, "missing_or_ambiguous_quantity"
    if not _all_numbers_consumed(text, spans):
        return None, "mixed_or_unconsumed_request"
    if not _all_words_consumed(text, spans, _KINEMATICS_ALLOWED_WORDS):
        return None, "mixed_or_unconsumed_request"

    if _parse_fraction(elapsed["si_value"]) <= 0:
        return None, "invalid_quantity_domain"
    return _make_plan(text, formula, quantities), ""


_IDEAL_INPUTS = {
    "P": (_PRESSURE_RE, "pressure"),
    "V": (_VOLUME_RE, "volume"),
    "T": (_TEMPERATURE_RE, "temperature"),
    "n": (_AMOUNT_RE, "amount"),
}


def _parse_ideal_gas(text: str, target: str) -> tuple[Optional[dict[str, Any]], str]:
    formula = _FORMULA_BY_TARGET[("ideal_gas", target)]
    target_symbol = str(formula["target_symbol"])
    quantities_by_symbol: dict[str, dict[str, Any]] = {}

    for symbol, (pattern, dimension) in _IDEAL_INPUTS.items():
        matches = list(pattern.finditer(text))
        expected = symbol in formula["inputs"]
        if (expected and len(matches) != 1) or (not expected and matches):
            return None, "missing_or_ambiguous_quantity"
        if matches:
            try:
                quantities_by_symbol[symbol] = _quantity_from_match(
                    text, symbol, dimension, matches[0]
                )
            except (_ScienceLimit, ValueError, InvalidOperation):
                return None, "missing_or_ambiguous_quantity"

    if target_symbol in quantities_by_symbol:
        return None, "missing_or_ambiguous_quantity"
    quantities = tuple(quantities_by_symbol[str(symbol)] for symbol in formula["inputs"])
    spans = tuple((item["span"]["start"], item["span"]["end"]) for item in quantities)
    if not _spans_are_disjoint(spans):
        return None, "missing_or_ambiguous_quantity"
    if not _all_numbers_consumed(text, spans):
        return None, "mixed_or_unconsumed_request"
    if not _all_words_consumed(text, spans, _IDEAL_GAS_ALLOWED_WORDS):
        return None, "mixed_or_unconsumed_request"
    if any(_parse_fraction(item["si_value"]) <= 0 for item in quantities):
        return None, "invalid_quantity_domain"
    return _make_plan(text, formula, quantities), ""


def _parse_with_reason(query: Any) -> tuple[Optional[dict[str, Any]], str]:
    text, reason = _clean_query(query)
    if text is None:
        return None, reason

    constant_matches = list(_CONSTANT_ASSUMPTION_RE.finditer(text))
    ideal_matches = list(_IDEAL_GAS_ASSUMPTION_RE.finditer(text))
    if constant_matches and ideal_matches:
        return None, "ambiguous_scenario"
    if len(constant_matches) > 1 or len(ideal_matches) > 1:
        return None, "ambiguous_scenario"
    if not constant_matches and not ideal_matches:
        return None, "missing_explicit_assumption"

    scenario = "constant_acceleration" if constant_matches else "ideal_gas"
    target, target_reason = _target_for_scenario(text, scenario)
    if target is None:
        return None, target_reason
    if scenario == "constant_acceleration":
        return _parse_kinematics(text, target)
    return _parse_ideal_gas(text, target)


def parse_science_scenario(query: Any) -> Optional[dict[str, Any]]:
    """Parse one fully consumed supported request into an integrity-bound plan.

    Unsupported, ambiguous, open-world, high-stakes, or mixed requests return
    ``None``.  The returned mapping is JSON-safe and contains no raw query or
    raw source substrings.
    """

    plan, _reason = _parse_with_reason(query)
    return plan


def _validate_quantity(
    value: Any,
    *,
    expected_symbol: str,
    expected_dimension: str,
) -> tuple[dict[str, Any], Fraction]:
    if not isinstance(value, Mapping) or set(value) != {
        "symbol",
        "source_value",
        "source_unit",
        "si_value",
        "si_unit",
        "dimension",
        "span",
    }:
        raise ValueError("Science-plan quantity has unexpected fields.")
    if value.get("symbol") != expected_symbol or value.get("dimension") != expected_dimension:
        raise ValueError("Science-plan quantity identity is invalid.")

    source_value_text = value.get("source_value")
    si_value_text = value.get("si_value")
    if not isinstance(source_value_text, str) or not isinstance(si_value_text, str):
        raise ValueError("Science-plan quantity values must be text.")
    source_value = _parse_fraction(source_value_text)
    si_value = _parse_fraction(si_value_text)
    if _fraction_text(source_value) != source_value_text or _fraction_text(si_value) != si_value_text:
        raise ValueError("Science-plan quantity values are not canonical.")

    source_unit = value.get("source_unit")
    if source_unit == "rest":
        if expected_symbol != "u" or source_value != 0 or si_value != 0 or value.get("si_unit") != "m/s":
            raise ValueError("Science-plan rest binding is invalid.")
    else:
        if not isinstance(source_unit, str):
            raise ValueError("Science-plan source unit is invalid.")
        canonical_unit, factor, offset, si_unit = _unit_conversion(
            expected_dimension, source_unit
        )
        if canonical_unit != source_unit or value.get("si_unit") != si_unit:
            raise ValueError("Science-plan canonical unit binding is invalid.")
        if _guard_fraction(source_value * factor + offset) != si_value:
            raise ValueError("Science-plan SI conversion is invalid.")

    span = value.get("span")
    if not isinstance(span, Mapping) or set(span) != {"start", "end", "sha256"}:
        raise ValueError("Science-plan source span is invalid.")
    start = span.get("start")
    end = span.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= MAX_QUERY_CHARS
        or not _is_sha256(span.get("sha256"))
    ):
        raise ValueError("Science-plan source span metadata is invalid.")
    return dict(value), si_value


def _validate_plan(plan: Any) -> tuple[dict[str, Any], Mapping[str, Any], dict[str, Fraction]]:
    if not isinstance(plan, Mapping) or set(plan) != {
        "schema_version",
        "registry_version",
        "registry_sha256",
        "scenario",
        "target",
        "assumptions",
        "quantities",
        "steps",
        "query_sha256",
        "plan_sha256",
    }:
        raise ValueError("Science plan has unexpected fields.")
    if plan.get("schema_version") != SCIENCE_PLAN_SCHEMA_VERSION:
        raise ValueError("Science-plan schema is unsupported.")
    if plan.get("registry_version") != SCIENCE_FORMULA_REGISTRY_VERSION:
        raise ValueError("Science-plan registry version is unsupported.")
    if plan.get("registry_sha256") != SCIENCE_FORMULA_REGISTRY_SHA256:
        raise ValueError("Science-plan registry digest is invalid.")
    if not _is_sha256(plan.get("query_sha256")) or not _is_sha256(plan.get("plan_sha256")):
        raise ValueError("Science-plan digest is invalid.")

    without_digest = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _plan_sha256(without_digest) != plan.get("plan_sha256"):
        raise ValueError("Science-plan content digest is invalid.")

    scenario = plan.get("scenario")
    target = plan.get("target")
    formula = _FORMULA_BY_TARGET.get((scenario, target))
    if formula is None:
        raise ValueError("Science-plan scenario or target is unsupported.")

    assumptions = plan.get("assumptions")
    if (
        isinstance(assumptions, (str, bytes))
        or not isinstance(assumptions, Sequence)
        or list(assumptions) != list(formula["required_assumptions"])
    ):
        raise ValueError("Science-plan assumptions are invalid.")

    steps = plan.get("steps")
    if (
        isinstance(steps, (str, bytes))
        or not isinstance(steps, Sequence)
        or not 1 <= len(steps) <= MAX_PLAN_STEPS
        or len(steps) != 1
    ):
        raise ValueError("Science-plan steps are invalid.")
    step = steps[0]
    if not isinstance(step, Mapping) or set(step) != {"formula_id", "solve_for", "inputs"}:
        raise ValueError("Science-plan step has unexpected fields.")
    if (
        step.get("formula_id") != formula["formula_id"]
        or step.get("solve_for") != formula["target_symbol"]
        or step.get("inputs") != list(formula["inputs"])
    ):
        raise ValueError("Science-plan step does not match the registry.")

    quantities = plan.get("quantities")
    if (
        isinstance(quantities, (str, bytes))
        or not isinstance(quantities, Sequence)
        or not 1 <= len(quantities) <= MAX_QUANTITIES
        or len(quantities) != len(formula["inputs"])
    ):
        raise ValueError("Science-plan quantities are invalid.")

    normalized_quantities: list[dict[str, Any]] = []
    values: dict[str, Fraction] = {}
    spans: list[tuple[int, int]] = []
    for raw_quantity, symbol, dimension in zip(
        quantities, formula["inputs"], formula["input_dimensions"]
    ):
        quantity, si_value = _validate_quantity(
            raw_quantity,
            expected_symbol=str(symbol),
            expected_dimension=str(dimension),
        )
        normalized_quantities.append(quantity)
        values[str(symbol)] = si_value
        spans.append((quantity["span"]["start"], quantity["span"]["end"]))
    if not _spans_are_disjoint(spans):
        raise ValueError("Science-plan source spans overlap.")

    normalized = dict(plan)
    normalized["assumptions"] = list(assumptions)
    normalized["quantities"] = normalized_quantities
    normalized["steps"] = [dict(step)]
    return normalized, formula, values


def _dimension_add(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(a + b for a, b in zip(left, right))


def _dimension_scale(value: tuple[int, ...], factor: int) -> tuple[int, ...]:
    return tuple(item * factor for item in value)


def _dimension_check(formula: Mapping[str, Any]) -> bool:
    velocity = _DIMENSION_VECTORS["velocity"]
    acceleration = _DIMENSION_VECTORS["acceleration"]
    elapsed = _DIMENSION_VECTORS["time"]
    length = _DIMENSION_VECTORS["length"]
    if formula["formula_id"] == "constant_acceleration.final_velocity":
        return _dimension_add(acceleration, elapsed) == velocity
    if formula["formula_id"] == "constant_acceleration.displacement":
        return (
            _dimension_add(velocity, elapsed) == length
            and _dimension_add(acceleration, _dimension_scale(elapsed, 2)) == length
        )
    left = _dimension_add(_DIMENSION_VECTORS["pressure"], _DIMENSION_VECTORS["volume"])
    right = _dimension_add(
        _DIMENSION_VECTORS["amount"],
        _dimension_add(
            _DIMENSION_VECTORS["gas_constant"],
            _DIMENSION_VECTORS["temperature"],
        ),
    )
    return left == right


def _compute(formula: Mapping[str, Any], values: Mapping[str, Fraction]) -> Fraction:
    formula_id = formula["formula_id"]
    if formula_id == "constant_acceleration.final_velocity":
        return _guard_fraction(values["u"] + values["a"] * values["t"])
    if formula_id == "constant_acceleration.displacement":
        return _guard_fraction(
            values["u"] * values["t"]
            + values["a"] * values["t"] * values["t"] / 2
        )
    if formula_id == "ideal_gas.pressure":
        return _guard_fraction(values["n"] * _MOLAR_GAS_CONSTANT * values["T"] / values["V"])
    if formula_id == "ideal_gas.volume":
        return _guard_fraction(values["n"] * _MOLAR_GAS_CONSTANT * values["T"] / values["P"])
    if formula_id == "ideal_gas.temperature":
        return _guard_fraction(values["P"] * values["V"] / (values["n"] * _MOLAR_GAS_CONSTANT))
    if formula_id == "ideal_gas.amount":
        return _guard_fraction(values["P"] * values["V"] / (_MOLAR_GAS_CONSTANT * values["T"]))
    raise ValueError("Science-plan formula is unsupported.")


def _domain_check(
    formula: Mapping[str, Any],
    values: Mapping[str, Fraction],
    answer: Fraction,
) -> bool:
    if formula["scenario"] == "constant_acceleration":
        return values["t"] > 0
    completed = dict(values)
    completed[str(formula["target_symbol"])] = answer
    return all(completed[symbol] > 0 for symbol in ("P", "V", "n", "T"))


def _substitution_check(
    formula: Mapping[str, Any],
    values: Mapping[str, Fraction],
    answer: Fraction,
) -> bool:
    formula_id = formula["formula_id"]
    if formula_id == "constant_acceleration.final_velocity":
        return answer - values["u"] == values["a"] * values["t"]
    if formula_id == "constant_acceleration.displacement":
        return 2 * (answer - values["u"] * values["t"]) == values["a"] * values["t"] ** 2
    completed = dict(values)
    completed[str(formula["target_symbol"])] = answer
    return completed["P"] * completed["V"] == completed["n"] * _MOLAR_GAS_CONSTANT * completed["T"]


def _receipt(
    plan: Mapping[str, Any],
    formula: Mapping[str, Any],
    checks: Mapping[str, bool],
    *,
    decision: str,
) -> dict[str, Any]:
    input_spans = []
    for quantity in plan.get("quantities", []):
        if not isinstance(quantity, Mapping):
            continue
        span = quantity.get("span")
        if not isinstance(span, Mapping):
            continue
        input_spans.append(
            {
                "symbol": quantity.get("symbol"),
                "start": span.get("start"),
                "end": span.get("end"),
                "sha256": span.get("sha256"),
            }
        )
    return {
        "schema_version": SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION,
        "decision": decision,
        "scenario": formula.get("scenario", ""),
        "target": formula.get("target", ""),
        "formula_ids": [formula.get("formula_id", "")],
        "registry_version": SCIENCE_FORMULA_REGISTRY_VERSION,
        "registry_sha256": SCIENCE_FORMULA_REGISTRY_SHA256,
        "query_sha256": plan.get("query_sha256", ""),
        "plan_sha256": plan.get("plan_sha256", ""),
        "input_spans": input_spans,
        "checks": dict(checks),
        "epistemics": {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        },
        "diagnostic_only": True,
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_tools": False,
            "controls_permissions": False,
            "controls_safety": False,
        },
    }


def _empty_result(reason: str, *, attempted: bool = False, query_sha256: str = "") -> dict[str, Any]:
    safe_reason = reason if reason in _REASON_ALLOWLIST else "invalid_plan"
    return {
        "schema_version": SCIENCE_PLAN_SCHEMA_VERSION,
        "engine_version": SCIENCE_PLAN_ENGINE_VERSION,
        "registry_version": SCIENCE_FORMULA_REGISTRY_VERSION,
        "registry_sha256": SCIENCE_FORMULA_REGISTRY_SHA256,
        "attempted": bool(attempted),
        "solved": False,
        "override_allowed": False,
        "scenario": "",
        "target": "",
        "formula_id": "",
        "reason": safe_reason,
        "answer": {
            "exact": "",
            "display": "",
            "approximation": "",
            "approximate": False,
            "unit": "",
        },
        "steps": [],
        "assumptions": [],
        "verification": {
            "checked": False,
            "passed": False,
            "method": "none",
            "independent": False,
            "checks": {
                "registry_integrity": False,
                "plan_integrity": False,
                "input_bindings": False,
                "dimensions": False,
                "domain": False,
                "substitution": False,
            },
        },
        "epistemics": {
            "model_conditional": False,
            "assumptions_explicit": False,
            "calibration_claimed": False,
        },
        "budget": {
            "quantities": 0,
            "quantity_limit": MAX_QUANTITIES,
            "steps": 0,
            "step_limit": MAX_PLAN_STEPS,
        },
        "receipt": {
            "schema_version": SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION,
            "decision": "abstained",
            "scenario": "",
            "target": "",
            "formula_ids": [],
            "registry_version": SCIENCE_FORMULA_REGISTRY_VERSION,
            "registry_sha256": SCIENCE_FORMULA_REGISTRY_SHA256,
            "query_sha256": query_sha256 if _is_sha256(query_sha256) else "",
            "plan_sha256": "",
            "input_spans": [],
            "checks": {},
            "epistemics": {
                "model_conditional": False,
                "assumptions_explicit": False,
                "calibration_claimed": False,
            },
            "diagnostic_only": True,
            "authority": {
                "controls_compute": False,
                "controls_routes": False,
                "controls_interaction_strategy": False,
                "controls_tools": False,
                "controls_permissions": False,
                "controls_safety": False,
            },
        },
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_tools": False,
            "controls_permissions": False,
            "controls_safety": False,
        },
    }


def execute_science_plan(plan: Any) -> dict[str, Any]:
    """Validate and execute one non-executable science-plan mapping.

    Invalid or tampered mappings fail closed as JSON-safe ``invalid_plan``
    results.  No caller-controlled expression is ever evaluated.
    """

    try:
        normalized, formula, values = _validate_plan(plan)
        answer = _compute(formula, values)
        checks = {
            "registry_integrity": normalized["registry_sha256"] == SCIENCE_FORMULA_REGISTRY_SHA256,
            "plan_integrity": _plan_sha256(
                {key: value for key, value in normalized.items() if key != "plan_sha256"}
            )
            == normalized["plan_sha256"],
            "input_bindings": len(normalized["quantities"]) == len(formula["inputs"]),
            "dimensions": _dimension_check(formula),
            "domain": _domain_check(formula, values, answer),
            "substitution": _substitution_check(formula, values, answer),
        }
    except (_ScienceLimit, ArithmeticError, InvalidOperation, KeyError, TypeError, ValueError):
        return _empty_result("invalid_plan", attempted=True)

    passed = all(checks.values())
    if not passed:
        return _empty_result("verification_failed", attempted=True, query_sha256=normalized["query_sha256"])

    display, approximate = _display_number(answer)
    result = {
        "schema_version": SCIENCE_PLAN_SCHEMA_VERSION,
        "engine_version": SCIENCE_PLAN_ENGINE_VERSION,
        "registry_version": SCIENCE_FORMULA_REGISTRY_VERSION,
        "registry_sha256": SCIENCE_FORMULA_REGISTRY_SHA256,
        "attempted": True,
        "solved": True,
        "override_allowed": True,
        "scenario": formula["scenario"],
        "target": formula["target"],
        "formula_id": formula["formula_id"],
        "reason": "verified_science_plan",
        "answer": {
            "exact": _fraction_text(answer),
            "display": display,
            "approximation": display if approximate else "",
            "approximate": approximate,
            "unit": formula["target_unit"],
        },
        "steps": [
            f"Apply registry formula {formula['formula_id']} in canonical SI units.",
            "Check dimensions, quantity domains, and substitution into the defining equation.",
        ],
        "assumptions": list(formula["required_assumptions"]),
        "verification": {
            "checked": True,
            "passed": True,
            "method": "registry_dimension_domain_and_substitution",
            "independent": False,
            "checks": checks,
        },
        "epistemics": {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        },
        "budget": {
            "quantities": len(normalized["quantities"]),
            "quantity_limit": MAX_QUANTITIES,
            "steps": len(normalized["steps"]),
            "step_limit": MAX_PLAN_STEPS,
        },
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_tools": False,
            "controls_permissions": False,
            "controls_safety": False,
        },
    }
    result["receipt"] = _receipt(normalized, formula, checks, decision="verified")
    return result


def solve_science_scenario(query: Any) -> dict[str, Any]:
    """Parse, execute, and verify one supported closed-world science request."""

    plan, reason = _parse_with_reason(query)
    if plan is None:
        query_digest = _sha256_text(query.strip()) if isinstance(query, str) and query.strip() else ""
        attempted = reason not in {"empty_query", "query_too_long", "invalid_query_text"}
        return _empty_result(reason, attempted=attempted, query_sha256=query_digest)
    return execute_science_plan(plan)


def science_plan_diagnostics(result: Any) -> dict[str, Any]:
    """Return bounded, allowlisted diagnostics with no prompt or source text."""

    if not isinstance(result, Mapping):
        result = {}
    verification = result.get("verification")
    if not isinstance(verification, Mapping):
        verification = {}
    epistemics = result.get("epistemics")
    if not isinstance(epistemics, Mapping):
        epistemics = {}
    budget = result.get("budget")
    if not isinstance(budget, Mapping):
        budget = {}

    scenario = result.get("scenario")
    if scenario not in {"constant_acceleration", "ideal_gas"}:
        scenario = ""
    target = result.get("target")
    if (scenario, target) not in _FORMULA_BY_TARGET:
        target = ""
    formula_id = result.get("formula_id")
    if formula_id not in _FORMULA_BY_ID:
        formula_id = ""
    reason = result.get("reason")
    if reason not in _REASON_ALLOWLIST:
        reason = "invalid_plan"

    def bounded_count(value: Any, limit: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            return 0
        return min(limit, max(0, value))

    return {
        "schema_version": SCIENCE_PLAN_SCHEMA_VERSION,
        "engine_version": SCIENCE_PLAN_ENGINE_VERSION,
        "registry_version": SCIENCE_FORMULA_REGISTRY_VERSION,
        "registry_sha256": SCIENCE_FORMULA_REGISTRY_SHA256,
        "attempted": bool(result.get("attempted", False)),
        "solved": bool(result.get("solved", False)),
        "override_allowed": bool(result.get("override_allowed", False)),
        "scenario": scenario,
        "target": target,
        "formula_id": formula_id,
        "reason": reason,
        "verification_passed": bool(verification.get("passed", False)),
        "verification_independent": bool(verification.get("independent", False)),
        "model_conditional": bool(epistemics.get("model_conditional", False)),
        "assumptions_explicit": bool(epistemics.get("assumptions_explicit", False)),
        "calibration_claimed": False,
        "quantities": bounded_count(budget.get("quantities"), MAX_QUANTITIES),
        "steps": bounded_count(budget.get("steps"), MAX_PLAN_STEPS),
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_tools": False,
            "controls_permissions": False,
            "controls_safety": False,
        },
    }


__all__ = [
    "FORMULA_REGISTRY_SHA256",
    "MAX_PLAN_STEPS",
    "MAX_QUANTITIES",
    "SCIENCE_FORMULA_REGISTRY_CANONICAL_JSON",
    "SCIENCE_FORMULA_REGISTRY_SHA256",
    "SCIENCE_FORMULA_REGISTRY_VERSION",
    "SCIENCE_PLAN_ENGINE_VERSION",
    "SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION",
    "SCIENCE_PLAN_SCHEMA_VERSION",
    "execute_science_plan",
    "parse_science_scenario",
    "science_plan_diagnostics",
    "solve_science_scenario",
]
