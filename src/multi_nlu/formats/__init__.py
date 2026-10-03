"""The modeling formats: ways of writing an annotation for a generative model.

A structured JSON form is still to come. Register a new Format here and every
command can use it.
"""

from multi_nlu.formats.base import Format, schema_block
from multi_nlu.formats.offset import OffsetFormat
from multi_nlu.formats.xml import XmlFormat

FORMATS: dict[str, Format] = {f.name: f for f in [XmlFormat(), OffsetFormat()]}

DEFAULT = "xml"


def get_format(name: str) -> Format:
    if name not in FORMATS:
        raise ValueError(f"unknown format {name!r}; available: {', '.join(format_names())}")
    return FORMATS[name]


def format_names() -> list[str]:
    return list(FORMATS)


__all__ = [
    "DEFAULT",
    "FORMATS",
    "Format",
    "OffsetFormat",
    "XmlFormat",
    "format_names",
    "get_format",
    "schema_block",
]
