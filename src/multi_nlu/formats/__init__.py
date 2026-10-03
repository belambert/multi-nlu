"""The modeling formats: ways of writing an annotation for a generative model.

Inline XML and structured-JSON dict are implemented; offset annotations are next.
Register a new Format here and every command can use it.
"""

from multi_nlu.formats.base import Format, schema_block
from multi_nlu.formats.dict_format import DictFormat
from multi_nlu.formats.xml import XmlFormat

FORMATS: dict[str, Format] = {f.name: f for f in [XmlFormat(), DictFormat()]}

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
    "DictFormat",
    "Format",
    "XmlFormat",
    "format_names",
    "get_format",
    "schema_block",
]
