import pytest

from multi_nlu.annotation import Annotation, is_faithful

TEXT = "book a table in ames and play jazz"


@pytest.mark.parametrize(
    "copied,faithful",
    [
        (TEXT, True),
        (TEXT.replace(" in ", " in  "), True),  # whitespace only
        (TEXT.replace("ames", "boston"), False),
    ],
)
def test_faithfulness_ignores_whitespace_but_not_words(copied, faithful):
    assert is_faithful(Annotation(copied), TEXT) is faithful
