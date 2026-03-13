from overrides.text_processing import normalize_text


def test_normalize_text_expands_fixed_units() -> None:
    cases = {
        "32f": "degrees fahrenheit",
        "20c": "degrees celsius",
        "5ml": "milliliters",
        "7mm/s": "millimeters per second",
        "2cm/h": "centimeters per hour",
        "$2.50": "two dollars and fifty cents",
    }

    for text, expected in cases.items():
        assert expected in normalize_text(text)
