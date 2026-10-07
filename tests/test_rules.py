from app.services.rules import (
    FACE_STEER,
    TEXT_STEER,
    classify_message,
    parse_items,
)


def test_parse_multiline_instruction():
    assert parse_items("Make a wheel with these:\nAna\nBob\nChi") == [
        "Ana",
        "Bob",
        "Chi",
    ]


def test_parse_colon_line():
    assert parse_items("wheel: Ana, Bob, Chi") == ["Ana", "Bob", "Chi"]


def test_parse_bullets_and_numbering():
    assert parse_items("1. Ana\n2) Bob\n- Chi\n* Delta") == [
        "Ana",
        "Bob",
        "Chi",
        "Delta",
    ]


def test_parse_single_sentence_not_split():
    assert parse_items("pick the names from the list") == [
        "pick the names from the list"
    ]


def test_create_text():
    intent = classify_message("Make a wheel with these:\nAna\nBob\nChi", 0)

    assert intent is not None
    assert intent.action == "create_text"
    assert intent.entries == ["Ana", "Bob", "Chi"]


def test_create_text_requires_two_items():
    assert classify_message("hello there", 0) is None
    assert classify_message("Ana", 0) is None


def test_comma_list_needs_verb():
    assert classify_message("Ana, Bob", 0) is None

    intent = classify_message("make a wheel: Ana, Bob", 0)

    assert intent is not None
    assert intent.action == "create_text"
    assert intent.entries == ["Ana", "Bob"]


def test_one_image_routes_to_create_image():
    assert classify_message("wheel from this", 1).action == "create_image"
    assert classify_message("", 1).action == "create_image"


def test_explicit_list_wins_over_image():
    assert classify_message("wheel these:\nAna\nBob", 1).action == "create_text"


def test_two_images_route_to_create_images():
    assert classify_message("avatars", 2).action == "create_images"


def test_remove_single_label():
    intent = classify_message("remove Bob", 0)

    assert intent is not None
    assert intent.action == "remove"
    assert intent.labels == ["Bob"]


def test_remove_multiple_labels():
    intent = classify_message("remove Ana, Bob and Chi", 0)

    assert intent is not None
    assert intent.action == "remove"
    assert intent.labels == ["Ana", "Bob", "Chi"]


def test_add_single_entry():
    intent = classify_message("add Sam", 0)

    assert intent is not None
    assert intent.action == "add"
    assert intent.entries == ["Sam"]


def test_add_multiline_entries():
    intent = classify_message("add\nSam\nTia", 0)

    assert intent is not None
    assert intent.action == "add"
    assert intent.entries == ["Sam", "Tia"]


def test_switch_to_column():
    intent = classify_message("switch to ids", 0)

    assert intent is not None
    assert intent.action == "switch_column"
    assert intent.column == "ids"


def test_reset_and_clear():
    assert classify_message("reset", 0).action == "reset"
    assert classify_message("clear", 0).action == "reset"


def test_spin_and_go():
    assert classify_message("spin", 0).action == "spin"
    assert classify_message("go", 0).action == "spin"


def test_spin_the_wheel():
    assert classify_message("spin the wheel", 0).action == "spin"


def test_smalltalk_returns_none():
    assert classify_message("hello!", 0) is None
    assert classify_message("what can you do?", 0) is None


def test_steer_constants():
    assert TEXT_STEER == {
        "name",
        "names",
        "student",
        "students",
        "id",
        "ids",
        "email",
        "emails",
        "roster",
        "list",
        "table",
        "column",
    }
    assert FACE_STEER == {
        "photo",
        "photos",
        "picture",
        "pictures",
        "face",
        "faces",
        "avatar",
        "avatars",
        "selfie",
        "headshot",
    }
