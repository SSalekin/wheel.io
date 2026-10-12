from app.services.rules import (
    FACE_STEER,
    TEXT_STEER,
    classify_message,
    duplicates_directive,
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


def test_parse_keeps_entry_containing_instruction_word():
    # Regression: "Random Guy" was dropped as an instruction word and the
    # message lost an entry (or failed classification entirely).
    assert parse_items("wheel: Random Guy, Ana, Bob") == ["Random Guy", "Ana", "Bob"]


def test_parse_strips_unambiguous_instruction_prefix():
    # A connective (with/for/from/of/these) proves the leading run is
    # instruction language fused to the first item, not content.
    assert parse_items("make me a wheel with Ana, Bob, Chi") == ["Ana", "Bob", "Chi"]


def test_parse_keeps_glued_verb_prefix():
    # Pinned tradeoff: "wheel Ana" (single glued verb, no connective) is
    # ambiguous, so the segment is kept whole — a visible wrong entry
    # beats an invisible missing one.
    assert parse_items("wheel Ana, Bob, Chi") == ["wheel Ana", "Bob", "Chi"]


def test_parse_keeps_entry_after_connective_verb():
    # Regression (audit #1): the instruction-prefix strip used to keep
    # consuming words after the connective, eating the entry's own verb:
    # "make a wheel with Pick Me, Ana" lost "Pick Me" entirely.
    assert parse_items("make a wheel with Pick Me, Ana") == ["Pick Me", "Ana"]


def test_parse_keeps_entry_starting_with_instruction_word():
    # Same regression with a mid-phrase entry: "Random Guy" must not
    # collapse to "Guy" just because "random" is an instruction token.
    assert parse_items("make a wheel with Random Guy, Pick Me") == [
        "Random Guy",
        "Pick Me",
    ]


def test_parse_keeps_the_rock_after_connective():
    # "The" is filler and may stay in the run; the entry must not.
    assert parse_items("make a wheel with The Rock, Ana") == ["The Rock", "Ana"]


def test_parse_keeps_time_like_colons_in_data():
    # A colon inside data — a time — is not instruction language; only
    # pre-colon TEXT THAT IS instruction-like gets dropped (audit #7).
    assert parse_items("9:30, Ana, Bob") == ["9:30", "Ana", "Bob"]


def test_parse_keeps_colons_in_later_lines():
    # Mid-message colons belong to the data ("Bob: red team").
    assert parse_items("Ana\nBob: red team\nCarol") == ["Ana", "Bob: red team", "Carol"]


def test_parse_keeps_urls():
    assert parse_items("https://x.com, Ana, Bob") == ["https://x.com", "Ana", "Bob"]


def test_parse_full_width_colon_still_strips_instruction():
    # The full-width ： carries the same instruction meaning as :.
    assert parse_items("wheel： Ana, Bob") == ["Ana", "Bob"]


def test_parse_keeps_instruction_word_only_entry():
    assert parse_items("wheel: Pick Me, Ana") == ["Pick Me", "Ana"]


def test_parse_drops_instruction_only_segment():
    assert parse_items("make a wheel with the names, Ana, Bob") == ["Ana", "Bob"]
    assert parse_items("pick the names from the list, Ana, Bob") == ["Ana", "Bob"]


def test_parse_instruction_only_message_kept_whole():
    assert parse_items("make a wheel") == ["make a wheel"]


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


def test_prose_with_instruction_word_is_not_a_list():
    # An instruction word inside a prose clause is not a create request:
    # the verb must live in the message's lead, before the first list
    # separator — else "hey, take Bob off the wheel" becomes two junk
    # entries instead of escalating (to a Gemini "remove").
    assert classify_message("hey, take Bob off the wheel", 0) is None


def test_create_text_keeps_instruction_word_entry():
    # Regression: the entry "Random Guy" reached the wheel (classify-level).
    intent = classify_message("wheel: Random Guy, Ana, Bob", 0)

    assert intent is not None
    assert intent.action == "create_text"
    assert intent.entries == ["Random Guy", "Ana", "Bob"]


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


def test_add_beats_trailing_spin_or_go():
    # Audit #9: "spin|go" was checked before "add", so "add Bob and go
    # home" spun the wheel instead of adding Bob.
    assert classify_message("add Bob and go home", 0).action == "add"
    # Plain spin/go messages keep their meaning.
    assert classify_message("spin the wheel", 0).action == "spin"
    assert classify_message("go", 0).action == "spin"


# --- duplicates directive ("no duplicates please" / "skip duplicates") ---


def test_classify_duplicates_standalone_on():
    from app.models import Intent

    assert classify_message("no duplicates please", 0) == Intent(
        action="duplicates", enabled=True
    )
    assert classify_message("Skip duplicates", 0) == Intent(
        action="duplicates", enabled=True
    )
    assert classify_message("no repeats", 0) == Intent(
        action="duplicates", enabled=True
    )
    assert classify_message("don't repeat", 0) == Intent(
        action="duplicates", enabled=True
    )
    assert classify_message('  "No duplicates!"  ', 0) == Intent(
        action="duplicates", enabled=True
    )


def test_classify_duplicates_standalone_off():
    from app.models import Intent

    assert classify_message("allow duplicates", 0) == Intent(
        action="duplicates", enabled=False
    )
    assert classify_message("duplicates are fine", 0) == Intent(
        action="duplicates", enabled=False
    )
    assert classify_message("duplicates ok", 0) == Intent(
        action="duplicates", enabled=False
    )


def test_classify_duplicates_never_hijacks_other_messages():
    # A directive phrase riding along with a real instruction leaves the
    # instruction in charge; the chat layer reads the phrase separately.
    assert classify_message("spin, no duplicates", 0).action == "spin"
    assert classify_message("add Ana, no duplicates", 0).action == "add"


def test_classify_duplicates_leaves_remove_duplicates_alone():
    # "remove duplicates" asks to remove an entry NAMED duplicates —
    # never the directive (the ON phrases need no/skip/without).
    assert classify_message("remove duplicates", 0).action == "remove"


def test_parse_lifts_duplicates_directive_out_of_lists():
    # The directive line/segment must never become a wheel entry.
    assert parse_items("Ana, Bob, no duplicates please") == ["Ana", "Bob"]
    assert parse_items("Ana\nBob\nNo duplicates please") == ["Ana", "Bob"]
    assert parse_items("No duplicates please\nAna\nBob") == ["Ana", "Bob"]
    assert parse_items("Ana, Bob, allow duplicates") == ["Ana", "Bob"]


def test_parse_keeps_names_containing_duplicates_word():
    # Only the directive phrases are lifted; an ordinary name that
    # merely contains the word stays an entry.
    assert parse_items("Duplicate Drew, Ana, Bob") == ["Duplicate Drew", "Ana", "Bob"]


def test_create_with_directive_line_builds_list_without_it():
    intent = classify_message("No duplicates please\nAna\nBob", 0)
    assert intent.action == "create_text"
    assert intent.entries == ["Ana", "Bob"]


def test_duplicates_directive_helper():
    assert duplicates_directive("hey, no duplicates please") is True
    assert duplicates_directive("allow duplicates") is False
    assert duplicates_directive("hello there") is None
    assert duplicates_directive("wheel: Ana, Bob") is None
    # Both phrases present: the ON phrase wins.
    assert duplicates_directive("no duplicates — wait, allow duplicates") is True
