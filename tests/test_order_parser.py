"""Tests for deterministic order parser and confirmation parser."""

import pytest
from cade.fsm.menu_context import MenuContextProvider, MenuContext, MenuItem
from cade.fsm.order_parser import DeterministicOrderParser, ConfirmationParser
from cade.brain.schemas import OrderAction, OrderItem


FOOD_ALIASES = {
    "water": ["water", "bottle of water"],
    "coke": ["coke", "cola", "coca cola"],
    "burger": ["burger", "hamburger"],
    "fries": ["fries", "french fries", "chips"],
    "fried_rice": ["fried rice"],
    "coffee": ["coffee", "latte"],
    "pizza": ["pizza"],
    "tea": ["tea"],
    "salad": ["salad"],
    "noodles": ["noodles", "ramen"],
}


@pytest.fixture
def menu_provider():
    return MenuContextProvider(FOOD_ALIASES)


@pytest.fixture
def full_menu(menu_provider):
    return MenuContext(
        candidates=[MenuItem(canonical=k, aliases=v) for k, v in FOOD_ALIASES.items()],
        canonical_names=menu_provider.all_canonical_names,
    )


@pytest.fixture
def order_parser():
    return DeterministicOrderParser()


@pytest.fixture
def confirm_parser():
    return ConfirmationParser()


# ------------------------------------------------------------------
# Order parser
# ------------------------------------------------------------------

class TestDeterministicOrderParser:

    def test_simple_item(self, order_parser, full_menu):
        result = order_parser.parse_order("coke", full_menu)
        assert result.order is not None
        assert result.confidence >= 0.9
        names = [i.name for i in result.order.items]
        assert "coke" in names

    def test_quantity_word(self, order_parser, full_menu):
        result = order_parser.parse_order("two cokes", full_menu)
        assert result.order is not None
        items = {i.name: i.qty for i in result.order.items}
        assert items.get("coke") == 2

    def test_multiple_items(self, order_parser, full_menu):
        result = order_parser.parse_order("two burgers and a water", full_menu)
        assert result.order is not None
        items = {i.name: i.qty for i in result.order.items}
        assert items.get("burger") == 2
        assert items.get("water") == 1

    def test_alias_resolution(self, order_parser, full_menu):
        result = order_parser.parse_order("cola", full_menu)
        assert result.order is not None
        names = [i.name for i in result.order.items]
        assert "coke" in names

    def test_chips_to_fries(self, order_parser, full_menu):
        result = order_parser.parse_order("chips", full_menu)
        assert result.order is not None
        names = [i.name for i in result.order.items]
        assert "fries" in names

    def test_number_prefix(self, order_parser, full_menu):
        result = order_parser.parse_order("3 waters", full_menu)
        assert result.order is not None
        items = {i.name: i.qty for i in result.order.items}
        assert items.get("water") == 3

    def test_off_domain_returns_no_items(self, order_parser, full_menu):
        result = order_parser.parse_order("hello what is your name", full_menu)
        assert result.order is None or len(result.order.items) == 0

    def test_empty_input(self, order_parser, full_menu):
        result = order_parser.parse_order("", full_menu)
        assert result.order is None

    def test_filler_words_high_confidence(self, order_parser, full_menu):
        result = order_parser.parse_order("please can I have a coke", full_menu)
        assert result.order is not None
        assert result.confidence >= 0.7

    def test_a_burger(self, order_parser, full_menu):
        result = order_parser.parse_order("a burger", full_menu)
        assert result.order is not None
        items = {i.name: i.qty for i in result.order.items}
        assert items.get("burger") == 1


# ------------------------------------------------------------------
# Confirmation parser
# ------------------------------------------------------------------

class TestConfirmationParser:

    def test_yes_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("yes", full_menu)
        assert result.result == "correct"
        assert result.confidence >= 0.9

    def test_yeah_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("yeah that is correct", full_menu)
        assert result.result == "correct"

    def test_sure_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("sure", full_menu)
        assert result.result == "correct"

    def test_no_is_wrong(self, confirm_parser, full_menu):
        result = confirm_parser.parse("no", full_menu)
        assert result.result == "wrong"

    def test_no_with_fix(self, confirm_parser, full_menu):
        result = confirm_parser.parse("no, two waters instead", full_menu)
        assert result.result == "wrong"
        assert result.fix_order is not None
        items = {i.name: i.qty for i in result.fix_order.items}
        assert items.get("water") == 2

    def test_not_sure_is_wrong(self, confirm_parser, full_menu):
        result = confirm_parser.parse("not sure", full_menu)
        assert result.result == "wrong"
        assert result.reply is not None

    def test_wrong_is_wrong(self, confirm_parser, full_menu):
        result = confirm_parser.parse("wrong", full_menu)
        assert result.result == "wrong"

    def test_right_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("right", full_menu)
        assert result.result == "correct"

    # --- "no change" / "nothing changed" confirmation patterns ---

    def test_no_change_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("no change", full_menu)
        assert result.result == "correct"
        assert result.confidence >= 0.9

    def test_nothing_changed_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("nothing changed", full_menu)
        assert result.result == "correct"
        assert result.confidence >= 0.9

    def test_nothing_to_change_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("nothing to change", full_menu)
        assert result.result == "correct"

    def test_thats_right_no_change_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("that's right, there's no change need to make", full_menu)
        assert result.result == "correct"

    def test_no_changes_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("no changes", full_menu)
        assert result.result == "correct"

    def test_no_need_to_change_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("no need to change", full_menu)
        assert result.result == "correct"

    # --- Extended positive phrases ---

    def test_looks_good_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("looks good", full_menu)
        assert result.result == "correct"

    def test_thats_right_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("that's right", full_menu)
        assert result.result == "correct"

    def test_sounds_good_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("sounds good", full_menu)
        assert result.result == "correct"

    def test_all_good_is_correct(self, confirm_parser, full_menu):
        result = confirm_parser.parse("all good", full_menu)
        assert result.result == "correct"

    # --- Modification signals still work ---

    def test_change_fries_to_salad_is_wrong(self, confirm_parser, full_menu):
        result = confirm_parser.parse("change the fries to salad", full_menu)
        assert result.result == "wrong"
        assert result.fix_order is not None

    def test_add_fries_is_wrong(self, confirm_parser, full_menu):
        result = confirm_parser.parse("add fries", full_menu)
        assert result.result == "wrong"


# ------------------------------------------------------------------
# Menu context
# ------------------------------------------------------------------

class TestMenuContext:

    def test_candidates_for_coke(self, menu_provider):
        ctx = menu_provider.get_candidates("I want a coke")
        assert any(c.canonical == "coke" for c in ctx.candidates)

    def test_candidates_for_chips(self, menu_provider):
        ctx = menu_provider.get_candidates("chips")
        assert any(c.canonical == "fries" for c in ctx.candidates)

    def test_all_canonical_names(self, menu_provider):
        names = menu_provider.all_canonical_names
        assert "coke" in names
        assert "burger" in names
        assert "fries" in names

    def test_empty_text_returns_all(self, menu_provider):
        ctx = menu_provider.get_candidates("")
        assert len(ctx.canonical_names) > 0

    def test_fuzzy_match_for_misspelling(self, menu_provider):
        ctx = menu_provider.get_candidates("bugger")
        # Should fuzzy-match to burger
        assert len(ctx.candidates) > 0
