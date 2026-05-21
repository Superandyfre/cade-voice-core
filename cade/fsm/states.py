"""Ordering sub-FSM state enum."""

from enum import Enum


class OrderState(str, Enum):
    """States for the ordering sub-FSM lifecycle."""

    NOT_PERMITTED = "NOT_PERMITTED"
    PERMITTED = "PERMITTED"
    ASK = "ASK"
    LISTEN = "LISTEN"
    REPEAT = "REPEAT"
    CHECK = "CHECK"
    FINISH = "FINISH"
