import pytest

from fran.events import EventLogger
from fran.common import Special


def test_eventlogger():
    el = EventLogger()

    assert list(el.start_stop_pairs("a")) == []


def test_simple():
    el = EventLogger()
    assert list(el.start_stop_pairs("a")) == []


def test_simple_startstop():
    el = EventLogger()

    el.insert("a", 10)
    el.insert("A", 20)

    assert list(el.start_stop_pairs("a")) == [(10, 20)]


def test_overlapping():
    el = EventLogger()

    el.insert("a", 10)
    el.insert("a", 15)
    el.insert("A", 20)
    el.insert("A", 25)

    assert list(el.start_stop_pairs("a")) == [(10, 20), (15, 25)]


def test_start_stop_same():
    el = EventLogger()

    el.insert("a", 10)
    el.insert("a", 20)
    el.insert("A", 20)
    el.insert("A", 25)

    assert list(el.start_stop_pairs("a")) == [(10, 20), (20, 25)]


def test_from_before():
    el = EventLogger()
    el.insert("A", 20)
    assert list(el.start_stop_pairs("a")) == [(Special.BEFORE, 20)]


def test_to_after():
    el = EventLogger()
    el.insert("a", 10)
    assert list(el.start_stop_pairs("a")) == [(10, Special.AFTER)]


def test_undo():
    el = EventLogger()

    el.insert("a", 10)
    el.insert("A", 20)

    assert list(el.start_stop_pairs("a")) == [(10, 20)]
    el.undo()
    assert list(el.start_stop_pairs("a")) == [(10, Special.AFTER)]
    el.undo()
    assert list(el.start_stop_pairs("a")) == []


def test_redo():
    el = EventLogger()

    el.insert("a", 10)
    el.insert("A", 20)

    assert list(el.start_stop_pairs("a")) == [(10, 20)]
    el.undo()
    assert list(el.start_stop_pairs("a")) == [(10, Special.AFTER)]
    el.undo()
    assert list(el.start_stop_pairs("a")) == []
    el.redo()
    assert list(el.start_stop_pairs("a")) == [(10, Special.AFTER)]
    el.redo()
    assert list(el.start_stop_pairs("a")) == [(10, 20)]


def test_extra_before():
    el = EventLogger()

    el.insert("a", 5)
    el.insert("a", 10)
    el.insert("A", 20)

    assert list(el.start_stop_pairs("a")) == [(5, 20), (10, Special.AFTER)]


@pytest.mark.xfail
def test_extra_after():
    el = EventLogger()

    el.insert("a", 10)
    el.insert("A", 20)
    el.insert("A", 25)

    assert list(el.start_stop_pairs("a")) == [(Special.BEFORE, 20), (10, 25)]
