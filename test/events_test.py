from fran.events import EventLogger


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
