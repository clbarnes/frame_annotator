from argparse import Namespace

from fran.__main__ import Config
from fran.constants import config_path, default_config_dict


def test_keys_updates():
    keys1 = {"a": "a", "b": "b"}
    c1 = Config(keys=keys1)
    keys2 = {"b": "not b", "c": "c"}
    c2 = c1.replace(keys=keys2)

    expected = keys1.copy()
    expected.update(keys2)
    assert c2.keys == expected


def test_from_toml():
    Config().replace_from_toml(config_path)


def test_from_dict():
    Config().replace_from_config_dict(default_config_dict)


def test_from_kwargs():
    Config().replace(infile="potato")


def test_from_namespace():
    ns = Namespace(infile="potato")
    Config().replace_from_namespace(ns)
