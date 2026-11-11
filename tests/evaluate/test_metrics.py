# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import pytest

import dspy
from dspy.evaluate.metrics import EM, F1, answer_exact_match, answer_passage_match, normalize_text
from dspy.predict import Predict


def test_answer_exact_match_string():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = dspy.Example(
        question="What is 1+1?",
        answer=["2", "two"],
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("\u201cParis\u201d", "Paris"),  # curly double quotes
        ("\u2018Paris\u2019", "Paris"),  # curly single quotes
        ("O\u2019Brien", "O'Brien"),  # typographic apostrophe
        ("Paris\u2026", "Paris"),  # ellipsis
        ("\u00abParis\u00bb", "Paris"),  # guillemets
        ("Paris\u2014France", "Paris-France"),  # em dash vs hyphen
        ("\u5df4\u9ece\u3002", "\u5df4\u9ece"),  # ideographic full stop
        ("\uff0842\uff09", "42"),  # full-width parentheses
    ],
)
def test_unicode_punctuation_is_removed(prediction, gold):
    assert normalize_text(prediction) == normalize_text(gold)
    assert EM(prediction, [gold])
    assert F1(prediction, [gold]) == 1.0


def test_normalize_text_keeps_non_punctuation_characters():
    # Letters, combining accents and currency symbols are not punctuation.
    assert normalize_text("\u5df4\u9ece") == "\u5df4\u9ece"
    assert not EM("caf\u00e9", ["cafe"])
    assert normalize_text("\u20ac5") == "\u20ac5"


def test_answer_exact_match_ignores_curly_quotes():
    example = dspy.Example(question="Capital of France?", answer="Paris").with_inputs("question")
    assert answer_exact_match(example, dspy.Prediction(answer="\u201cParis\u201d"))
    assert answer_exact_match(example, dspy.Prediction(answer="It is \u201cParis\u201d."), frac=0.5)


@pytest.mark.parametrize(
    ("answer", "passage"),
    [
        ("1990", "He served from 1990\u20131995."),  # en-dash range
        ("Paris", "France\u2014Paris\u2014is large."),  # unspaced em dashes
        ("\u4e0a\u6d77", "\u5317\u4eac\u3001\u4e0a\u6d77\u3001\u5e7f\u5dde"),  # ideographic comma
        ("Paris", "The capital is \u201cParis\u201d."),  # curly quotes
    ],
)
def test_answer_passage_match_with_unicode_punctuation(answer, passage):
    example = dspy.Example(question="q", answer=answer).with_inputs("question")
    assert answer_passage_match(example, dspy.Prediction(context=[passage]))
