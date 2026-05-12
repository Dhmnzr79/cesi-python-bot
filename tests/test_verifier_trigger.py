"""Детерминированный high-risk триггер A7 (без LLM)."""
from __future__ import annotations

import unittest

from verifier import collect_high_risk_signals


class TestVerifierTrigger(unittest.TestCase):
    def test_digit_triggers(self) -> None:
        self.assertIn("digit", collect_high_risk_signals("Гарантия 3 года."))

    def test_percent_triggers(self) -> None:
        s = collect_high_risk_signals("Эффективность 12,5%.")
        self.assertIn("percent", s)

    def test_currency_rub(self) -> None:
        self.assertIn("currency_rub", collect_high_risk_signals("Стоимость 40 000 ₽."))

    def test_payment_word(self) -> None:
        self.assertIn("payment_or_price_word", collect_high_risk_signals("Есть рассрочка без переплат."))

    def test_time_promise(self) -> None:
        self.assertIn("time_promise", collect_high_risk_signals("Можно сделать за один день."))

    def test_absolute(self) -> None:
        self.assertIn("absolute_claim", collect_high_risk_signals("Вам точно можно без ограничений."))

    def test_soft_answer_no_trigger(self) -> None:
        self.assertEqual(collect_high_risk_signals("Процедура проходит под анестезией."), [])

    def test_sorted_stable(self) -> None:
        a = collect_high_risk_signals("Цена 10 000 ₽ и скидка 5%.")
        b = collect_high_risk_signals("Цена 10 000 ₽ и скидка 5%.")
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))


class TestVerifierSourceAppend(unittest.TestCase):
    def test_no_append_returns_chunk_only(self) -> None:
        from chunk_responder import verifier_effective_source_body

        self.assertEqual(
            verifier_effective_source_body(chunk_md_body="  body  ", generator_append_text=None),
            "body",
        )

    def test_append_merged_with_marker(self) -> None:
        from chunk_responder import verifier_effective_source_body

        out = verifier_effective_source_body(chunk_md_body="chunk", generator_append_text="Цена 100 ₽")
        self.assertTrue(out.startswith("chunk"))
        self.assertIn("Цена 100 ₽", out)
        self.assertIn("детерминированное", out.lower())


if __name__ == "__main__":
    unittest.main()
