"""
test_fol.py — Unit tests for the FOL backward chaining engine.

Run with:  python -m pytest test_fol.py -v
    or:    python test_fol.py
"""

import unittest
from fol import (
    KnowledgeBase, Rule, Predicate, Constant, Variable,
    pred, const, var, unify, apply_sub, FAILURE
)


class TestUnification(unittest.TestCase):
    """Test the unification algorithm."""

    def test_identical_constants(self):
        sub = unify(Constant("A"), Constant("A"))
        self.assertIsNotNone(sub)

    def test_different_constants_fail(self):
        sub = unify(Constant("A"), Constant("B"))
        self.assertIsNone(sub)

    def test_variable_binds_to_constant(self):
        sub = unify(Variable("?x"), Constant("A"))
        self.assertIsNotNone(sub)
        self.assertEqual(sub[Variable("?x")], Constant("A"))

    def test_predicate_unification(self):
        p1 = pred("P", "?x", "B")
        p2 = pred("P", "A", "?y")
        sub = unify(p1, p2)
        self.assertIsNotNone(sub)
        self.assertEqual(apply_sub(Variable("?x"), sub), Constant("A"))
        self.assertEqual(apply_sub(Variable("?y"), sub), Constant("B"))

    def test_predicate_name_mismatch_fails(self):
        p1 = pred("P", "A")
        p2 = pred("Q", "A")
        self.assertIsNone(unify(p1, p2))

    def test_predicate_arity_mismatch_fails(self):
        p1 = pred("P", "A", "B")
        p2 = pred("P", "A")
        self.assertIsNone(unify(p1, p2))

    def test_conflicting_bindings_fail(self):
        # ?x can't be both A and B
        p1 = pred("P", "?x", "?x")
        p2 = pred("P", "A", "B")
        self.assertIsNone(unify(p1, p2))

    def test_transitive_binding(self):
        p1 = pred("P", "?x", "?y")
        p2 = pred("P", "?y", "C")
        sub = unify(p1, p2)
        self.assertIsNotNone(sub)
        self.assertEqual(apply_sub(Variable("?x"), sub), Constant("C"))
        self.assertEqual(apply_sub(Variable("?y"), sub), Constant("C"))

    def test_occurs_check(self):
        # ?x unifying with P(?x) should fail (infinite structure)
        p1 = Variable("?x")
        p2 = pred("P", "?x")
        sub = unify(p1, p2)
        self.assertIsNone(sub)


class TestBackwardChaining(unittest.TestCase):
    """Test backward chaining inference."""

    def _build_criminal_kb(self) -> KnowledgeBase:
        """Build the standard Robert-the-criminal KB."""
        kb = KnowledgeBase()
        kb.add_fact(pred("American", "Robert"))
        kb.add_fact(pred("Enemy", "A", "America"))
        kb.add_fact(pred("Owns", "A", "T1"))
        kb.add_fact(pred("Missile", "T1"))

        kb.add_rule(Rule(
            [pred("American", "?p"), pred("Weapon", "?q"),
             pred("Sells", "?p", "?q", "?r"), pred("Hostile", "?r")],
            pred("Criminal", "?p"), "R1"
        ))
        kb.add_rule(Rule(
            [pred("Missile", "?p"), pred("Owns", "A", "?p")],
            pred("Sells", "Robert", "?p", "A"), "R2"
        ))
        kb.add_rule(Rule(
            [pred("Missile", "?p")],
            pred("Weapon", "?p"), "R3"
        ))
        kb.add_rule(Rule(
            [pred("Enemy", "?p", "America")],
            pred("Hostile", "?p"), "R4"
        ))
        return kb

    def test_criminal_proven(self):
        kb = self._build_criminal_kb()
        results = kb.query(pred("Criminal", "?x"))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], pred("Criminal", "Robert"))

    def test_criminal_specific(self):
        kb = self._build_criminal_kb()
        results = kb.query(pred("Criminal", "Robert"))
        self.assertEqual(len(results), 1)

    def test_non_criminal(self):
        kb = self._build_criminal_kb()
        results = kb.query(pred("Criminal", "Alice"))
        self.assertEqual(len(results), 0)

    def test_fact_lookup(self):
        kb = self._build_criminal_kb()
        results = kb.query(pred("American", "Robert"))
        self.assertEqual(len(results), 1)

    def test_derived_fact(self):
        kb = self._build_criminal_kb()
        results = kb.query(pred("Weapon", "T1"))
        self.assertEqual(len(results), 1)

    def test_ancestor_transitive(self):
        kb = KnowledgeBase()
        kb.add_fact(pred("Parent", "Tom", "Bob"))
        kb.add_fact(pred("Parent", "Bob", "Ann"))

        kb.add_rule(Rule(
            [pred("Parent", "?x", "?y")],
            pred("Ancestor", "?x", "?y"), "base"
        ))
        kb.add_rule(Rule(
            [pred("Parent", "?x", "?z"), pred("Ancestor", "?z", "?y")],
            pred("Ancestor", "?x", "?y"), "transitive"
        ))

        results = kb.query(pred("Ancestor", "Tom", "?who"))
        names = {str(r) for r in results}
        self.assertIn("Ancestor(Tom, Bob)", names)
        self.assertIn("Ancestor(Tom, Ann)", names)

    def test_no_matching_rule(self):
        kb = KnowledgeBase()
        kb.add_fact(pred("A", "x"))
        results = kb.query(pred("B", "x"))
        self.assertEqual(len(results), 0)

    def test_multiple_facts(self):
        kb = KnowledgeBase()
        kb.add_fact(pred("Color", "Red"))
        kb.add_fact(pred("Color", "Blue"))
        kb.add_fact(pred("Color", "Green"))
        results = kb.query(pred("Color", "?x"))
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
