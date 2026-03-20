"""
fol.py — First-Order Logic (FOL) Backward Chaining Engine
==========================================================

This module implements:
  1. A simple FOL representation (predicates, variables, constants, rules)
  2. A unification algorithm for FOL terms
  3. A backward chaining inference engine

Concepts
--------
- A **term** is either a Constant (e.g. "Robert"), a Variable (e.g. "?x"),
  or a Predicate with arguments (e.g. American(?x)).
- A **fact** is a ground predicate (no variables) asserted to be true.
- A **rule** (Horn clause) has the form:
      antecedent_1 AND antecedent_2 AND ... => consequent
  Read as: "If all antecedents are true, then the consequent is true."
- **Unification** finds a substitution (mapping of variables to terms) that
  makes two expressions identical.
- **Backward chaining** starts from a goal and works backward through rules
  to see if the goal can be proven from known facts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import copy


# ---------------------------------------------------------------------------
# 1. FOL Data Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Constant:
    """A constant symbol, e.g. 'Robert', 'America', 'T1'."""
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Variable:
    """A variable symbol, e.g. '?x', '?y'. Variables start with '?'."""
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Predicate:
    """
    A predicate (atomic sentence) like American(Robert) or Sells(?x, ?y, ?z).

    Attributes:
        name: The predicate name, e.g. "American"
        args: Tuple of arguments — each is a Constant or Variable
    """
    name: str
    args: tuple  # tuple of Constant | Variable

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.name}({args_str})"


@dataclass
class Rule:
    """
    A Horn clause rule: antecedents => consequent.

    Example:
        American(?p) AND Weapon(?q) AND Sells(?p, ?q, ?r) AND Hostile(?r)
            => Criminal(?p)

    Attributes:
        antecedents: list of Predicate (the IF part)
        consequent:  Predicate (the THEN part)
        name:        optional label for display
    """
    antecedents: list[Predicate]
    consequent: Predicate
    name: str = ""

    def __repr__(self):
        antes = " AND ".join(str(a) for a in self.antecedents)
        return f"[{self.name}] {antes} => {self.consequent}"


# ---------------------------------------------------------------------------
# 2. Substitution helpers
# ---------------------------------------------------------------------------

Substitution = dict  # Variable -> Constant | Variable


def apply_sub(term, sub: Substitution):
    """Apply a substitution to a term (Constant, Variable, or Predicate)."""
    if isinstance(term, Constant):
        return term
    if isinstance(term, Variable):
        if term in sub:
            # Follow the chain in case of transitive bindings
            return apply_sub(sub[term], sub)
        return term
    if isinstance(term, Predicate):
        new_args = tuple(apply_sub(a, sub) for a in term.args)
        return Predicate(term.name, new_args)
    raise TypeError(f"Unknown term type: {type(term)}")


# ---------------------------------------------------------------------------
# 3. Unification
# ---------------------------------------------------------------------------

FAILURE = None  # Sentinel for unification failure


def unify(x, y, sub: Optional[Substitution] = None) -> Optional[Substitution]:
    """
    Unify two terms and return the resulting substitution, or FAILURE.

    This implements the standard unification algorithm:
      - Two constants unify iff they are the same.
      - A variable unifies with anything; bind it in the substitution.
      - Two predicates unify iff they have the same name and arity,
        and all corresponding arguments unify.

    Parameters:
        x, y: terms to unify (Constant, Variable, or Predicate)
        sub:  existing substitution to extend (default: empty)

    Returns:
        Extended substitution dict, or None on failure.
    """
    if sub is None:
        sub = {}

    # If already failed, propagate
    if sub is FAILURE:
        return FAILURE

    # Apply current substitution first
    x = apply_sub(x, sub)
    y = apply_sub(y, sub)

    # Identical after substitution? Done.
    if x == y:
        return sub

    # Variable cases
    if isinstance(x, Variable):
        return _unify_var(x, y, sub)
    if isinstance(y, Variable):
        return _unify_var(y, x, sub)

    # Predicate cases
    if isinstance(x, Predicate) and isinstance(y, Predicate):
        if x.name != y.name or len(x.args) != len(y.args):
            return FAILURE
        for ax, ay in zip(x.args, y.args):
            sub = unify(ax, ay, sub)
            if sub is FAILURE:
                return FAILURE
        return sub

    return FAILURE


def _unify_var(var: Variable, term, sub: Substitution) -> Optional[Substitution]:
    """Bind var to term in sub, with occurs check."""
    if _occurs_check(var, term, sub):
        return FAILURE
    sub = dict(sub)  # copy so we don't mutate caller's dict
    sub[var] = term
    return sub


def _occurs_check(var: Variable, term, sub: Substitution) -> bool:
    """Return True if var occurs in term (prevents infinite loops)."""
    term = apply_sub(term, sub)
    if var == term:
        return True
    if isinstance(term, Predicate):
        return any(_occurs_check(var, a, sub) for a in term.args)
    return False


# ---------------------------------------------------------------------------
# 4. Standardize variables (rename to avoid clashes between rules)
# ---------------------------------------------------------------------------

_var_counter = 0


def _fresh_var() -> Variable:
    global _var_counter
    _var_counter += 1
    return Variable(f"?_v{_var_counter}")


def standardize_rule(rule: Rule) -> Rule:
    """
    Return a copy of the rule with all variables renamed to fresh names.
    This prevents variable name collisions when a rule is used multiple times.
    """
    mapping: dict[Variable, Variable] = {}

    def rename(term):
        if isinstance(term, Constant):
            return term
        if isinstance(term, Variable):
            if term not in mapping:
                mapping[term] = _fresh_var()
            return mapping[term]
        if isinstance(term, Predicate):
            return Predicate(term.name, tuple(rename(a) for a in term.args))
        return term

    new_antes = [rename(a) for a in rule.antecedents]
    new_conseq = rename(rule.consequent)
    return Rule(new_antes, new_conseq, rule.name)


# ---------------------------------------------------------------------------
# 5. Knowledge Base
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    A knowledge base stores facts and rules for inference.

    Usage:
        kb = KnowledgeBase()
        kb.add_fact(Predicate("American", (Constant("Robert"),)))
        kb.add_rule(Rule([...], consequent, name="R1"))
        results = kb.backward_chain(goal)
    """

    def __init__(self):
        self.facts: list[Predicate] = []
        self.rules: list[Rule] = []

    def add_fact(self, fact: Predicate):
        """Add a ground fact to the knowledge base."""
        self.facts.append(fact)

    def add_rule(self, rule: Rule):
        """Add a Horn clause rule to the knowledge base."""
        self.rules.append(rule)

    # ------------------------------------------------------------------
    # 5a. Backward Chaining — main entry point
    # ------------------------------------------------------------------

    def backward_chain(self, goal: Predicate, trace: bool = False) -> list[Substitution]:
        """
        Prove a goal using backward chaining and return all successful
        substitutions.

        Algorithm (from MIT 6.034 notes):
          1. Try to unify the goal with each known fact.
          2. If no fact matches, find rules whose consequent unifies with
             the goal, then recursively prove every antecedent.
          3. If nothing works, the goal cannot be proven.

        Parameters:
            goal:  the Predicate to prove
            trace: if True, print the proof search tree

        Returns:
            A list of substitutions. Each substitution, when applied to the
            goal, gives a proven instance. An empty list means the goal
            cannot be proven.
        """
        results = list(self._bc(goal, {}, depth=0, trace=trace))
        return results

    def _bc(self, goal: Predicate, sub: Substitution, depth: int, trace: bool):
        """
        Generator that yields every substitution that proves `goal`.

        This is the recursive heart of backward chaining.
        """
        indent = "  " * depth
        if trace:
            print(f"{indent}PROVE: {apply_sub(goal, sub)}")

        # ----------------------------------------------------------
        # Step 1: Try to match goal against known facts
        # ----------------------------------------------------------
        for fact in self.facts:
            unified = unify(goal, fact, dict(sub))
            if unified is not FAILURE:
                if trace:
                    print(f"{indent}  ✓ Matched fact: {fact}")
                yield unified

        # ----------------------------------------------------------
        # Step 2: Try rules whose consequent matches the goal
        # ----------------------------------------------------------
        for rule in self.rules:
            # Standardize variables so each rule application is independent
            std_rule = standardize_rule(rule)

            # Can the rule's consequent unify with our goal?
            unified = unify(goal, std_rule.consequent, dict(sub))
            if unified is FAILURE:
                continue

            if trace:
                print(f"{indent}  → Trying rule: {rule.name}")

            # Recursively prove all antecedents
            yield from self._prove_all(std_rule.antecedents, unified, depth + 1, trace)

    def _prove_all(self, goals: list[Predicate], sub: Substitution, depth: int, trace: bool):
        """
        Prove a list of goals conjunctively (all must succeed).
        Yields every substitution that satisfies ALL goals.
        """
        if not goals:
            # All goals proven — success!
            yield sub
            return

        first, rest = goals[0], goals[1:]

        # Prove the first goal; for each successful substitution,
        # continue proving the remaining goals.
        for sub1 in self._bc(first, sub, depth, trace):
            yield from self._prove_all(rest, sub1, depth, trace)

    # ------------------------------------------------------------------
    # 5b. Query helper
    # ------------------------------------------------------------------

    def query(self, goal: Predicate, trace: bool = False) -> list[Predicate]:
        """
        Query the KB and return all proven instances of the goal.

        Example:
            >>> kb.query(Predicate("Criminal", (Variable("?x"),)))
            [Criminal(Robert)]
        """
        results = []
        for sub in self.backward_chain(goal, trace=trace):
            result = apply_sub(goal, sub)
            if result not in results:
                results.append(result)
        return results


# ---------------------------------------------------------------------------
# 6. Convenience constructors (for cleaner test code)
# ---------------------------------------------------------------------------

def const(name: str) -> Constant:
    return Constant(name)


def var(name: str) -> Variable:
    if not name.startswith("?"):
        name = "?" + name
    return Variable(name)


def pred(name: str, *args) -> Predicate:
    """
    Build a Predicate from a name and a mix of strings.
    Strings starting with '?' become Variables; others become Constants.

    Example:
        pred("Sells", "?p", "?q", "?r")
        # => Predicate("Sells", (Variable("?p"), Variable("?q"), Variable("?r")))
    """
    parsed = []
    for a in args:
        if isinstance(a, (Constant, Variable)):
            parsed.append(a)
        elif isinstance(a, str) and a.startswith("?"):
            parsed.append(Variable(a))
        elif isinstance(a, str):
            parsed.append(Constant(a))
        else:
            raise ValueError(f"Unknown arg type: {a}")
    return Predicate(name, tuple(parsed))
