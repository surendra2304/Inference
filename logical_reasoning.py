"""
Inference/logical_reasoning.py — Inference Logical Reasoning Engine.

Extends Inference's capabilities with:
1. Propositional Logic: evaluate truth values of logical propositions
2. Predicate Logic: reason about relationships between objects
3. Inference Rules: apply modus ponens, modus tollens, syllogism
4. Formal Proof Construction: build formal logical proofs
5. Knowledge Base Reasoning: reason over a knowledge base
6. Non-Monotonic Logic: handle defeasible reasoning
7. Fuzzy Logic: reason with degrees of truth
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Proposition:
    """A logical proposition."""
    id: str
    statement: str
    truth_value: bool = False
    is_premise: bool = False
    dependencies: list[str] = field(default_factory=list)


@dataclass
class Predicate:
    """A predicate in predicate logic."""
    name: str
    arguments: list[str]
    arity: int
    truth_value: bool = False


@dataclass
class Rule:
    """A logical inference rule."""
    rule_id: str
    name: str
    antecedent: list[str]  # Proposition IDs
    consequent: str  # Proposition ID
    confidence: float = 1.0
    is_defeasible: bool = False


@dataclass
class ProofStep:
    """A step in a formal proof."""
    step_id: int
    statement: str
    rule_used: str
    premises: list[str]
    conclusion: str
    confidence: float = 1.0


@dataclass
class Proof:
    """A formal logical proof."""
    proof_id: str
    statement: str
    steps: list[ProofStep] = field(default_factory=list)
    is_valid: bool = False
    confidence: float = 0.0


@dataclass
class ReasoningResult:
    """Result of logical reasoning."""
    query: str
    answer: str
    confidence: float
    proof: Proof | None
    assumptions: list[str]
    contradictions: list[str]


class LogicalReasoning:
    """Logical reasoning engine."""

    def __init__(self):
        self._propositions: dict[str, Proposition] = {}
        self._predicates: dict[str, Predicate] = {}
        self._rules: dict[str, Rule] = {}
        self._proofs: dict[str, Proof] = {}

    # ── Propositional Logic ─────────────────────────────────────────────────

    def evaluate_proposition(
        self,
        statement: str,
        facts: dict[str, bool],
    ) -> bool:
        """Evaluate a proposition given known facts."""
        # Parse the statement
        # Handle basic operators: AND, OR, NOT, IMPLIES

        # Replace fact names with their truth values
        eval_str = statement

        for fact, value in facts.items():
            eval_str = eval_str.replace(fact, str(value).lower())

        # Replace logical operators
        eval_str = eval_str.replace("AND", "and")
        eval_str = eval_str.replace("OR", "or")
        eval_str = eval_str.replace("NOT", "not")
        eval_str = eval_str.replace("IMPLIES", "->")

        try:
            return eval(eval_str)
        except Exception:
            return False

    def apply_modus_ponens(
        self,
        rule_id: str,
        proposition_id: str,
    ) -> bool:
        """Apply modus ponens: if A AND (A -> B), then B."""
        rule = self._rules.get(rule_id)
        prop = self._propositions.get(proposition_id)

        if not rule or not prop:
            return False

        # Check if antecedent is satisfied
        for ant in rule.antecedent:
            ant_prop = self._propositions.get(ant)
            if not ant_prop or not ant_prop.truth_value:
                return False

        # Set consequent to true
        consequent = self._propositions.get(rule.consequent)
        if consequent:
            consequent.truth_value = True
            return True

        return False

    def apply_modus_tollens(
        self,
        rule_id: str,
        proposition_id: str,
    ) -> bool:
        """Apply modus tollens: if NOT B AND (A -> B), then NOT A."""
        rule = self._rules.get(rule_id)
        prop = self._propositions.get(proposition_id)

        if not rule or not prop:
            return False

        # Check if consequent is false
        consequent = self._propositions.get(rule.consequent)
        if consequent and consequent.truth_value:
            return False

        # Negate all antecedents
        for ant in rule.antecedent:
            ant_prop = self._propositions.get(ant)
            if ant_prop:
                ant_prop.truth_value = False

        return True

    # ── Predicate Logic ─────────────────────────────────────────────────────

    def evaluate_predicate(
        self,
        predicate: str,
        facts: list[dict],
    ) -> bool:
        """Evaluate a predicate against facts."""
        # Parse predicate: name(arg1, arg2, ...)
        match = re.match(r"(\w+)\(([\w, ]+)\)", predicate)
        if not match:
            return False

        pred_name = match.group(1)
        args = match.group(2).split(",")
        args = [a.strip() for a in args]

        # Check if any fact matches
        for fact in facts:
            if fact.get("predicate", "") == pred_name:
                fact_args = fact.get("arguments", [])
                if fact_args == args:
                    return fact.get("value", False)

        return False

    def universal_quantifier(
        self,
        predicate: str,
        domain: list[str],
        facts: list[dict],
    ) -> bool:
        """Check if a predicate holds for all elements in a domain."""
        for element in domain:
            # Substitute variable
            test_pred = predicate.replace("x", element)
            if not self.evaluate_predicate(test_pred, facts):
                return False

        return True

    def existential_quantifier(
        self,
        predicate: str,
        domain: list[str],
        facts: list[dict],
    ) -> bool:
        """Check if a predicate holds for at least one element in a domain."""
        for element in domain:
            test_pred = predicate.replace("x", element)
            if self.evaluate_predicate(test_pred, facts):
                return True

        return False

    # ── Inference Rules ─────────────────────────────────────────────────────

    def add_rule(
        self,
        name: str,
        antecedent: list[str],
        consequent: str,
        confidence: float = 1.0,
        is_defeasible: bool = False,
    ) -> Rule:
        """Add an inference rule."""
        rule_id = f"rule_{len(self._rules) + 1}"

        rule = Rule(
            rule_id=rule_id,
            name=name,
            antecedent=antecedent,
            consequent=consequent,
            confidence=confidence,
            is_defeasible=is_defeasible,
        )

        self._rules[rule_id] = rule
        return rule

    def add_proposition(
        self,
        statement: str,
        truth_value: bool = False,
        is_premise: bool = False,
    ) -> Proposition:
        """Add a proposition."""
        prop_id = f"prop_{len(self._propositions) + 1}"

        prop = Proposition(
            id=prop_id,
            statement=statement,
            truth_value=truth_value,
            is_premise=is_premise,
        )

        self._propositions[prop_id] = prop
        return prop

    def syllogism(
        self,
        major_premise: str,
        minor_premise: str,
        conclusion: str,
    ) -> Proof:
        """Apply syllogistic reasoning."""
        proof_id = f"proof_{int(time.time())}"
        proof = Proof(proof_id=proof_id, statement=conclusion)

        # Add premises
        p1 = self.add_proposition(major_premise, truth_value=True, is_premise=True)
        p2 = self.add_proposition(minor_premise, truth_value=True, is_premise=True)
        p3 = self.add_proposition(conclusion)

        # Add rule
        rule = self.add_rule(
            "syllogism",
            [p1.id, p2.id],
            p3.id,
        )

        # Apply rule
        success = self.apply_modus_ponens(rule.rule_id, p1.id)

        proof.steps.append(ProofStep(
            step_id=1,
            statement=major_premise,
            rule_used="premise",
            premises=[],
            conclusion=major_premise,
        ))

        proof.steps.append(ProofStep(
            step_id=2,
            statement=minor_premise,
            rule_used="premise",
            premises=[],
            conclusion=minor_premise,
        ))

        proof.steps.append(ProofStep(
            step_id=3,
            statement=conclusion,
            rule_used="syllogism",
            premises=[p1.id, p2.id],
            conclusion=p3.id,
            confidence=1.0 if success else 0.0,
        ))

        proof.is_valid = success
        proof.confidence = 1.0 if success else 0.0

        self._proofs[proof_id] = proof
        return proof

    # ── Knowledge Base Reasoning ─────────────────────────────────────────────

    def reason_over_kb(
        self,
        query: str,
        premises: list[str],
        rules: list[dict],
    ) -> ReasoningResult:
        """Reason over a knowledge base."""
        # Add premises
        premise_props = []
        for premise in premises:
            prop = self.add_proposition(premise, truth_value=True, is_premise=True)
            premise_props.append(prop)

        # Add rules
        for rule in rules:
            self.add_rule(
                rule["name"],
                rule["antecedent"],
                rule["consequent"],
                rule.get("confidence", 1.0),
            )

        # Create query proposition
        query_prop = self.add_proposition(query)

        # Apply forward chaining
        changed = True
        iterations = 0

        while changed and iterations < 100:
            changed = False
            iterations += 1

            for rule_id, rule in self._rules.items():
                if all(
                    self._propositions.get(ant) and self._propositions[ant].truth_value
                    for ant in rule.antecedent
                ):
                    consequent = self._propositions.get(rule.consequent)
                    if consequent and not consequent.truth_value:
                        consequent.truth_value = True
                        changed = True

        # Check if query is entailed
        answer = query_prop.truth_value
        confidence = query_prop.truth_value * (1.0 if query_prop.truth_value else 0.5)

        # Check for contradictions
        contradictions = []
        for p1_id, p1 in self._propositions.items():
            for p2_id, p2 in self._propositions.items():
                if p1_id != p2_id and p1.statement.lower() == p2.statement.lower() and p1.truth_value != p2.truth_value:
                    contradictions.append(f"Contradiction: {p1.statement}")

        return ReasoningResult(
            query=query,
            answer="True" if answer else "Unknown",
            confidence=confidence,
            proof=None,
            assumptions=premises,
            contradictions=contradictions,
        )

    # ── Non-Monotonic Logic ───────────────────────────────────────────────────

    def defeasible_reasoning(
        self,
        rules: list[dict],
        facts: list[str],
    ) -> dict[str, Any]:
        """Apply defeasible reasoning."""
        # Apply all rules
        derived = set(facts)
        defeated = set()

        # Apply rules in order
        for rule in rules:
            antecedent = rule["antecedent"]
            consequent = rule["consequent"]
            confidence = rule.get("confidence", 1.0)
            is_defeasible = rule.get("is_defeasible", False)

            # Check if antecedent is satisfied
            if all(ant in derived for ant in antecedent):
                if consequent in defeated:
                    continue  # Defeated by stronger rule

                derived.add(consequent)

        return {
            "derived_facts": list(derived),
            "defeated_facts": list(defeated),
            "confidence": len(derived) / max(1, len(facts) + len(rules)),
        }

    # ── Fuzzy Logic ─────────────────────────────────────────────────────────

    def fuzzy_evaluate(
        self,
        expression: str,
        memberships: dict[str, float],
    ) -> float:
        """Evaluate a fuzzy logic expression."""
        # Replace variables with membership values
        eval_str = expression

        for var, value in memberships.items():
            eval_str = eval_str.replace(var, str(value))

        # Replace fuzzy operators
        eval_str = eval_str.replace("AND", "min")
        eval_str = eval_str.replace("OR", "max")
        eval_str = eval_str.replace("NOT", "1 -")

        # Define min and max functions
        def min_(a, b):
            return min(a, b)

        def max_(a, b):
            return max(a, b)

        try:
            return eval(eval_str, {"min": min_, "max": max_})
        except Exception:
            return 0.0

    def fuzzy_inference(
        self,
        rules: list[dict],
        facts: dict[str, float],
    ) -> dict[str, float]:
        """Apply fuzzy inference."""
        conclusions = {}

        for rule in rules:
            antecedent = rule["antecedent"]
            consequent = rule["consequent"]

            # Calculate antecedent strength
            strength = 1.0
            for ant in antecedent:
                strength = min(strength, facts.get(ant, 0.0))

            # Apply to consequent
            if consequent not in conclusions:
                conclusions[consequent] = 0.0
            conclusions[consequent] = max(conclusions[consequent], strength)

        return conclusions

    # ── Reporting ──────────────────────────────────────────────────────────

    def get_reasoning_report(self) -> dict[str, Any]:
        """Generate a reasoning report."""
        return {
            "propositions": len(self._propositions),
            "rules": len(self._rules),
            "proofs": len(self._proofs),
            "valid_proofs": sum(1 for p in self._proofs.values() if p.is_valid),
        }
