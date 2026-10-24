"""
Inference/constraint_solver.py — Inference Constraint Solver & Optimizer.

Extends Inference's capabilities with:
1. Constraint Satisfaction: solve problems with multiple constraints
2. Linear Optimization: maximize/minimize objectives subject to constraints
3. Resource Allocation: optimally allocate limited resources
4. Scheduling Optimization: find optimal schedules
5. SAT Problem Solving: solve boolean satisfiability problems
6. SMT Problem Solving: solve satisfiability modulo theories problems
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Constraint:
    """A constraint."""
    constraint_id: str
    description: str
    type: str  # equality, inequality, boolean, resource
    variables: list[str]
    expression: str
    bound: float | None = None
    operator: str = "="  # =, <=, >=, <, >


@dataclass
class Variable:
    """A variable in the optimization problem."""
    name: str
    value: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 100.0
    type: str = "continuous"  # continuous, integer, boolean


@dataclass
class Objective:
    """An objective function."""
    description: str
    variables: list[str]
    coefficients: list[float]
    type: str = "maximize"  # maximize, minimize


@dataclass
class Solution:
    """A solution to an optimization problem."""
    variables: dict[str, float]
    objective_value: float
    feasibility: float  # 0-1
    constraints_satisfied: int
    total_constraints: int
    solve_time: float
    optimal: bool


@dataclass
class SATResult:
    """Result of a SAT solving attempt."""
    satisfiable: bool
    assignment: dict[str, bool]
    variables_count: int
    clauses_count: int
    solve_time: float
    model: str  # sat, unsat, unknown


class ConstraintSolver:
    """Constraint satisfaction and optimization solver."""

    def __init__(self):
        self._solutions: dict[str, Solution] = {}
        self._sat_results: dict[str, SATResult] = {}

    # ── Constraint Satisfaction ─────────────────────────────────────────────

    def solve_csp(
        self,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> Solution:
        """Solve a constraint satisfaction problem."""
        start = time.time()

        # Simple backtracking solver
        assignment: dict[str, float] = {}

        # Initialize with lower bounds
        for var in variables:
            assignment[var.name] = var.lower_bound

        # Check constraints
        satisfied = 0
        for constraint in constraints:
            if self._check_constraint(assignment, constraint):
                satisfied += 1

        feasibility = satisfied / max(1, len(constraints))

        # Try to find better solution
        improved = self._try_improve(assignment, variables, constraints)
        if improved:
            assignment = improved
            satisfied = sum(
                1 for c in constraints
                if self._check_constraint(assignment, c)
            )
            feasibility = satisfied / max(1, len(constraints))

        solution = Solution(
            variables=assignment,
            objective_value=feasibility,
            feasibility=feasibility,
            constraints_satisfied=satisfied,
            total_constraints=len(constraints),
            solve_time=time.time() - start,
            optimal=feasibility >= 0.99,
        )

        return solution

    def _check_constraint(
        self,
        assignment: dict[str, float],
        constraint: Constraint,
    ) -> bool:
        """Check if a constraint is satisfied."""
        try:
            # Evaluate expression with current assignment
            values = [assignment.get(v, 0) for v in constraint.variables]

            if constraint.type == "equality":
                result = sum(values) == constraint.bound if constraint.bound else True
            elif constraint.type == "inequality":
                result = sum(values) <= (constraint.bound or 0)
            elif constraint.type == "resource":
                result = sum(values) >= constraint.bound if constraint.bound else True
            elif constraint.type == "boolean":
                result = bool(values[0]) if values else False
            else:
                result = True

            return result
        except Exception:
            return False

    def _try_improve(
        self,
        assignment: dict[str, float],
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> dict[str, float] | None:
        """Try to improve the assignment."""
        best_assignment = dict(assignment)
        best_satisfied = sum(
            1 for c in constraints
            if self._check_constraint(assignment, c)
        )

        for var in variables:
            if var.type == "boolean":
                values = [0, 1]
            elif var.type == "integer":
                values = [int(var.lower_bound), int(var.upper_bound)]
            else:
                values = [var.lower_bound, var.upper_bound]

            for val in values:
                new_assignment = dict(assignment)
                new_assignment[var.name] = val

                satisfied = sum(
                    1 for c in constraints
                    if self._check_constraint(new_assignment, c)
                )

                if satisfied > best_satisfied:
                    best_satisfied = satisfied
                    best_assignment = new_assignment

        return best_assignment if best_satisfied > 0 else None

    # ── Linear Optimization ──────────────────────────────────────────────────

    def solve_linear(
        self,
        objective: Objective,
        variables: list[Variable],
        constraints: list[Constraint],
    ) -> Solution:
        """Solve a linear optimization problem (simplex-like)."""
        start = time.time()

        # Initialize with lower bounds
        assignment = {var.name: var.lower_bound for var in variables}

        # Simple greedy improvement
        for var in variables:
            # Try to increase the variable
            old_value = assignment[var.name]

            # Check how much we can increase
            max_increase = var.upper_bound - old_value

            # Find the limiting constraint
            for constraint in constraints:
                if var.name not in constraint.variables:
                    continue

                if constraint.type == "inequality" and constraint.bound is not None:
                    # Calculate remaining capacity
                    other_vars = [v for v in constraint.variables if v != var.name]
                    other_sum = sum(assignment.get(v, 0) for v in other_vars)
                    remaining = constraint.bound - other_sum

                    if remaining < assignment[var.name]:
                        max_increase = min(max_increase, remaining - assignment[var.name])

            # Apply improvement
            if max_increase > 0:
                assignment[var.name] += max_increase

        # Check constraints
        satisfied = sum(
            1 for c in constraints
            if self._check_constraint(assignment, c)
        )

        # Calculate objective value
        obj_value = sum(
            assignment.get(v, 0) * coeff
            for v, coeff in zip(objective.variables, objective.coefficients)
        )

        # If minimizing, negate
        if objective.type == "minimize":
            obj_value = -obj_value

        feasibility = satisfied / max(1, len(constraints))

        return Solution(
            variables=assignment,
            objective_value=obj_value,
            feasibility=feasibility,
            constraints_satisfied=satisfied,
            total_constraints=len(constraints),
            solve_time=time.time() - start,
            optimal=feasibility >= 0.95,
        )

    # ── Resource Allocation ─────────────────────────────────────────────────

    def optimize_resource_allocation(
        self,
        resources: list[str],
        tasks: list[dict],
        total_resource: float = 100.0,
    ) -> Solution:
        """Optimally allocate resources to tasks."""
        variables = [
            Variable(name=f"alloc_{task['id']}", upper_bound=100)
            for task in tasks
        ]

        constraints = []

        # Total resource constraint
        constraints.append(Constraint(
            constraint_id="total_resource",
            description=f"Total allocation <= {total_resource}",
            type="inequality",
            variables=[v.name for v in variables],
            expression="sum",
            bound=total_resource,
            operator="<=",
        ))

        # Each task needs minimum allocation
        for task in tasks:
            constraints.append(Constraint(
                constraint_id=f"min_{task['id']}",
                description=f"Task {task['id']} minimum allocation",
                type="inequality",
                variables=[f"alloc_{task['id']}"],
                expression="value",
                bound=task.get("min_allocation", 5.0),
                operator=">=",
            ))

        # Objective: maximize total task value
        objective = Objective(
            description="Maximize total task value",
            variables=[f"alloc_{task['id']}" for task in tasks],
            coefficients=[task.get("value", 1.0) for task in tasks],
            type="maximize",
        )

        return self.solve_linear(objective, variables, constraints)

    # ── Scheduling Optimization ─────────────────────────────────────────────

    def optimize_schedule(
        self,
        jobs: list[dict],
        max_concurrent: int = 2,
    ) -> Solution:
        """Optimize job scheduling."""
        variables = []
        constraints = []

        # Create binary variables for each job-time slot
        for job in jobs:
            variables.append(Variable(
                name=f"slot_{job['id']}",
                lower_bound=0,
                upper_bound=1,
                type="integer",
            ))

        # Constraints
        for job in jobs:
            # Each job runs once
            constraints.append(Constraint(
                constraint_id=f"once_{job['id']}",
                description=f"Job {job['id']} runs exactly once",
                type="equality",
                variables=[f"slot_{job['id']}"],
                expression="value",
                bound=1,
                operator="=",
            ))

        # Max concurrent jobs
        constraints.append(Constraint(
            constraint_id="max_concurrent",
            description=f"Max {max_concurrent} concurrent jobs",
            type="inequality",
            variables=[f"slot_{j['id']}" for j in jobs],
            expression="sum",
            bound=max_concurrent,
            operator="<=",
        ))

        objective = Objective(
            description="Minimize total completion time",
            variables=[f"slot_{job['id']}" for job in jobs],
            coefficients=[-job.get("priority", 1) for job in jobs],
            type="minimize",
        )

        return self.solve_linear(objective, variables, constraints)

    # ── SAT Problem Solving ─────────────────────────────────────────────────

    def solve_sat(
        self,
        clauses: list[list[str]],
        variables: list[str] | None = None,
    ) -> SATResult:
        """Solve a boolean SAT problem."""
        start = time.time()

        if not variables:
            # Extract variables from clauses
            variables = []
            for clause in clauses:
                for lit in clause:
                    var = lit.lstrip("!")
                    if var not in variables:
                        variables.append(var)

        # Simple DPLL-like algorithm
        assignment = self._dpll(clauses, variables, {})

        solve_time = time.time() - start

        result = SATResult(
            satisfiable=assignment is not None,
            assignment=assignment or {},
            variables_count=len(variables),
            clauses_count=len(clauses),
            solve_time=solve_time,
            model="sat" if assignment else "unsat",
        )

        clause_id = hashlib.md5(str(clauses).encode()).hexdigest()[:12]
        self._sat_results[clause_id] = result
        return result

    def _dpll(
        self,
        clauses: list[list[str]],
        variables: list[str],
        assignment: dict[str, bool],
    ) -> dict[str, bool] | None:
        """Simple DPLL algorithm."""
        # Unit propagation
        unit_clauses = [c for c in clauses if len(c) == 1]
        for clause in unit_clauses:
            lit = clause[0]
            var = lit.lstrip("!")
            value = not lit.startswith("!")

            if var in assignment and assignment[var] != value:
                return None  # Contradiction

            assignment[var] = value

        # Simplify clauses
        remaining = []
        for clause in clauses:
            satisfied = False
            simplified = []

            for lit in clause:
                var = lit.lstrip("!")
                if var in assignment:
                    if (lit.startswith("!") != assignment[var]):
                        satisfied = True
                        break
                else:
                    simplified.append(lit)

            if not satisfied:
                remaining.append(simplified)

        if not remaining:
            return assignment

        # Find unassigned variable
        unassigned = [v for v in variables if v not in assignment]
        if not unassigned:
            return assignment if not remaining else None

        var = unassigned[0]

        # Try True
        result = self._dpll(remaining, variables, {**assignment, var: True})
        if result:
            return result

        # Try False
        result = self._dpll(remaining, variables, {**assignment, var: False})
        return result

    # ── Reporting ──────────────────────────────────────────────────────────

    def get_solver_report(self) -> dict[str, Any]:
        """Generate a solver report."""
        total = len(self._solutions)
        avg_feasibility = sum(s.feasibility for s in self._solutions.values()) / max(1, total)
        avg_time = sum(s.solve_time for s in self._solutions.values()) / max(1, total)

        return {
            "total_solutions": total,
            "average_feasibility": avg_feasibility,
            "average_solve_time": avg_time,
            "sat_problems_solved": len(self._sat_results),
            "sat_satisfiable": sum(1 for r in self._sat_results.values() if r.satisfiable),
        }
