from __future__ import annotations

"""Wrapper around pyperplan to compute plans."""

from typing import List

from pyperplan.pddl.parser import Parser
from pyperplan.planner import _ground
from pyperplan.search import breadth_first_search


def plan(domain_pddl: str, problem_pddl: str) -> List[str]:
    """Return a list of action strings forming a plan."""
    parser = Parser(domain_pddl, problem_pddl)
    parser.domInput = domain_pddl
    domain = parser.parse_domain(read_from_file=False)
    parser.probInput = problem_pddl
    problem = parser.parse_problem(domain, read_from_file=False)
    task = _ground(problem)
    solution = breadth_first_search(task)
    if not solution:
        return []
    return [op.name for op in solution]
