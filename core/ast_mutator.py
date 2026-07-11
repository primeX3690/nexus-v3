"""
core/ast_mutator.py

Genetic-programming mutation operators over Python ASTs.

Design note: mutating raw source text is fragile (easy to produce
syntactically invalid code). Mutating the AST and re-unparsing guarantees
every mutant is at least syntactically valid Python - semantic validity
(does it run, does it do the right thing) is what fitness_scorer.py
checks afterwards.

12 mutation types, grouped by what they target:
  Constants:      1. NumericPerturbation  2. NumericSignFlip
  Operators:      3. ArithmeticOpSwap     4. ComparisonOpSwap
                  5. BoolOpSwap           6. UnaryOpFlip
  Control flow:   7. ConditionNegate      8. BranchSwap (if/else)
                  9. LoopBoundShift
  Structure:     10. StatementDuplicate  11. StatementDelete
                 12. StatementReorder (within a block)
"""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from utils.seeding import get_rng


@dataclass
class MutationResult:
    mutated_tree: ast.AST
    mutation_applied: str
    success: bool
    error: str | None = None


class _NodeCollector(ast.NodeVisitor):
    """Collects nodes matching a predicate, tagged with their parent +
    field name so a mutator can replace them in place."""

    def __init__(self, predicate):
        self.predicate = predicate
        self.matches: list[tuple[ast.AST, str, int | None]] = []

    def generic_visit(self, node):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        if self.predicate(item):
                            self.matches.append((node, field, i))
                        self.visit(item)
            elif isinstance(value, ast.AST):
                if self.predicate(value):
                    self.matches.append((node, field, None))
                self.visit(value)


def _find_nodes(tree: ast.AST, predicate) -> list[tuple[ast.AST, str, int | None]]:
    collector = _NodeCollector(predicate)
    collector.visit(tree)
    return collector.matches


def _get_field(parent, field, idx):
    val = getattr(parent, field)
    return val[idx] if idx is not None else val


def _set_field(parent, field, idx, new_node):
    if idx is not None:
        getattr(parent, field)[idx] = new_node
    else:
        setattr(parent, field, new_node)


ARITH_OPS = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod]
COMPARE_OPS = [ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq]
BOOL_OPS = [ast.And, ast.Or]


def mutate_numeric_perturbation(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool))
    if not matches:
        return MutationResult(tree, "NumericPerturbation", False, "no numeric constants found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    delta = rng.normal(0, max(abs(node.value) * 0.2, 0.5))
    new_val = node.value + delta
    if isinstance(node.value, int):
        new_val = int(round(new_val))
    _set_field(parent, field, idx, ast.copy_location(ast.Constant(value=new_val), node))
    return MutationResult(tree, "NumericPerturbation", True)


def mutate_numeric_sign_flip(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool))
    if not matches:
        return MutationResult(tree, "NumericSignFlip", False, "no numeric constants found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    _set_field(parent, field, idx, ast.copy_location(ast.Constant(value=-node.value), node))
    return MutationResult(tree, "NumericSignFlip", True)


def mutate_arithmetic_op_swap(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.BinOp) and type(n.op) in ARITH_OPS)
    if not matches:
        return MutationResult(tree, "ArithmeticOpSwap", False, "no arithmetic ops found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    choices = [op for op in ARITH_OPS if op is not type(node.op)]
    node.op = choices[rng.integers(len(choices))]()
    return MutationResult(tree, "ArithmeticOpSwap", True)


def mutate_comparison_op_swap(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.Compare))
    if not matches:
        return MutationResult(tree, "ComparisonOpSwap", False, "no comparisons found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    pos = rng.integers(len(node.ops))
    choices = [op for op in COMPARE_OPS if op is not type(node.ops[pos])]
    node.ops[pos] = choices[rng.integers(len(choices))]()
    return MutationResult(tree, "ComparisonOpSwap", True)


def mutate_bool_op_swap(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.BoolOp))
    if not matches:
        return MutationResult(tree, "BoolOpSwap", False, "no bool ops found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    return MutationResult(tree, "BoolOpSwap", True)


def mutate_unary_op_flip(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)))
    if not matches:
        return MutationResult(tree, "UnaryOpFlip", False, "no unary +/- found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    node.op = ast.UAdd() if isinstance(node.op, ast.USub) else ast.USub()
    return MutationResult(tree, "UnaryOpFlip", True)


def mutate_condition_negate(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.If))
    if not matches:
        return MutationResult(tree, "ConditionNegate", False, "no if statements found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    ast.fix_missing_locations(node.test)
    return MutationResult(tree, "ConditionNegate", True)


def mutate_branch_swap(tree: ast.AST, rng) -> MutationResult:
    matches = _find_nodes(tree, lambda n: isinstance(n, ast.If) and len(n.orelse) > 0)
    if not matches:
        return MutationResult(tree, "BranchSwap", False, "no if/else with else-branch found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    node.body, node.orelse = node.orelse, node.body
    return MutationResult(tree, "BranchSwap", True)


def mutate_loop_bound_shift(tree: ast.AST, rng) -> MutationResult:
    def is_range_call(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "range" and len(n.args) >= 1
                and isinstance(n.args[-1], ast.Constant) and isinstance(n.args[-1].value, int))
    matches = _find_nodes(tree, is_range_call)
    if not matches:
        return MutationResult(tree, "LoopBoundShift", False, "no range(...) calls with constant bound found")
    parent, field, idx = matches[rng.integers(len(matches))]
    node = _get_field(parent, field, idx)
    shift = int(rng.integers(-2, 3))
    last_arg = node.args[-1]
    new_val = max(1, last_arg.value + shift)
    node.args[-1] = ast.copy_location(ast.Constant(value=new_val), last_arg)
    return MutationResult(tree, "LoopBoundShift", True)


def _statement_lists(tree: ast.AST):
    """Find every AST node that has a 'body' list of statements
    (FunctionDef, If, For, While, ...) so we can duplicate/delete/reorder
    a statement within a real block scope."""
    out = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            if hasattr(node, field):
                stmts = getattr(node, field)
                if isinstance(stmts, list) and len(stmts) >= 1 and all(isinstance(s, ast.stmt) for s in stmts):
                    out.append((node, field, stmts))
    return out


def mutate_statement_duplicate(tree: ast.AST, rng) -> MutationResult:
    blocks = [b for b in _statement_lists(tree) if len(b[2]) >= 1]
    if not blocks:
        return MutationResult(tree, "StatementDuplicate", False, "no statement blocks found")
    node, field, stmts = blocks[rng.integers(len(blocks))]
    pos = int(rng.integers(len(stmts)))
    stmts.insert(pos, copy.deepcopy(stmts[pos]))
    return MutationResult(tree, "StatementDuplicate", True)


def mutate_statement_delete(tree: ast.AST, rng) -> MutationResult:
    blocks = [b for b in _statement_lists(tree) if len(b[2]) >= 2]  # keep >=1 stmt after delete
    if not blocks:
        return MutationResult(tree, "StatementDelete", False, "no block with >=2 statements found")
    node, field, stmts = blocks[rng.integers(len(blocks))]
    pos = int(rng.integers(len(stmts)))
    del stmts[pos]
    return MutationResult(tree, "StatementDelete", True)


def mutate_statement_reorder(tree: ast.AST, rng) -> MutationResult:
    blocks = [b for b in _statement_lists(tree) if len(b[2]) >= 2]
    if not blocks:
        return MutationResult(tree, "StatementReorder", False, "no block with >=2 statements found")
    node, field, stmts = blocks[rng.integers(len(blocks))]
    i, j = rng.choice(len(stmts), size=2, replace=False)
    stmts[i], stmts[j] = stmts[j], stmts[i]
    return MutationResult(tree, "StatementReorder", True)


MUTATION_OPERATORS = {
    "numeric_perturbation": mutate_numeric_perturbation,
    "numeric_sign_flip": mutate_numeric_sign_flip,
    "arithmetic_op_swap": mutate_arithmetic_op_swap,
    "comparison_op_swap": mutate_comparison_op_swap,
    "bool_op_swap": mutate_bool_op_swap,
    "unary_op_flip": mutate_unary_op_flip,
    "condition_negate": mutate_condition_negate,
    "branch_swap": mutate_branch_swap,
    "loop_bound_shift": mutate_loop_bound_shift,
    "statement_duplicate": mutate_statement_duplicate,
    "statement_delete": mutate_statement_delete,
    "statement_reorder": mutate_statement_reorder,
}


class ASTMutator:
    """Applies random mutation operators to Python source code, retrying
    other operators if the chosen one doesn't find an applicable node
    (e.g. no if-statements to negate)."""

    def __init__(self, seed_stream: str = "ast_mutator"):
        self.rng = get_rng(seed_stream)
        self.operator_names = list(MUTATION_OPERATORS.keys())

    def mutate_source(self, source: str, max_attempts: int = 12) -> tuple[str, str]:
        """Returns (mutated_source, mutation_name_applied). Falls back to
        returning the original source unchanged (mutation_name='none') if
        no operator was applicable after max_attempts tries."""
        tree = ast.parse(source)
        order = list(self.operator_names)
        self.rng.shuffle(order)

        for name in order[:max_attempts]:
            candidate = copy.deepcopy(tree)
            result = MUTATION_OPERATORS[name](candidate, self.rng)
            if result.success:
                ast.fix_missing_locations(candidate)
                try:
                    mutated_source = ast.unparse(candidate)
                    compile(mutated_source, "<mutant>", "exec")  # syntax check
                    return mutated_source, name
                except (SyntaxError, ValueError):
                    continue
        return source, "none"