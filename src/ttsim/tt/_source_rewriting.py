"""Rewrite author-written ``and``/``or``/``not`` into ``logical_*`` calls.

Python's ``and`` and ``or`` short-circuit through ``__bool__`` and hand back one
operand whole; ``not`` consumes ``__bool__`` and returns a plain ``bool``. None
of the three can therefore combine or preserve a custom object. Rewriting them
into ``{module}.logical_and`` / ``logical_or`` / ``logical_not`` calls on a
supplied module object routes them through code that can.

Two independent clients rewrite the same way:

- the array vectorizer, whose numpy/JAX arrays raise on ``__bool__``;
- the build-time unit checker, whose quantity type carries a grouping level that
  ``not`` would drop.

Neither knows about the other; both import the machinery from here.
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import types
from collections.abc import Callable, Mapping
from typing import Any, cast

from dags.signature import rename_arguments

# `functools.WRAPPER_ASSIGNMENTS` minus the annotation attributes. Used at
# every `functools.wraps` site that wraps a user policy function: if we let
# the user's scalar annotations leak onto the column-typed wrapper,
# beartype rejects the wrapper's column-typed arguments against the
# wrapper's inherited scalar signature.
#
# `__annotate__` is the PEP 649 (Python 3.14+) deferred-evaluation pair to
# `__annotations__` and needs the same treatment.
WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS: tuple[str, ...] = tuple(
    a
    for a in functools.WRAPPER_ASSIGNMENTS
    if a not in ("__annotations__", "__annotate__")
)


def recompile_with_logical_ops_as_calls(
    func: Callable[..., Any],
    module: str,
    module_obj: Any,  # noqa: ANN401
    extra_globals: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    """Return a copy of ``func`` with ``and``/``or``/``not`` as ``{module}.logical_*``
    calls.

    The recompiled body binds ``module`` to ``module_obj``, so the caller decides
    what ``logical_and``, ``logical_or`` and ``logical_not`` mean for its own value
    type. Everything else about the body is left alone.

    Args:
        func: The function whose source is rewritten.
        module: The name the rewritten calls are attributed to, bound in the
            recompiled body's scope.
        module_obj: The object bound to ``module`` — it must export
            ``logical_and``, ``logical_or`` and ``logical_not``.
        extra_globals: Module-level names to rebind in the recompiled body's
            scope, so a body that calls them runs against the caller's stand-ins.
            When given, the body is rebound even if it has no ``and``/``or``/``not``
            — the rebinding, not the rewrite, is then the point.

    Returns:
        The recompiled function, or ``func`` itself when there is nothing to do: a
        lambda, a body with neither a boolean operator nor ``extra_globals``, or a
        function whose source is unavailable (a builtin, a C function, a REPL
        definition) or unparseable.
    """
    if is_lambda_function(func):
        return func
    try:
        tree = func_to_ast(func)
    except (OSError, TypeError):
        return func
    has_boolop = any(isinstance(node, ast.BoolOp) for node in ast.walk(tree))
    has_not = any(
        isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        for node in ast.walk(tree)
    )
    if not has_boolop and not has_not and not extra_globals:
        return func

    if has_boolop or has_not:

        class _LogicalOpRewriter(ast.NodeTransformer):
            def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
                self.generic_visit(node)
                return boolop_to_call(node=node, module=module)

            def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp | ast.Call:
                self.generic_visit(node)
                if isinstance(node.op, ast.Not):
                    return not_to_call(node=node, module=module)
                return node

        _LogicalOpRewriter().visit(tree)
        ast.fix_missing_locations(tree)
    return recompile_from_ast(
        func=func,
        tree=tree,
        scope_bindings={module: module_obj, **(extra_globals or {})},
        filename="<logical-ops-as-calls>",
    )


def recompile_from_ast(
    func: Callable[..., Any],
    tree: ast.Module,
    scope_bindings: Mapping[str, Any],
    filename: str,
) -> Callable[..., Any]:
    """Execute a rewritten AST in ``func``'s scope and restore ``func``'s identity.

    Rebuilds the defining scope (module globals plus dereferenced closure cells),
    overlays ``scope_bindings``, executes ``tree``, and wraps the resulting
    function with ``func``'s metadata (annotations excluded — the rewrite changes
    the calling convention). The AST carries the original argument names, and the
    rewrites never touch the argument list, so any names renamed dynamically after
    definition are matched positionally against ``func``'s live signature.
    """
    scope = dict(func.__globals__)  # ty: ignore[unresolved-attribute]
    if func.__closure__:  # ty: ignore[unresolved-attribute]
        closure_vars = func.__code__.co_freevars  # ty: ignore[unresolved-attribute]
        closure_cells = [c.cell_contents for c in func.__closure__]  # ty: ignore[unresolved-attribute]
        scope.update(dict(zip(closure_vars, closure_cells, strict=False)))
    scope.update(scope_bindings)
    exec(compile(tree, filename, "exec"), scope)  # noqa: S102
    recompiled = functools.wraps(func, assigned=WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS)(
        scope[func.__name__]  # ty: ignore[unresolved-attribute]
    )
    args_name_mapper = dict(
        zip(
            _args_from_func_ast(tree),
            list(inspect.signature(func).parameters),
            strict=False,
        )
    )
    return rename_arguments(recompiled, mapper=args_name_mapper)


def not_to_call(node: ast.UnaryOp, module: str) -> ast.Call:
    """Transform negation operation to Call."""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=module, ctx=ast.Load()),
            attr="logical_not",
            ctx=ast.Load(),
        ),
        args=[node.operand],
        keywords=[],
    )


def boolop_to_call(node: ast.BoolOp, module: str) -> ast.Call:
    """Transform BoolOp operation to Call."""
    _boolop_registry: dict[type[ast.boolop], str] = {
        ast.And: "logical_and",
        ast.Or: "logical_or",
    }
    operation = _boolop_registry[type(node.op)]

    def _constructor(left: ast.Call | ast.expr, right: ast.Call | ast.expr) -> ast.Call:
        """Construct calls of the form `module.logical_(and|or)(left, right)`."""
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=module, ctx=ast.Load()),
                attr=operation,
                ctx=ast.Load(),
            ),
            args=[left, right],
            keywords=[],
        )

    values: list[ast.Call | ast.expr] = [
        boolop_to_call(v, module=module) if isinstance(v, ast.BoolOp) else v
        for v in node.values
    ]

    return cast("ast.Call", functools.reduce(_constructor, values))


def func_to_ast(func: Callable[..., Any]) -> ast.Module:
    """Parse ``func``'s source into an AST, stripping its decorator lines."""
    source = inspect.getsource(func)
    source_dedented = textwrap.dedent(source)
    source_without_decorators = _remove_decorator_lines(source_dedented)
    return ast.parse(source_without_decorators)


def is_lambda_function(obj: object) -> bool:
    """Whether ``obj`` is a lambda, which has no rewritable named-function source."""
    return isinstance(obj, types.FunctionType) and obj.__name__ == "<lambda>"


def _args_from_func_ast(func_ast: ast.Module) -> list[str]:
    """Get function arguments from function ast."""
    return [arg.arg for arg in func_ast.body[0].args.args]  # ty: ignore[unresolved-attribute]


def _remove_decorator_lines(source: str) -> str:
    """Removes leading decorator lines from function source code."""
    if source.startswith("def "):
        return source
    return "def " + source.split("\ndef ")[1]
