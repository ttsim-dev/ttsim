from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import types
from collections.abc import Callable, Mapping
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy
from dags.signature import rename_arguments

from ttsim.exceptions import TTSIMError
from ttsim.tt.type_resolution import (
    build_beartype_checkable_wrapper,
    create_vectorized_annotations,
)

if TYPE_CHECKING:
    from types import ModuleType


BACKEND_TO_MODULE = {"jax": "jax.numpy", "numpy": "numpy"}


# `functools.WRAPPER_ASSIGNMENTS` minus the annotation attributes. Used at
# every `functools.wraps` site that wraps a user policy function: if we let
# the user's scalar annotations leak onto the column-typed wrapper,
# beartype rejects the wrapper's column-typed arguments against the
# wrapper's inherited scalar signature.
#
# `__annotate__` is the PEP 649 (Python 3.14+) deferred-evaluation pair to
# `__annotations__` and needs the same treatment.
_WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS: tuple[str, ...] = tuple(
    a
    for a in functools.WRAPPER_ASSIGNMENTS
    if a not in ("__annotations__", "__annotate__")
)


def vectorize_function(
    func: Callable[..., Any],
    vectorization_strategy: Literal["loop", "vectorize"],
    backend: Literal["numpy", "jax"],
    xnp: ModuleType,
) -> Callable[..., Any]:
    """Returns a new PolicyFunction with the function attribute vectorized.

    Args:
        policy_function: PolicyFunction to vectorize.
        vectorization_strategy: Strategy to use for vectorization.
        backend: Backend to use for vectorization.
        xnp: Module to use for vectorization.

    Returns:
        New PolicyFunction with the function attribute vectorized.

    Raises:
        ValueError: If the vectorization strategy is not supported.
        TranslateToVectorizableError: If the function cannot be vectorized.

    """

    vectorized: Callable[..., Any]
    if vectorization_strategy == "loop":
        assigned = (
            "__signature__",
            "__globals__",
            "__closure__",
            *_WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS,
        )
        vectorized = functools.wraps(func, assigned=assigned)(numpy.vectorize(func))
    elif vectorization_strategy == "vectorize":
        vectorized = _make_vectorizable(func, backend=backend, xnp=xnp)
    else:
        raise ValueError(
            f"Vectorization strategy {vectorization_strategy} is not supported. "
            "Use 'loop' or 'vectorize'.",
        )

    # Wrap the vectorized callable in a typed forwarder whose parameters and
    # return are annotated with the concrete column-type aliases resolved via
    # `ttsim.tt.type_resolution`. The forwarder advertises an honest producer
    # type to the DAG's annotation-consistency check and — being a real-
    # parameter, non-isomorphic, non-nested function defined against
    # `ttsim.typing` — is itself directly `@beartype`-decorable.
    return build_beartype_checkable_wrapper(
        vectorized,
        annotations=create_vectorized_annotations(func),
        node_name=getattr(func, "__name__", "<vectorized node>"),
    )


def _make_vectorizable(
    func: Callable[..., Any],
    backend: str,
    xnp: ModuleType,
) -> Callable[..., Any]:
    """Redefine function to be vectorizable given backend.

    Args:
        func: Function.
        backend: Backend library. Currently supported backends are 'jax' and 'numpy'.
            Array module must export function `where` that behaves as `numpy.where`.

    Returns:
        New function with altered ast.
    """
    if _is_lambda_function(func):
        raise TranslateToVectorizableError(
            "Lambda functions are not supported for vectorization. Please define a "
            "named function and use that.",
        )

    module = _module_from_backend(backend)
    tree = _make_vectorizable_ast(func, module=module, xnp=xnp)

    # recreate scope of function, add array library
    scope = dict(func.__globals__)  # ty: ignore[unresolved-attribute]
    if func.__closure__:  # ty: ignore[unresolved-attribute]
        closure_vars = func.__code__.co_freevars  # ty: ignore[unresolved-attribute]
        closure_cells = [c.cell_contents for c in func.__closure__]  # ty: ignore[unresolved-attribute]
        scope.update(dict(zip(closure_vars, closure_cells, strict=False)))

    scope[module] = import_module(module)

    # execute new ast
    compiled = compile(tree, "<ast>", "exec")
    exec(compiled, scope)  # noqa: S102

    # assign created function
    new_func = scope[func.__name__]  # ty: ignore[unresolved-attribute]
    _vectorized = functools.wraps(func, assigned=_WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS)(
        new_func
    )

    # For functions whose argument names are renamed dynamically, we need to match the
    # argument names, since the vectorization works on the AST level, which is not
    # affected by the original renaming. This assumes that the argument ordering is
    # the same in the function and its AST.
    _original_args = _args_from_func_ast(_func_to_ast(func))
    _args_name_mapper = dict(
        zip(
            _original_args,
            list(inspect.signature(func).parameters),
            strict=False,
        )
    )
    return rename_arguments(_vectorized, mapper=_args_name_mapper)


def make_vectorizable_source(
    func: Callable[..., Any],
    backend: str,
    xnp: ModuleType,
) -> str:
    """Redefine function source to be vectorizable given backend.

    Args:
        func: Function.
        backend: Backend library. See dict `BACKEND_TO_MODULE` for currently supported
            backends. Array module must export function `where` that behaves as
            `numpy.where`.

    Returns:
        Source code of new function with altered ast.
    """
    if _is_lambda_function(func):
        raise TranslateToVectorizableError(
            "Lambda functions are not supported for vectorization. Please define a "
            "named function and use that.",
        )

    module = _module_from_backend(backend)
    tree = _make_vectorizable_ast(func, module=module, xnp=xnp)
    return ast.unparse(tree)


def recompile_with_logical_ops_as_calls(
    func: Callable[..., Any],
    module: str,
    module_obj: Any,  # noqa: ANN401
    extra_globals: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    """Return a copy of ``func`` with ``and``/``or``/``not`` as ``{module}.logical_*``
    calls.

    Python ``and``/``or`` short-circuit through ``__bool__`` and yield one operand
    whole, and ``not`` consumes ``__bool__`` and returns a plain ``bool``, so none
    of them can combine or preserve a custom object. The unit check reuses
    the array vectorizer's :func:`_boolop_to_call` / :func:`_not_to_call` rewrites,
    binding ``module`` to ``module_obj`` (an ``xnp`` stand-in whose ``logical_*`` route
    through the leveled-boolean combine) so author-written ``and``/``or``/``not``
    are checked the way they run — a ``not`` on a leveled boolean keeps its level,
    exactly as ``~`` does. The numeric runtime is untouched.

    ``extra_globals`` rebinds module-level names in the recompiled body's scope —
    the unit check uses it to swap ``piecewise_polynomial``/``join`` for unit-only
    stand-ins, so a body that calls them is checked rather than executed. When it is
    given, the body is rebound even if it has no ``and``/``or`` (the rebinding,
    not the rewrite, is then the point).

    A function with no ``and``/``or`` and no ``extra_globals`` is returned
    unchanged. Falls back to the original when source is unavailable (a builtin, a
    C function, a REPL definition) or unparseable, so the unit check sees the original
    body.
    """
    if _is_lambda_function(func):
        return func
    try:
        tree = _func_to_ast(func)
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
                return _boolop_to_call(node=node, module=module)

            def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp | ast.Call:
                self.generic_visit(node)
                if isinstance(node.op, ast.Not):
                    return _not_to_call(node=node, module=module)
                return node

        _LogicalOpRewriter().visit(tree)
        ast.fix_missing_locations(tree)
    scope = dict(func.__globals__)  # ty: ignore[unresolved-attribute]
    if func.__closure__:  # ty: ignore[unresolved-attribute]
        closure_vars = func.__code__.co_freevars  # ty: ignore[unresolved-attribute]
        closure_cells = [c.cell_contents for c in func.__closure__]  # ty: ignore[unresolved-attribute]
        scope.update(dict(zip(closure_vars, closure_cells, strict=False)))
    scope[module] = module_obj
    if extra_globals:
        scope.update(extra_globals)
    exec(compile(tree, "<unit-check-logical-ops>", "exec"), scope)  # noqa: S102
    rewritten = functools.wraps(func, assigned=_WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS)(
        scope[func.__name__]  # ty: ignore[unresolved-attribute]
    )
    # The AST carries the original argument names; match any renamed dynamically
    # after definition, exactly as `_make_vectorizable` does. The rewrite only
    # touches `BoolOp` nodes, so `tree`'s argument list is the original one.
    args_name_mapper = dict(
        zip(
            _args_from_func_ast(tree),
            list(inspect.signature(func).parameters),
            strict=False,
        )
    )
    return rename_arguments(rewritten, mapper=args_name_mapper)


def _make_vectorizable_ast(
    func: Callable[..., Any],
    module: str,
    xnp: ModuleType,
) -> ast.Module:
    """Change if statement to where call in the ast of func and return new ast.

    Args:
        func: Function.
        module: Module which exports the function `where` that behaves as `numpy.where`.

    Returns:
        AST of new function with altered ast.
    """
    tree = _func_to_ast(func)

    # get function location for error messages
    func_loc = f"{func.__module__}/{func.__name__}"  # ty: ignore[unresolved-attribute]

    # transform tree nodes
    new_tree = Transformer(module=module, func_loc=func_loc, xnp=xnp).visit(tree)
    return ast.fix_missing_locations(new_tree)


def _func_to_ast(func: Callable[..., Any]) -> ast.Module:
    source = inspect.getsource(func)
    source_dedented = textwrap.dedent(source)
    source_without_decorators = _remove_decorator_lines(source_dedented)
    return ast.parse(source_without_decorators)


def _args_from_func_ast(func_ast: ast.Module) -> list[str]:
    """Get function arguments from function ast."""
    return [arg.arg for arg in func_ast.body[0].args.args]  # ty: ignore[unresolved-attribute]


def _remove_decorator_lines(source: str) -> str:
    """Removes leading decorator lines from function source code."""
    if source.startswith("def "):
        return source
    return "def " + source.split("\ndef ")[1]


# ======================================================================================
# Transformation class
# ======================================================================================


class Transformer(ast.NodeTransformer):
    def __init__(self, module: str, func_loc: str, xnp: ModuleType) -> None:
        self.module = module
        self.func_loc = func_loc
        self.xnp = xnp

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Forbid type-conversion calls
        forbidden_type_conversions = {"float", "int", "bool", "complex", "str"}
        if hasattr(node.func, "id") and node.func.id in forbidden_type_conversions:
            msg = (
                f"Forbidden type conversion '{node.func.id}' detected in function. "
                f"Type conversions like float(), int(), bool(), complex(), str() are "
                f"not allowed in vectorized functions.\n\nFunction: {self.func_loc}\n\n"
                f"Problematic source code: \n\n{_node_to_formatted_source(node)}\n"
            )
            raise TranslateToVectorizableError(msg)
        self.generic_visit(node)
        return _call_to_call_from_module(
            node,
            module=self.module,
            func_loc=self.func_loc,
            xnp=self.xnp,
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        # Forbid any augmented assignment (+=, -=, *=, /=, etc.)
        msg = (
            "Augmented assignment is not allowed in vectorized functions. "
            "Operations like +=, -=, *=, /=, etc. are forbidden.\n\n"
            f"Function: {self.func_loc}\n\n"
            f"Problematic source code: \n\n{_node_to_formatted_source(node)}\n"
        )
        raise TranslateToVectorizableError(msg)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp | ast.Call:
        if isinstance(node.op, ast.Not):
            return _not_to_call(node, module=self.module)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        self.generic_visit(node)
        return _boolop_to_call(node, module=self.module)

    def visit_If(
        self,
        node: ast.If,
    ) -> ast.Call | ast.Return | ast.Assign | ast.AugAssign:
        self.generic_visit(node)
        call = _if_to_call(node, module=self.module, func_loc=self.func_loc)
        out: ast.Call | ast.Return | ast.Assign | ast.AugAssign
        if isinstance(node.body[0], ast.Return):
            out = ast.Return(call)
        elif isinstance(node.body[0], (ast.Assign, ast.AugAssign)):
            out = node.body[0]
            out.value = call
        else:
            out = call
        return out

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        return _ifexp_to_call(node, module=self.module)


# ======================================================================================
# Transformation functions on node level
# ======================================================================================


def _not_to_call(node: ast.UnaryOp, module: str) -> ast.Call:
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


def _if_to_call(node: ast.If, module: str, func_loc: str) -> ast.Call:
    """Transform If statement to Call."""
    args: list[ast.expr] = [node.test, node.body[0].value]  # ty: ignore[unresolved-attribute]

    if len(node.orelse) > 1 or len(node.body) > 1:
        msg = _too_many_operations_error_message(node, func_loc=func_loc)
        raise TranslateToVectorizableError(msg)
    if node.orelse == []:
        if isinstance(node.body[0], ast.Return):
            msg = _return_and_no_else_error_message(node.body[0], func_loc=func_loc)
            raise TranslateToVectorizableError(msg)
        if hasattr(node.body[0], "targets"):
            name = ast.Name(id=node.body[0].targets[0].id, ctx=ast.Load())  # ty: ignore[not-subscriptable]
        else:
            name = ast.Name(id=node.body[0].target.id, ctx=ast.Load())  # ty: ignore[unresolved-attribute]
        args.append(name)
    elif isinstance(node.orelse[0], ast.Return):
        args.append(cast("ast.expr", node.orelse[0].value))
    elif isinstance(node.orelse[0], ast.If):
        call_if = _if_to_call(node.orelse[0], module=module, func_loc=func_loc)
        args.append(call_if)
    elif isinstance(node.orelse[0], (ast.Assign, ast.AugAssign)):
        if isinstance(node.orelse[0].value, ast.IfExp):
            call_ifexp = _ifexp_to_call(node.orelse[0].value, module=module)
            args.append(call_ifexp)
        else:
            args.append(node.orelse[0].value)
    else:
        msg = _disallowed_operation_error_message(node.orelse[0], func_loc=func_loc)
        raise TranslateToVectorizableError(msg)

    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=module, ctx=ast.Load()),
            attr="where",
            ctx=ast.Load(),
        ),
        args=args,
        keywords=[],
    )


def _ifexp_to_call(node: ast.IfExp, module: str) -> ast.Call:
    """Transform IfExp expression to Call."""
    args = [node.test, node.body]

    if isinstance(node.orelse, ast.IfExp):
        call_ifexp = _ifexp_to_call(node.orelse, module=module)
        args.append(call_ifexp)
    else:
        args.append(node.orelse)

    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=module, ctx=ast.Load()),
            attr="where",
            ctx=ast.Load(),
        ),
        args=args,
        keywords=[],
    )


def _boolop_to_call(node: ast.BoolOp, module: str) -> ast.Call:
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
        _boolop_to_call(v, module=module) if isinstance(v, ast.BoolOp) else v
        for v in node.values
    ]

    return cast("ast.Call", functools.reduce(_constructor, values))


def _call_to_call_from_module(
    node: ast.Call,
    module: str,
    func_loc: str,
    xnp: ModuleType,
) -> ast.AST:
    """Transform built-in Calls to Calls from module."""
    to_transform = ("sum", "any", "all", "max", "min")

    if not isinstance(node.func, ast.Name) or node.func.id not in to_transform:
        return node

    func_id = node.func.id
    call = node
    args = node.args

    if len(args) == 1:
        if type(args) not in (list, tuple, xnp.ndarray):
            raise TranslateToVectorizableError(
                f"Argument of function {func_id} is not a list, tuple, or valid array."
                f"\n\nFunction: {func_loc}\n\n"
                f"Problematic source code: \n\n{_node_to_formatted_source(node)}\n",
            )

        call.func = ast.Attribute(
            value=ast.Name(id=module, ctx=ast.Load()),
            attr=func_id,
            ctx=ast.Load(),
        )
    elif func_id in ("max", "min") and len(args) == 2:  # noqa: PLR2004
        attr = func_id + "imum"  # max -> maximum, min -> minimum
        call.func = ast.Attribute(
            value=ast.Name(id=module, ctx=ast.Load()),
            attr=attr,
            ctx=ast.Load(),
        )
    else:
        msg = _too_many_arguments_call_error_message(node, func_loc=func_loc)
        raise TranslateToVectorizableError(msg)

    return call


# ======================================================================================
# Transformation errors and checks
# ======================================================================================


def _is_lambda_function(obj: object) -> bool:
    return isinstance(obj, types.FunctionType) and obj.__name__ == "<lambda>"


class TranslateToVectorizableError(TTSIMError, ValueError):
    """Error when function cannot be translated into vectorizable compatible format."""


def _too_many_arguments_call_error_message(node: ast.Call, func_loc: str) -> str:
    source = _node_to_formatted_source(node)
    _func_name = node.func.id  # ty: ignore[unresolved-attribute]
    return (
        "\n\n"
        f"The function {_func_name} is called with too many arguments. Please only use "
        "one iterable argument for (`sum`, `any`, `all`, `max`, `min`) or two "
        "arguments for (`max`, `min`)."
        f"\n\nFunction: {func_loc}\n\n"
        "Problematic source code (after transformations that were possible, if any):"
        f"\n\n{source}\n"
    )


def _return_and_no_else_error_message(node: ast.Return, func_loc: str) -> str:
    source = _node_to_formatted_source(node)
    return (
        "\n\n"
        "The if-clause body is a return statement, while the else clause is missing.\n"
        "Please swap the return statement for an assignment or add an else-clause."
        f"\n\nFunction: {func_loc}\n\n"
        "Problematic source code (after transformations that were possible, if any):"
        f"\n\n{source}\n"
    )


def _too_many_operations_error_message(node: ast.If, func_loc: str) -> str:
    source = _node_to_formatted_source(node)
    return (
        "\n\n"
        "An if statement is performing multiple operations, which is forbidden.\n"
        "Please only perform one operation in the body of an if-elif-else statement."
        f"\n\nFunction: {func_loc}\n\n"
        "Problematic source code (after transformations that were possible, if any):"
        f"\n\n{source}\n"
    )


def _disallowed_operation_error_message(node: ast.AST, func_loc: str) -> str:
    source = _node_to_formatted_source(node)
    return (
        "\n\n"
        f"An if-elif-else clause body is of type {type(node)}, which is forbidden.\n"
        "Allowed types are the following:\n\n"
        "ast.If : Another if-else-elif clause\n"
        "ast.IfExp : A one-line if-else statement. Example: 1 if flag else 0\n"
        "ast.Assign : An assignment. Example: x = 3\n"
        "ast.Return : A return statement. Example: return out"
        f"\n\nFunction: {func_loc}\n\n"
        "Problematic source code (after transformations that were possible, if any):"
        f"\n\n{source}\n"
    )


def _node_to_formatted_source(node: ast.AST) -> str:
    source = ast.unparse(node)
    return " > " + source[:-1].replace("\n", "\n > ")


def _module_from_backend(backend: str) -> str:
    try:
        return BACKEND_TO_MODULE[backend]
    except KeyError:
        msg = f"Argument 'backend' is {backend!r}, must be in {set(BACKEND_TO_MODULE)}."
        raise NotImplementedError(msg) from None
