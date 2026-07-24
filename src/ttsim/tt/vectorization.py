from __future__ import annotations

import ast
import functools
from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy

from ttsim.exceptions import TTSIMError
from ttsim.tt._source_rewriting import (
    WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS,
    boolop_to_call,
    func_to_ast,
    is_lambda_function,
    not_to_call,
    recompile_from_ast,
)
from ttsim.tt.type_resolution import (
    build_beartype_checkable_wrapper,
    create_vectorized_annotations,
)

if TYPE_CHECKING:
    from types import ModuleType


BACKEND_TO_MODULE = {"jax": "jax.numpy", "numpy": "numpy"}


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
            *WRAPPER_ASSIGNMENTS_NO_ANNOTATIONS,
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
    if is_lambda_function(func):
        raise TranslateToVectorizableError(
            "Lambda functions are not supported for vectorization. Please define a "
            "named function and use that.",
        )

    module = _module_from_backend(backend)
    tree = _make_vectorizable_ast(func, module=module, xnp=xnp)
    return recompile_from_ast(
        func=func,
        tree=tree,
        scope_bindings={module: import_module(module)},
        filename="<ast>",
    )


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
    if is_lambda_function(func):
        raise TranslateToVectorizableError(
            "Lambda functions are not supported for vectorization. Please define a "
            "named function and use that.",
        )

    module = _module_from_backend(backend)
    tree = _make_vectorizable_ast(func, module=module, xnp=xnp)
    return ast.unparse(tree)


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
    tree = func_to_ast(func)

    # get function location for error messages
    func_loc = f"{func.__module__}/{func.__name__}"  # ty: ignore[unresolved-attribute]

    # transform tree nodes
    new_tree = Transformer(module=module, func_loc=func_loc, xnp=xnp).visit(tree)
    return ast.fix_missing_locations(new_tree)


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
            return not_to_call(node, module=self.module)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:
        self.generic_visit(node)
        return boolop_to_call(node, module=self.module)

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
