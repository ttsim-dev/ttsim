from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import types
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy
from dags.signature import rename_arguments

if TYPE_CHECKING:
    from collections.abc import Callable
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

    Returns
    -------
        New PolicyFunction with the function attribute vectorized.

    Raises
    ------
        ValueError: If the vectorization strategy is not supported.
        TranslateToVectorizableError: If the function cannot be vectorized.

    """

    vectorized: Callable[..., Any]
    if vectorization_strategy == "loop":
        assigned = (
            "__signature__",
            "__globals__",
            "__closure__",
            *functools.WRAPPER_ASSIGNMENTS,
        )
        vectorized = functools.wraps(func, assigned=assigned)(numpy.vectorize(func))
    elif vectorization_strategy == "vectorize":
        vectorized = _make_vectorizable(func, backend=backend, xnp=xnp)
    else:
        raise ValueError(
            f"Vectorization strategy {vectorization_strategy} is not supported. "
            "Use 'loop' or 'vectorize'.",
        )

    # Update annotations and signature to reflect that the inputs are now expected to be
    # arrays.
    vectorized.__signature__ = _create_vectorized_signature(func)  # type: ignore[attr-defined]
    vectorized.__annotations__ = _create_vectorized_annotations(func)

    return vectorized


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

    Returns
    -------
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
    scope = dict(func.__globals__)
    if func.__closure__:
        closure_vars = func.__code__.co_freevars
        closure_cells = [c.cell_contents for c in func.__closure__]
        scope.update(dict(zip(closure_vars, closure_cells, strict=False)))

    scope[module] = import_module(module)

    # execute new ast
    compiled = compile(tree, "<ast>", "exec")
    exec(compiled, scope)  # noqa: S102

    # assign created function
    new_func = scope[func.__name__]
    _vectorized = functools.wraps(func)(new_func)

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

    Returns
    -------
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


def _make_vectorizable_ast(
    func: Callable[..., Any],
    module: str,
    xnp: ModuleType,
) -> ast.Module:
    """Change if statement to where call in the ast of func and return new ast.

    Args:
        func: Function.
        module: Module which exports the function `where` that behaves as `numpy.where`.

    Returns
    -------
        AST of new function with altered ast.
    """
    tree = _func_to_ast(func)

    # get function location for error messages
    func_loc = f"{func.__module__}/{func.__name__}"

    # transform tree nodes
    new_tree = Transformer(module, func_loc, xnp).visit(tree)
    return ast.fix_missing_locations(new_tree)


def _func_to_ast(func: Callable[..., Any]) -> ast.Module:
    source = inspect.getsource(func)
    source_dedented = textwrap.dedent(source)
    source_without_decorators = _remove_decorator_lines(source_dedented)
    return ast.parse(source_without_decorators)


def _args_from_func_ast(func_ast: ast.Module) -> list[str]:
    """Get function arguments from function ast."""
    return [arg.arg for arg in func_ast.body[0].args.args]  # type: ignore[attr-defined]


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
    args = [node.test, node.body[0].value]  # type: ignore[attr-defined]

    if len(node.orelse) > 1 or len(node.body) > 1:
        msg = _too_many_operations_error_message(node, func_loc=func_loc)
        raise TranslateToVectorizableError(msg)
    if node.orelse == []:
        if isinstance(node.body[0], ast.Return):
            msg = _return_and_no_else_error_message(node.body[0], func_loc=func_loc)
            raise TranslateToVectorizableError(msg)
        if hasattr(node.body[0], "targets"):
            name = ast.Name(id=node.body[0].targets[0].id, ctx=ast.Load())
        else:
            name = ast.Name(id=node.body[0].target.id, ctx=ast.Load())  # type: ignore[attr-defined]
        args.append(name)
    elif isinstance(node.orelse[0], ast.Return):
        args.append(node.orelse[0].value)
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
    operation = {ast.And: "logical_and", ast.Or: "logical_or"}[type(node.op)]

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

    transform_node = hasattr(node.func, "id") and node.func.id in to_transform

    if not transform_node:
        return node

    func_id = node.func.id  # type: ignore[attr-defined]
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


class TranslateToVectorizableError(ValueError):
    """Error when function cannot be translated into vectorizable compatible format."""


def _too_many_arguments_call_error_message(node: ast.Call, func_loc: str) -> str:
    source = _node_to_formatted_source(node)
    _func_name = node.func.id  # type: ignore[attr-defined]
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
    if backend in BACKEND_TO_MODULE:
        return BACKEND_TO_MODULE[backend]

    raise NotImplementedError(
        f"Argument 'backend' is {backend} but must be in {BACKEND_TO_MODULE.keys()}.",
    )


# ======================================================================================
# Signature and annotations
# ======================================================================================


def _create_vectorized_signature(func: Callable[..., Any]) -> inspect.Signature:
    """Create a signature for the vectorized function."""
    parameters = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=scalar_type_to_array_type(param.annotation),
        )
        for param in inspect.signature(func).parameters.values()
    ]
    return_annotation = scalar_type_to_array_type(
        inspect.signature(func).return_annotation
    )
    return inspect.Signature(parameters=parameters, return_annotation=return_annotation)


def _create_vectorized_annotations(func: Callable[..., Any]) -> dict[str, Any]:
    """Create annotations for the vectorized function."""
    parameters_and_return = ["return", *inspect.signature(func).parameters]
    annotations = inspect.get_annotations(func)
    return {
        name: scalar_type_to_array_type(
            # If no annotation is available, we assume it is a numerical scalar type,
            # which is converted to an array type.
            annotations.get(name, "IntColumn | FloatColumn | BoolColumn"),
        )
        for name in parameters_and_return
    }


def scalar_type_to_array_type(orig_type: Literal["int", "float", "bool"]) -> str:
    """Convert a scalar type to the corresponding array type."""
    registry = {
        "int": "IntColumn",
        "float": "FloatColumn",
        "bool": "BoolColumn",
    }
    if orig_type in registry:
        return registry[orig_type]
    return orig_type
