import ast
import functools
import inspect
import textwrap
import types
from collections.abc import Callable
from importlib import import_module
from typing import Literal, cast

import numpy

from ttsim.config import IS_JAX_INSTALLED
from ttsim.config import numpy_or_jax as np

BACKEND_TO_MODULE = {"jax": "jax.numpy", "numpy": "numpy"}


def vectorize_function(
    func: Callable, vectorization_strategy: Literal["loop", "vectorize"]
) -> Callable:
    vectorized: Callable
    if vectorization_strategy == "loop":
        vectorized = functools.wraps(func)(numpy.vectorize(func))
        vectorized.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
        vectorized.__globals__ = func.__globals__  # type: ignore[attr-defined]
        vectorized.__closure__ = func.__closure__  # type: ignore[attr-defined]
    elif vectorization_strategy == "vectorize":
        backend = "jax" if IS_JAX_INSTALLED else "numpy"
        vectorized = _make_vectorizable(func, backend=backend)
    else:
        raise ValueError(
            f"Vectorization strategy {vectorization_strategy} is not supported. "
            "Use 'loop' or 'vectorize'."
        )
    return vectorized


def _make_vectorizable(func: Callable, backend: str) -> Callable:
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
            "named function and use that."
        )

    module = _module_from_backend(backend)
    tree = _make_vectorizable_ast(func, module=module)

    # recreate scope of function, add array library
    scope = dict(func.__globals__)
    if func.__closure__:
        closure_vars = func.__code__.co_freevars
        closure_cells = [c.cell_contents for c in func.__closure__]
        scope.update(dict(zip(closure_vars, closure_cells)))

    scope[module] = import_module(module)

    # execute new ast
    compiled = compile(tree, "<ast>", "exec")
    exec(compiled, scope)  # noqa: S102

    # assign created function
    new_func = scope[func.__name__]
    return functools.wraps(func)(new_func)


def make_vectorizable_source(func: Callable, backend: str) -> str:
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
            "named function and use that."
        )

    module = _module_from_backend(backend)
    tree = _make_vectorizable_ast(func, module=module)
    return ast.unparse(tree)


def _make_vectorizable_ast(func: Callable, module: str) -> ast.Module:
    """Change if statement to where call in the ast of func and return new ast.

    Args:
        func: Function.
        module: Module which exports the function `where` that behaves as `numpy.where`.

    Returns:
        AST of new function with altered ast.
    """
    tree = _func_to_ast(func)

    # get function location for error messages
    func_loc = f"{func.__module__}/{func.__name__}"

    # transform tree nodes
    new_tree = Transformer(module, func_loc).visit(tree)
    return ast.fix_missing_locations(new_tree)


def _func_to_ast(func: Callable) -> ast.Module:
    source = inspect.getsource(func)
    source_dedented = textwrap.dedent(source)
    source_without_decorators = _remove_decorator_lines(source_dedented)
    return ast.parse(source_without_decorators)


def _remove_decorator_lines(source: str) -> str:
    """Removes leading decorator lines from function source code."""
    if source.startswith("def "):
        return source
    else:
        return "def " + source.split("\ndef ")[1]


# ======================================================================================
# Transformation class
# ======================================================================================


class Transformer(ast.NodeTransformer):
    def __init__(self, module: str, func_loc: str) -> None:
        self.module = module
        self.func_loc = func_loc

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        return _call_to_call_from_module(
            node, module=self.module, func_loc=self.func_loc
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.UnaryOp | ast.Call:  # noqa: N802
        if isinstance(node.op, ast.Not):
            return _not_to_call(node, module=self.module)
        else:
            return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.Call:  # noqa: N802
        self.generic_visit(node)
        return _boolop_to_call(node, module=self.module)

    def visit_If(  # noqa: N802
        self, node: ast.If
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

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:  # noqa: N802
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
    elif node.orelse == []:
        if isinstance(node.body[0], ast.Return):
            msg = _return_and_no_else_error_message(node.body[0], func_loc=func_loc)
            raise TranslateToVectorizableError(msg)
        elif hasattr(node.body[0], "targets"):
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
            value=ast.Name(id=module, ctx=ast.Load()), attr="where", ctx=ast.Load()
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
            value=ast.Name(id=module, ctx=ast.Load()), attr="where", ctx=ast.Load()
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


def _call_to_call_from_module(node: ast.Call, module: str, func_loc: str) -> ast.AST:
    """Transform built-in Calls to Calls from module."""
    to_transform = ("sum", "any", "all", "max", "min")

    transform_node = hasattr(node.func, "id") and node.func.id in to_transform

    if not transform_node:
        return node

    func_id = node.func.id  # type: ignore[attr-defined]
    call = node
    args = node.args

    if len(args) == 1:
        if type(args) not in (list, tuple, np.ndarray):
            raise TranslateToVectorizableError(
                f"Argument of function {func_id} is not a list, tuple, or valid array."
                f"\n\nFunction: {func_loc}\n\n"
                f"Problematic source code: \n\n{_node_to_formatted_source(node)}\n"
            )

        call.func = ast.Attribute(
            value=ast.Name(id=module, ctx=ast.Load()),
            attr=func_id,
            ctx=ast.Load(),
        )
    elif func_id in ("max", "min") and len(args) == 2:
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
        f"Argument 'backend' is {backend} but must be in {BACKEND_TO_MODULE.keys()}."
    )
