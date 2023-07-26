import inspect
from functools import wraps


def assign_init_args(func):
    """Class initialization decorator that automatically assigns the
    class initialization arguments to the class instance without the need to
    explicitl assignment in the initialization function.

    Note that the constructor of parent classes will be executed AFTER the
    assignment of the arguments. This might raise errors for classes that
    demands initialization before any assignment, such as the children
    classes of `torch.nn.Module`.

    Args:
        func: A class initialization function.

    References:
        https://stackoverflow.com/a/1389216

    Examples:
        >>> class Foo:
        ...     @assign_init_args
        ...     def __init__(self, a, b=True, c="pass"):
        ...         pass
        >>> p = Foo("a", True)
        >>> # noinspection PyUnresolvedReferences
        ... p.a, p.b, p.c
        ("a", True, "pass")

    Returns
        A class initialization decorator.

    """

    # names, varargs, keywords, defaults, _, _, _
    names, _, _, defaults, _, _, _ = inspect.getfullargspec(func)

    @wraps(func)
    def _assign_init_args_wrapper(self, *args, **kwargs):
        # noinspection PyTypeChecker
        for __name, __arg in list(zip(names[1:], args)) + list(kwargs.items()):
            setattr(self, __name, __arg)

        if defaults is not None:
            for __name, __default_arg in zip(
                reversed(names), reversed(defaults)
            ):
                if not hasattr(self, __name):
                    setattr(self, __name, __default_arg)

        func(self, *args, **kwargs)

    return _assign_init_args_wrapper
