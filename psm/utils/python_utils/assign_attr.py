from typing import Any, Dict


def assign_attr(
    target: Any,
    attributes: Dict[str, Any],
    omit_private: bool = True,
    overwrite: bool = False,
):
    """Assign attributes to a target object.

    Args:
        target: Target object to assign attributes to.
        attributes: Dictionary of attributes to assign to the target.
        omit_private: Boolean indicator for whether to omit private
            attributes, which are prefixed with an underscore.
        overwrite: Boolean indicator for whether to overwrite existing
            attributes. Will ignore if the attribute already exists.

    """
    _vars = vars(target)
    for __name, __value in attributes.items():
        if omit_private and __name.startswith("_"):
            continue
        if __name in _vars and not overwrite:
            continue
        setattr(target, __name, __value)
