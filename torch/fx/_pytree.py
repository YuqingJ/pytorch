from collections import namedtuple
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Type, Union

from torch.utils._pytree import LeafSpec, PyTree, TreeSpec

FlattenFuncSpec = Callable[[PyTree, TreeSpec], Union[List, Tuple[List, bool]]]

SUPPORTED_NODES: Dict[Type[Any], Any] = {}
def register_pytree_flatten_spec(typ: Any, flatten_fn_spec: FlattenFuncSpec) -> None:
    SUPPORTED_NODES[typ] = flatten_fn_spec

def tree_flatten_spec(pytree: PyTree, spec: TreeSpec, check_children_match=False) -> List[Any]:
    if isinstance(spec, LeafSpec):
        return [pytree]
    if spec.type not in SUPPORTED_NODES:
        raise RuntimeError(
            f"{type(pytree)} does not have a flatten_fn_spec associated with it. Please register one with "
            "torch.fx._pytree.register_pytree_flatten_spec.  If you have serialized your model, make "
            "sure that any custom pytrees have been registered before loading it.")
    flatten_fn_spec = SUPPORTED_NODES[spec.type]
    child_pytrees = flatten_fn_spec(pytree, spec, check_children_match)
    if isinstance(child_pytrees, tuple):
        child_pytrees, children_match = child_pytrees
        if check_children_match and not children_match:
            raise RuntimeError()
    result = []
    for child, child_spec in zip(child_pytrees, spec.children_specs):
        flat = tree_flatten_spec(child, child_spec)
        result += flat
    return result

def _dict_flatten_spec(d: Dict[Any, Any], spec: TreeSpec) -> Tuple[List[Any], bool]:
    return [d[k] for k in spec.context], len(d) == len(spec.context)

def _list_flatten_spec(d: List[Any], spec: TreeSpec) -> Tuple[List[Any], bool]:
    return [d[i] for i in range(len(spec.children_specs))], len(d) == len(spec.children_specs)

def _tuple_flatten_spec(d: Tuple[Any], spec: TreeSpec) -> Tuple[List[Any], bool]:
    return [d[i] for i in range(len(spec.children_specs))], len(d) == len(spec.children_specs)

def _namedtuple_flatten_spec(d: NamedTuple, spec: TreeSpec) -> Tuple[List[Any], bool]:
    return [d[i] for i in range(len(spec.children_specs))], len(d) == len(spec.children_specs)

register_pytree_flatten_spec(dict, _dict_flatten_spec)
register_pytree_flatten_spec(list, _list_flatten_spec)
register_pytree_flatten_spec(tuple, _tuple_flatten_spec)
register_pytree_flatten_spec(namedtuple, _tuple_flatten_spec)
