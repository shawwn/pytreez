from __future__ import annotations

__version__ = '1.0.1'

from enum import Enum
from dataclasses import dataclass
import typing as T
import builtins as py
import operator as op
import collections
import functools
from functools import partial
from copy import deepcopy

_NoneType = type(None)

class PyTreeKind(Enum):
    kLeaf = "leaf"
    kNone = "None"
    kTuple = "tuple"
    kNamedTuple = "collections.namedtuple"
    kList = "list"
    kDict = "dict"
    kCustom = "custom"


class PyTreeTypeRegistry:
    def __init__(self):
        self.registrations_ = {}
        def add_builtin_type(type_obj, kind: PyTreeKind):
            self.registrations_[type_obj] = self.Registration(kind=kind, type=type_obj)
        add_builtin_type(type(None), PyTreeKind.kNone)
        add_builtin_type(tuple, PyTreeKind.kTuple)
        add_builtin_type(list, PyTreeKind.kList)
        add_builtin_type(dict, PyTreeKind.kDict)

    @dataclass
    class Registration:
        kind: PyTreeKind

        # The following values are populated for custom types.
        # The Python type object, used to identify the type.
        type: T.Any # pybind11::object type;

        # A function with signature: object -> (iterable, aux_data)
        to_iterable: T.Callable = None # pybind11::function to_iterable;

        # A function with signature: (aux_data, iterable) -> object
        from_iterable: T.Callable = None # pybind11::function from_iterable;

        def __post_init__(self):
            if not isinstance(self.kind, PyTreeKind):
                for entry in PyTreeKind:
                    if self.kind == entry.value:
                        self.kind = entry
                        break
            if not isinstance(self.kind, PyTreeKind):
                raise ValueError(f"Expected kind to be PyTreeKind, got {self.kind!r}")

        def __eq__(self, other: PyTreeTypeRegistry.Registration):
            if self.kind.value != other.kind.value:
                return False
            if self.type != other.type:
                return False
            return True

    @classmethod
    def singleton(cls) -> PyTreeTypeRegistry:
        try:
            return cls.inst
        except AttributeError:
            cls.inst = cls()
            return cls.inst

    @classmethod
    def register(cls, type: T.Type, to_iterable: T.Callable, from_iterable: T.Callable):
        self = cls.singleton()
        if type in self.registrations_:
            raise ValueError("Duplicate custom PyTreeDef type registration for %s." % repr(type))
        registration = cls.Registration(PyTreeKind.kCustom, type, to_iterable, from_iterable)
        type._pytree_registration = registration
        self.registrations_[type] = registration

    @classmethod
    def lookup(cls, type: T.Type):
        try:
            return type._pytree_registration
        except AttributeError:
            self = cls.singleton()
            return self.registrations_.get(type)


def is_namedtuple(obj):
    return hasattr(obj, '_fields') and isinstance(obj, tuple)


class PyTreeDef:
    def __init__(self):
        self.traversal_ = []

    def get_kind(self, obj, objtype):
        registration = PyTreeTypeRegistry.lookup(objtype)
        if registration is not None:
            return registration.kind, registration
        elif is_namedtuple(obj):
            # We can only identify namedtuples heuristically, here by the presence of
            # a _fields attribute.
            return PyTreeKind.kNamedTuple, None
        else:
            return PyTreeKind.kLeaf, None


    def flatten_into(self, handle, leaves: T.List[T.Any], nodes: T.List[T.Any], leaf_predicate: T.Callable, start_num_nodes: int, start_num_leaves: int):
        # start_num_nodes = len(self.traversal_)
        # start_num_leaves = len(leaves)
        # assert leaf_predicate is None
        # if leaf_predicate is not None and leaf_predicate(handle):
        #     leaves.append(handle)
        # else:
        objtype = type(handle)
        node_data = None
        node_arity = 0
        num_nodes, num_leaves = start_num_nodes, start_num_leaves
        num_nodes += 1
        if objtype in [py.tuple, py.list, py.dict] or is_namedtuple(handle):
            node_arity = len(handle)
            if objtype is py.dict:
                node_data = list(sorted(handle.keys()))
                for k in node_data:
                    x = handle[k]
                    num_nodes, num_leaves = self.flatten_into(x, leaves, nodes, leaf_predicate, num_nodes, num_leaves)
            else:
                # node_data = handle
                for x in handle:
                    num_nodes, num_leaves = self.flatten_into(x, leaves, nodes, leaf_predicate, num_nodes, num_leaves)
        else:
            objtype = None
            # node_data = handle
            leaves.append(handle)
            num_leaves += 1
        node = (node_arity, num_leaves - start_num_leaves, num_nodes - start_num_nodes, objtype, node_data)
        nodes.append(node)
        return num_nodes, num_leaves

    def unflatten(self, leaves: T.List):
        leaf_count = 0
        for (node_arity, num_leaves, num_nodes, objtype, node_data) in self.traversal_:
            if objtype is None:
                leaf_count += num_leaves
            else:
                span = leaves[leaf_count - node_arity:leaf_count]
                # print(node_arity, objtype, span, node_data)
                if objtype is py.dict:
                    o = objtype(zip(node_data, span))
                elif objtype in [py.tuple, py.list]:
                    o = objtype(span)
                else:
                    o = objtype(*span)
                del leaves[leaf_count - node_arity:leaf_count]
                leaf_count -= node_arity
                leaves.insert(leaf_count, o)
                leaf_count += 1
        return leaves[-1]



def tree_flatten(tree: T.Any, is_leaf: T.Optional[T.Callable[[T.Any], bool]] = None) -> (T.Iterable, PyTreeDef):
    """Flattens a pytree.

    Args:
      tree: a pytree to flatten.
      is_leaf: an optionally specified function that will be called at each
        flattening step. It should return a boolean, which indicates whether
        the flattening should traverse the current object, or if it should be
        stopped immediately, with the whole subtree being treated as a leaf.

    Returns:
      A pair where the first element is a list of leaf values and the second
      element is a treedef representing the structure of the flattened tree.
    """
    leaves = []
    nodes = []
    pytree = PyTreeDef()
    pytree.flatten_into(tree, leaves, nodes, is_leaf, 0, 0)
    return leaves, nodes


def tree_unflatten(treedef: PyTreeDef, leaves: T.Iterable):
  """Reconstructs a pytree from the treedef and the leaves.

  The inverse of `tree_flatten`.

  Args:
    treedef: the treedef to reconstruct
    leaves: the list of leaves to use for reconstruction. The list must
      match the leaves of the treedef.
  Returns:
    The reconstructed pytree, containing the `leaves` placed in the
    structure described by `treedef`.
  """
  if isinstance(treedef, list):
      pytree = PyTreeDef()
      pytree.traversal_ = treedef
      treedef = pytree
  return treedef.unflatten(leaves)
