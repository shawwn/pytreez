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

_NoneType = type(None)


class PyTreeTypeRegistry:
    def __init__(self):
        self.registrations_ = {}

    @dataclass
    class Registration:
        # The following values are populated for custom types.
        # The Python type object, used to identify the type.
        type: T.Any # pybind11::object type;

        # A function with signature: object -> (iterable, aux_data)
        to_iterable: T.Callable = None # pybind11::function to_iterable;

        # A function with signature: (aux_data, iterable) -> object
        from_iterable: T.Callable = None # pybind11::function from_iterable;

        def __eq__(self, other: PyTreeTypeRegistry.Registration):
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
        registration = cls.Registration(type, to_iterable, from_iterable)
        self.registrations_[type] = registration

    @classmethod
    def lookup(cls, type: T.Type) -> PyTreeTypeRegistry.Registration:
        self = cls.singleton()
        return self.registrations_.get(type)


def is_namedtuple(obj, objtype):
    return hasattr(obj, '_fields') and \
            isinstance(obj, tuple)
           # objtype.__bases__ and objtype.__bases__[0] is tuple


class PyTreeDef:
    def __init__(self, nodes = None):
        self.traversal_ = nodes if nodes is not None else []

    def flatten_into(self, handle, leaves: T.List[T.Any], nodes: T.List[T.Any], leaf_predicate: T.Callable, start_num_nodes: int, start_num_leaves: int):
        objtype = type(handle)
        node_data = None
        node_arity = -1
        num_nodes, num_leaves = start_num_nodes + 1, start_num_leaves + 1
        if handle is None:
            pass
        elif objtype in [py.int, py.float, py.bool] \
                or (leaf_predicate and leaf_predicate(handle)):
            leaves.append(handle)
        else:
            reg = PyTreeTypeRegistry.lookup(objtype)
            if reg is not None or is_namedtuple(handle, objtype):
                blades, node_data = reg.to_iterable(handle) if reg else (handle, "namedtuple")
                node_arity = 0
                for x in blades:
                    node_arity += 1
                    num_nodes, num_leaves = self.flatten_into(x, leaves, nodes, leaf_predicate, num_nodes, num_leaves)
                num_leaves -= 1
            else:
                leaves.append(handle)
        node = (node_arity, num_leaves - start_num_leaves, num_nodes - start_num_nodes, objtype, node_data)
        nodes.append(node)
        return num_nodes, num_leaves

    def unflatten(self, leaves: T.List):
        leaves = list(leaves)
        leaf_count = 0
        for (node_arity, num_leaves, num_nodes, objtype, node_data) in self.traversal_:
            if node_arity == -1:
                if objtype is _NoneType:
                    leaves.insert(leaf_count, None)
                leaf_count += num_leaves
            else:
                span = leaves[leaf_count - node_arity:leaf_count]
                reg = PyTreeTypeRegistry.lookup(objtype)
                if reg is not None:
                    o = reg.from_iterable(node_data, span)
                else:
                    o = objtype(*span)
                del leaves[leaf_count - node_arity:leaf_count]
                leaf_count -= node_arity
                leaves.insert(leaf_count, o)
                leaf_count += 1
        return leaves[-1]

    def flatten_up_to(self, xs: typing.Any):
        """Flattens a Pytree up to this PyTreeDef. 'self' must be a tree prefix of
        the tree-structure of 'xs'.

        For example, if we flatten a value [(1, (2, 3)), {"foo": 4}] with a treedef
        [(*, *), *], the result is the list of leaves [1, (2, 3), {"foo": 4}]."""
        leaves = py.list(range(self.num_leaves))
        agenda = [xs]
        leaf = self.num_leaves - 1
        for (node_arity, num_leaves, num_nodes, objtype, node_data) in self.traversal_[::-1]:
            object = agenda.pop()
            if objtype is _NoneType:
                pass
            elif node_arity == -1:
                assert leaf >= 0, "Leaf count mismatch"
                leaves[leaf] = object
                leaf -= 1
            elif objtype is py.tuple:
                if not isinstance(object, py.tuple):
                    raise ValueError("Expected tuple, got %s." % py.repr(object))
                tuple: T.Tuple = object
                if len(tuple) != node_arity:
                    raise ValueError("Tuple arity mismatch: %d != %d; tuple: %s." % (
                        len(tuple), node_arity, py.repr(object)
                    ))
                agenda.extend(tuple)
            elif objtype is py.list:
                if not isinstance(object, py.list):
                    raise ValueError("Expected list, got %s." % py.repr(object))
                list: T.List = object
                if len(list) != node_arity:
                    raise ValueError("List arity mismatch: %d != %d; list: %s." % (
                        len(list), node_arity, py.repr(object)
                    ))
                agenda.extend(list)
            elif objtype is py.dict:
                if not isinstance(object, py.dict):
                    raise ValueError("Expected dict, got %s." % py.repr(object))
                dict: T.Dict = object
                keys = py.list(dict.keys())
                if keys != node_data:
                    raise ValueError("Dict key mismatch; expected keys: %s; dict: %s." % (
                        py.repr(node_data), py.repr(object)
                    ))
                agenda.extend(dict.values())
            elif node_data == "namedtuple" and issubclass(objtype, py.tuple):
                if not isinstance(object, py.tuple) or not hasattr(object, "_fields"):
                    raise ValueError("Expected named tuple, got %s." % py.repr(object))
                tuple: typing.NamedTuple = object
                if len(tuple) != node_arity:
                    raise ValueError("Named tuple arity mismatch: %d != %d; tuple: %s." % (
                        len(tuple), node_arity, py.repr(object)
                    ))
                if py.type(tuple) != objtype:
                    raise ValueError("Named tuple type mismatch: expected type: %s, tuple: %s." % (
                        py.repr(objtype), py.repr(object)
                    ))
                agenda.extend(tuple)
            else:
                reg = PyTreeTypeRegistry.lookup(objtype)
                if reg is None:
                    raise ValueError("Custom node type mismatch: expected type: %s, value: %s." % (
                        py.repr(objtype), py.repr(object)
                    ))
                out: T.Tuple = reg.to_iterable(object)
                if len(out) != 2:
                    raise RuntimeError("PyTree custom to_iterable function should return a pair")
                if node_data != out[1]:
                    raise ValueError("Mismatch custom node data: %s != %s; value: %s." % (
                        py.repr(node_data), py.repr(out[1]), py.repr(object)
                    ))
                arity = len(out[0])
                if arity != node_arity:
                    raise ValueError("Custom type arity mismatch: %d != %d; value: %s." % (
                        arity, node_arity, py.repr(object)
                    ))
                agenda.extend(out[0])
            if len(agenda) <= 0:
                break
        if leaf != -1:
            raise ValueError("Tree structures did not match: %s vs %s" % (
                py.repr(xs), py.str(self)
            ))
        return leaves

    def children(self) -> typing.List[PyTreeDef]:
        children = []
        pos = len(self.traversal_) - 1
        if pos >= 0:
            (root_arity, *root) = self.traversal_[-1]
            for i in range(root_arity - 1, -1, -1):
                child = PyTreeDef()
                (node_arity, num_leaves, num_nodes, objtype, node_data) = self.traversal_[pos - 1]
                assert pos >= num_nodes, "children() walked off start of array"
                child.traversal_.extend(self.traversal_[pos - num_nodes:pos])
                children.append(child)
                pos -= num_nodes
            assert pos == 0, "pos != 0 at end of PyTreeDef::Children"
        return children[::-1]

    def __str__(self):
        agenda = []
        for (node_arity, num_leaves, num_nodes, objtype, node_data) in self.traversal_:
            assert len(agenda) >= max(node_arity, 0), "Too few elements for container."
            representation = []
            if node_arity < 0 and objtype is not _NoneType:
                agenda.append("*")
                continue
            elif node_arity < 0 and objtype is _NoneType:
                representation.append("None")
            else:
                children = '' if node_arity <= 0 else str_join(agenda[-node_arity:], ", ")
                if objtype is py.tuple:
                    # Tuples with only one element must have a trailing comma.
                    if node_arity == 1:
                        children += ","
                    representation.append(str_cat("(", children, ")"))
                elif objtype is py.list:
                    representation.append(str_cat("[", children, "]"))
                elif objtype is py.dict:
                    separator = "{"
                    keys = node_data
                    values = agenda[-node_arity:]
                    for key, value in safe_zip(keys, values):
                        representation.append('%s%s: %s' % (separator, repr(key), value))
                        separator = ", "
                    representation.append('}')
                else:
                    kind = str(objtype)
                    if node_data == "namedtuple" and issubclass(objtype, tuple):
                        node_data, kind = kind, node_data
                    data = '[%s]' % str(node_data)
                    representation.append('CustomNode(%s%s, [%s])' % (kind, data, children))
            if node_arity > 0:
                del agenda[-node_arity:]
            agenda.append(''.join(representation))
        return str_cat("PyTreeDef(", agenda, ")")

    def __repr__(self) -> py.str:
        return str(self)

    def __eq__(self, other: PyTreeDef):
        if not isinstance(other, PyTreeDef):
            return False
        return self.traversal_ == other.traversal_

    @property
    def num_leaves(self) -> py.int:
        if len(self.traversal_) <= 0:
            return 0
        (node_arity, num_leaves, num_nodes, objtype, node_data) = self.traversal_[-1]
        return num_leaves

    @property
    def num_nodes(self) -> py.int:
        return len(self.traversal_)



def str_join(xs: T.Iterable, sep=', '):
    return sep.join([str(x) for x in xs])


def str_cat(*xs: T.Optional[T.Iterable, py.str], sep=', ') -> py.str:
    r = []
    for x in xs:
        if isinstance(x, str):
            r.append(x)
        else:
            r.append(str_join(x, sep=sep))
    return ''.join(r)


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
    pytree = PyTreeDef(nodes)
    pytree.flatten_into(tree, leaves, nodes, is_leaf, 0, 0)
    return leaves, pytree


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
      treedef = PyTreeDef(treedef)
  return treedef.unflatten(leaves)


def register_pytree_node(nodetype: T.Type, flatten_func: T.Callable, unflatten_func: T.Callable):
    """Extends the set of types that are considered internal nodes in pytrees.

    See `example usage <https://jax.readthedocs.io/en/latest/notebooks/JAX_pytrees.html#Pytrees-are-extensible>`_.

    Args:
      nodetype: a Python type to treat as an internal pytree node.
      flatten_func: a function to be used during flattening, taking a value
        of type `nodetype` and returning a pair, with (1) an iterable for
        the children to be flattened recursively, and (2) some auxiliary data
        to be stored in the treedef and to be passed to the `unflatten_func`.
      unflatten_func: a function taking two arguments: the auxiliary data that
        was returned by `flatten_func` and stored in the treedef, and the
        unflattened children. The function should return an instance of
        `nodetype`.
    """
    PyTreeTypeRegistry.register(nodetype, flatten_func, unflatten_func)
    # _registry[nodetype] = _RegistryEntry(flatten_func, unflatten_func)


def register_pytree_node_class(cls: T.Type):
    """Extends the set of types that are considered internal nodes in pytrees.

    This function is a thin wrapper around ``register_pytree_node``, and provides
    a class-oriented interface:

      @register_pytree_node_class
      class Special:
        def __init__(self, x, y):
          self.x = x
          self.y = y
        def tree_flatten(self):
          return ((self.x, self.y), None)
        @classmethod
        def tree_unflatten(cls, aux_data, children):
          return cls(*children)
    """
    register_pytree_node(cls, op.methodcaller('tree_flatten'), cls.tree_unflatten)
    return cls


def safe_zip(*args):
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
    return list(zip(*args))


register_pytree_node(
    list,
    lambda x: (x, None),
    lambda _, values: values)

register_pytree_node(
    tuple,
    lambda x: (x, tuple),
    lambda _, values: tuple(values))

register_pytree_node(
    collections.namedtuple,
    lambda x: (x, type(x)),
    lambda kind, values: kind(values))

register_pytree_node(
    dict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda keys, values: dict(safe_zip(keys, values)))

register_pytree_node(
    collections.OrderedDict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda keys, values: collections.OrderedDict(safe_zip(keys, values)))

register_pytree_node(
    collections.defaultdict,
    lambda x: (tuple(x.values()), (x.default_factory, tuple(x.keys()))),
    lambda s, values: collections.defaultdict(s[0], safe_zip(s[1], values)))
