from __future__ import annotations
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


class PyTreeDef:
    def __init__(self, nodes = None):
        self.traversal_ = nodes if nodes is not None else []

    @classmethod
    def flatten_into(cls, handle, leaves: T.List[T.Any], nodes: T.List[T.Any], leaf_predicate: T.Callable, start_num_nodes: py.int, start_num_leaves: py.int):
        objtype = type(handle)
        node_data = None
        node_arity = -1
        num_nodes, num_leaves = start_num_nodes + 1, start_num_leaves + 1
        if handle is None:
            num_leaves -= 1
        elif objtype in [py.int, py.float, py.bool] \
                or (leaf_predicate and leaf_predicate(handle)):
            objtype = None
            leaves.append(handle)
        else:
            reg = PyTreeTypeRegistry.lookup(objtype)
            if reg is not None or is_namedtuple(handle, objtype):
                blades, node_data = reg.to_iterable(handle) if reg else (handle, "namedtuple")
                node_arity = 0
                for x in blades:
                    node_arity += 1
                    num_nodes, num_leaves = cls.flatten_into(x, leaves, nodes, leaf_predicate, num_nodes, num_leaves)
                num_leaves -= 1
            else:
                objtype = None
                leaves.append(handle)
        node = (node_arity, num_leaves - start_num_leaves, num_nodes - start_num_nodes, objtype, node_data)
        nodes.append(node)
        return num_nodes, num_leaves

    def unflatten(self, leaves: T.Iterable):
        leaves: T.List = list(leaves)
        leaf_count = 0
        for (node_arity, num_leaves, num_nodes, objtype, node_data) in self.traversal_:
            if node_arity == -1:
                if objtype is _NoneType:
                    leaves.insert(leaf_count, None)
                    leaf_count += 1
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

    def compose(self, inner: PyTreeDef) -> PyTreeDef:
        """Composes two PyTreeDefs, replacing the leaves of this tree with copies of `inner`."""
        out_nodes = []
        for node in self.traversal_:
            (node_arity, num_leaves, num_nodes, objtype, node_data) = node
            if node_arity == -1 and objtype is not _NoneType:
                out_nodes.extend(inner.traversal_)
            else:
                out_nodes.append(node)
        _, root_num_leaves, root_num_nodes, _, _ = self.traversal_[-1]
        _, inner_root_num_leaves, inner_root_num_nodes, _, _ = inner.traversal_[-1]
        node_arity, num_leaves, num_nodes, objtype, node_data = out_nodes.pop()
        num_nodes = (root_num_nodes - root_num_leaves) + (inner_root_num_nodes * root_num_leaves)
        num_leaves *= inner_root_num_leaves
        node = (node_arity, num_leaves, num_nodes, objtype, node_data)
        out_nodes.append(node)
        return PyTreeDef(py.tuple(out_nodes))

    def walk(self, f_node: T.Optional[T.Callable], f_leaf: T.Optional[T.Callable], leaves: T.Iterable):
        """Maps a function over a PyTree structure, applying f_leaf to each leaf, and
        f_node to each container node.

        TODO(phawkins): use flattening everywhere instead and delete this method."""
        agenda = []
        it = iter(leaves)
        for node in self.traversal_:
            (node_arity, num_leaves, num_nodes, objtype, node_data) = node
            if node_arity == -1 and objtype is not _NoneType:
                ok, leaf = next_value(it)
                if not ok:
                    raise ValueError("Too few leaves for PyTreeDef")
                agenda.append(f_leaf(leaf) if f_leaf is not None else leaf)
            else:
                assert len(agenda) >= node_arity, "Too few elements for custom type."
                tuple = []
                if node_arity > 0:
                    tuple = agenda[-node_arity:]
                    del agenda[-node_arity:]
                tuple = py.tuple(tuple)
                agenda.append(f_node(tuple) if f_node is not None else tuple)
        ok, _ = next_value(it)
        if ok:
            raise ValueError("Too many leaves for PyTreeDef")
        assert len(agenda) == 1, "PyTreeDef traversal did not yield a singleton."
        return agenda[-1]

    def _from_iterable_tree_helper(self, tree: T.Any, it: T.Iterator):
        """Recursive helper used to implement from_iterable_tree()"""
        ok, node = next_value(it)
        if not ok:
            raise ValueError("Tree structures did not match.")
        (node_arity, num_leaves, num_nodes, objtype, node_data) = node
        if node_arity == -1:
            if objtype is _NoneType:
                return None
            return tree
        iterable: T.Iterable = tree
        ys = py.list(iterable)
        if len(ys) != node_arity:
            raise ValueError("Arity mismatch between trees")
        for j in range(node_arity - 1, -1, -1):
            ys[j] = self._from_iterable_tree_helper(ys[j], it)
        reg = PyTreeTypeRegistry.lookup(objtype)
        if reg is not None:
            o = reg.from_iterable(node_data, ys)
        else:
            o = objtype(*ys)
        return o

    def from_iterable_tree(self, tree: T.Iterable):
        """Given a tree of iterables with the same node/leaf structure as this PyTree,
        build the corresponding PyTree.

        TODO(phawkins): use flattening everywhere instead and delete this method."""
        it = iter(self.traversal_[::-1])
        out = self._from_iterable_tree_helper(tree, it)
        ok, _ = next_value(it)
        if ok:
            raise ValueError("Tree structures did not match.")
        return out

    def flatten_up_to(self, xs: T.Any):
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
                tuple: T.NamedTuple = object
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

    def children(self) -> T.List[PyTreeDef]:
        children = []
        pos = len(self.traversal_) - 1
        if pos >= 0:
            (root_arity, *root) = self.traversal_[-1]
            for i in range(root_arity - 1, -1, -1):
                (node_arity, num_leaves, num_nodes, objtype, node_data) = self.traversal_[pos - 1]
                assert pos >= num_nodes, "children() walked off start of array"
                child = PyTreeDef(py.tuple(self.traversal_[pos - num_nodes:pos]))
                children.append(child)
                pos -= num_nodes
            assert pos == 0, "pos != 0 at end of PyTreeDef::Children"
        return children[::-1]

    @staticmethod
    def tuple(defs: T.List[PyTreeDef]) -> PyTreeDef:
        """Makes a Tuple PyTreeDef out of a vector of PyTreeDefs."""
        defs = py.list(defs)
        nodes = []
        node_arity, num_leaves, num_nodes, objtype, node_data = len(defs), 0, 0, py.tuple, None
        for td in defs:
            nodes.extend(td.traversal_)
            num_leaves += td.num_leaves
            num_nodes += td.num_nodes
        node = (node_arity, num_leaves, num_nodes, objtype, node_data)
        nodes.append(node)
        return PyTreeDef(py.tuple(nodes))

    @classmethod
    def flatten(cls, tree: T.Any, is_leaf: T.Optional[T.Callable[[T.Any], py.bool]] = None) -> (T.Iterable, PyTreeDef):
        """Flattens a Pytree into a list of leaves and a PyTreeDef.

        Returns references to the flattened objects, which might be temporary
        objects in the case of custom pytype handlers."""
        leaves = []
        nodes = []
        cls.flatten_into(tree, leaves, nodes, is_leaf, 0, 0)
        return leaves, cls(py.tuple(nodes))

    @staticmethod
    def is_leaf(handle: T.Any) -> py.bool:
        objtype = type(handle)
        reg = PyTreeTypeRegistry.lookup(objtype)
        if reg is not None or is_namedtuple(handle, objtype) or objtype is _NoneType:
            return False
        return True

    @classmethod
    def all_leaves(cls, iterable: T.Iterable) -> py.bool:
        "Tests whether the given list is a flat list of leaves."
        for handle in iterable:
            if not cls.is_leaf(handle):
                return False
        return True

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
        if len(self.traversal_) != len(other.traversal_):
            return False
        for a, b in zip(self.traversal_, other.traversal_):
            (a_node_arity, a_num_leaves, a_num_nodes, a_objtype, a_node_data) = a
            (b_node_arity, b_num_leaves, b_num_nodes, b_objtype, b_node_data) = b
            if a_objtype != b_objtype or a_node_arity != b_node_arity:
                return False
            if (a_node_data is None) != (b_node_data is None):
                return False
            if a_node_data is not None and a_node_data != b_node_data:
                return False
            # We don't need to test equality of num_leaves and num_nodes since they
            # are derivable from the other node data.
        return True

    def __hash__(self):
        return py.hash(((node_arity, objtype) for (node_arity, num_leaves, num_nodes, objtype, node_data) in self.traversal_))

    @property
    def num_leaves(self) -> py.int:
        if len(self.traversal_) <= 0:
            return 0
        (node_arity, num_leaves, num_nodes, objtype, node_data) = self.traversal_[-1]
        return num_leaves

    @property
    def num_nodes(self) -> py.int:
        return len(self.traversal_)


_module = type(collections)
pytree = _module("pytree")
pytree.flatten = PyTreeDef.flatten
pytree.tuple = PyTreeDef.tuple
pytree.all_leaves = PyTreeDef.all_leaves
pytree.register_node = PyTreeTypeRegistry.register


def tree_flatten(tree: T.Any, is_leaf: T.Optional[T.Callable[[T.Any], py.bool]] = None) -> (T.Iterable, PyTreeDef):
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
    return pytree.flatten(tree, is_leaf)


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
    return treedef.unflatten(leaves)


def tree_leaves(tree) -> T.List:
    """Gets the leaves of a pytree."""
    return pytree.flatten(tree)[0]


def tree_structure(tree) -> PyTreeDef:
    """Gets the treedef for a pytree."""
    return pytree.flatten(tree)[1]


def treedef_tuple(treedefs: T.Iterable[PyTreeDef]) -> PyTreeDef:
    """Makes a tuple treedef from a list of child treedefs."""
    return pytree.tuple(list(treedefs))


def treedef_children(treedef: PyTreeDef) -> T.List[PyTreeDef]:
    return treedef.children()


def treedef_is_leaf(treedef: PyTreeDef) -> py.bool:
    return treedef.num_nodes == 1


def all_leaves(iterable: T.Iterable) -> py.bool:
    """Tests whether all elements in the given iterable are all leaves.

    >>> tree = {"a": [1, 2, 3]}
    >>> assert all_leaves(jax.tree_leaves(tree))
    >>> assert not all_leaves([tree])

    This function is useful in advanced cases, for example if a library allows
    arbitrary map operations on a flat list of leaves it may want to check if
    the result is still a flat list of leaves.

    Args:
      iterable: Iterable of leaves.

    Returns:
      A boolean indicating if all elements in the input are leaves.
    """
    return pytree.all_leaves(iterable)


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
    pytree.register_node(nodetype, flatten_func, unflatten_func)


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


def tree_map(f: T.Callable[..., T.Any], tree: T.Any, *rest: T.Any,
             is_leaf: T.Optional[T.Callable[[T.Any], py.bool]] = None) -> T.Any:
    """Maps a multi-input function over pytree args to produce a new pytree.

    Args:
      f: function that takes ``1 + len(rest)`` arguments, to be applied at the
        corresponding leaves of the pytrees.
      tree: a pytree to be mapped over, with each leaf providing the first
        positional argument to ``f``.
      *rest: a tuple of pytrees, each of which has the same structure as tree or
        or has tree as a prefix.
      is_leaf: an optionally specified function that will be called at each
        flattening step. It should return a boolean, which indicates whether
        the flattening should traverse the current object, or if it should be
        stopped immediately, with the whole subtree being treated as a leaf.

    Returns:
      A new pytree with the same structure as ``tree`` but with the value at each
      leaf given by ``f(x, *xs)`` where ``x`` is the value at the corresponding
      leaf in ``tree`` and ``xs`` is the tuple of values at corresponding nodes in
      ``rest``.
    """
    leaves, treedef = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


tree_multimap = tree_map


# TODO(mattjj,phawkins): consider removing this function
def _process_pytree(process_node: T.Callable, tree: T.Any):
    leaves, treedef = pytree.flatten(tree)
    return treedef.walk(process_node, None, leaves), treedef


def build_tree(treedef: PyTreeDef, xs: T.Any):
    return treedef.from_iterable_tree(xs)


def tree_transpose(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef, pytree_to_transpose: T.Any):
    flat, treedef = tree_flatten(pytree_to_transpose)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        expected_treedef = outer_treedef.compose(inner_treedef)
        raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
    flat = iter(flat)
    lol = [[next(flat) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = zip(*lol)
    subtrees = map(partial(tree_unflatten, outer_treedef), transposed_lol)
    return tree_unflatten(inner_treedef, subtrees)


no_initializer = py.object()
U = T.TypeVar("U")


@T.overload
def tree_reduce(function: T.Callable[[U, T.Any], U],
                tree: T.Any) -> U:
    ...


@T.overload
def tree_reduce(function: T.Callable[[U, T.Any], U],
                tree: T.Any,
                initializer: U) -> U:
    ...


def tree_reduce(function: T.Callable[[U, T.Any], U],
                tree: T.Any,
                initializer: T.Any = no_initializer) -> U:
    if initializer is no_initializer:
        return functools.reduce(function, tree_leaves(tree))
    else:
        return functools.reduce(function, tree_leaves(tree), initializer)


def tree_all(tree):
    return all(tree_leaves(tree))


def is_namedtuple(obj, objtype):
    return hasattr(obj, '_fields') and isinstance(obj, tuple)
    # objtype.__bases__ and objtype.__bases__[0] is tuple


def next_value(it):
    try:
        return True, next(it)
    except StopIteration:
        return False, None


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

class Partial(functools.partial):
    """A version of functools.partial that works in pytrees.

    Use it for partial function evaluation in a way that is compatible with JAX's
    transformations, e.g., ``Partial(func, *args, **kwargs)``.

    (You need to explicitly opt-in to this behavior because we didn't want to give
    functools.partial different semantics than normal function closures.)
    """

register_pytree_node(
    Partial,
    lambda partial_: ((partial_.args, partial_.keywords), partial_.func),
    lambda func, xs: Partial(func, *xs[0], **xs[1]),
)
