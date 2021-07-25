from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
import typing
import builtins as py
import operator as op
import collections
import functools
from functools import partial
from copy import deepcopy

T = typing.TypeVar("T")


# https://stackoverflow.com/questions/1966591/hasnext-in-python-iterators
class iter(object):
    def __init__(self, it):
        self.it = py.iter(it)
        self._hasnext = None
    def __iter__(self):
        return self
    def next(self):
        if hasattr(self, '_thenext'):
            value = self._thenext
            delattr(self, '_thenext')
            self._hasnext = None
            return value
        else:
            try:
                return py.next(self.it)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
    def hasnext(self):
        if self._hasnext is None:
            try:
                self._thenext = py.next(self.it)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
        return self._hasnext


def next(it: iter):
    return it.next()


def finished_iterating(x: iter):
    return not x.hasnext()


def not_equal(a, b) -> py.bool:
    return a != b


def str_join(xs: typing.Iterable, sep=', '):
    return sep.join([str(x) for x in xs])


def str_cat(*xs: typing.Optional[typing.Iterable, py.str], sep=', ') -> py.str:
    r = []
    for x in xs:
        if isinstance(x, str):
            r.append(x)
        else:
            r.append(str_join(x, sep=sep))
    return ''.join(r)


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
        self.registrations_ = []
        def add_builtin_type(type_obj, kind: PyTreeKind):
            registration = self.Registration(kind=kind, type=type_obj)
            self.registrations_.append(registration)
        add_builtin_type(type(None), PyTreeKind.kNone)
        add_builtin_type(tuple, PyTreeKind.kTuple)
        add_builtin_type(list, PyTreeKind.kList)
        add_builtin_type(dict, PyTreeKind.kDict)

    @dataclass
    class Registration:
        kind: PyTreeKind

        # The following values are populated for custom types.
        # The Python type object, used to identify the type.
        type: typing.Any # pybind11::object type;

        # A function with signature: object -> (iterable, aux_data)
        to_iterable: typing.Callable = None # pybind11::function to_iterable;

        # A function with signature: (aux_data, iterable) -> object
        from_iterable: typing.Callable = None # pybind11::function from_iterable;

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
        if not hasattr(cls, 'inst'):
            cls.inst = cls()
        return cls.inst

    @classmethod
    def register(cls, type: typing.Type, to_iterable: typing.Callable, from_iterable: typing.Callable):
        self = cls.singleton()
        registration = cls.Registration(PyTreeKind.kCustom, type, to_iterable, from_iterable)
        if registration in self.registrations_:
            raise ValueError("Duplicate custom PyTreeDef type registration for %s." % repr(type))
        self.registrations_.append(registration)

    @classmethod
    def lookup(cls, type: typing.Type):
        self = cls.singleton()
        for registration in self.registrations_:
            if registration.type == type:
                return registration


class PyTreeDef:
    """A PyTreeDef describes the tree structure of a PyTree.

    A PyTree is a tree of Python values, where the interior nodes are tuples, lists,
    dictionaries, or user-defined containers, and the leaves are other objects."""
    def __init__(self):
        self.traversal_: typing.List[PyTreeDef.Node] = []

    @dataclass
    class Node:
        kind: PyTreeKind = PyTreeKind.kLeaf # PyTreeKind kind = PyTreeKind::kLeaf;

        # Arity for non-kLeaf types.
        arity: int = 0 # int arity = 0;

        # Kind-specific auxiliary data. For a kNamedTuple, contains the tuple type
        # object. For a kDict, contains a sorted list of keys. For a kCustom type,
        # contains the auxiliary data returned by the `to_iterable` function.
        node_data: typing.Any = None # pybind11::object node_data;

        custom: PyTreeTypeRegistry.Registration = None # const PyTreeTypeRegistry::Registration* custom = nullptr;

        # Number of leaf nodes in the subtree rooted at this node.
        num_leaves: int = 0 # int num_leaves = 0;

        # Number of leaf and interior nodes in the subtree rooted at this node.
        num_nodes: int = 0 # int num_nodes = 0;

        def __eq__(a: PyTreeDef.Node, b: PyTreeDef.Node):
            if a.kind.value != b.kind.value:
                return False
            if a.arity != b.arity:
                return False
            if (a.node_data is None) != (b.node_data is None):
                return False
            if a.custom != b.custom:
                return False
            if a.node_data is not None and not_equal(a.node_data, b.node_data):
                return False
            return True

        def str(node, agenda: typing.List):
            assert len(agenda) >= node.arity, "Too few elements for container."
            children = str_join(agenda[-node.arity:], ", ")
            representation = []
            if node.kind == PyTreeKind.kLeaf:
                agenda.append("*")
                return
            elif node.kind == PyTreeKind.kNone:
                representation.append("None")
            elif node.kind == PyTreeKind.kTuple:
                # Tuples with only one element must have a trailing comma.
                if node.arity == 1:
                    children += ","
                representation.append(str_cat("(", children, ")"))
            elif node.kind == PyTreeKind.kList:
                representation.append(str_cat("[", children, "]"))
            elif node.kind == PyTreeKind.kDict:
                assert len(node.node_data) == node.arity, "Number of keys and entries does not match."
                separator = "{"
                keys = node.node_data
                values = agenda[-node.arity:]
                for key, value in zip(keys, values):
                    representation.append('%s%s: %s' % (separator, repr(key), value))
                    separator = ", "
                representation.append('}')
            elif node.kind == PyTreeKind.kNamedTuple or node.kind == PyTreeKind.kCustom:
                if node.kind == PyTreeKind.kNamedTuple:
                    kind = "namedtuple"
                else:
                    kind = str(node.custom.type)
                data = '[%s]' % str(node.node_data)
                representation.append('CustomNode(%s%s, [%s])' % (kind, data, children))
            for i in range(node.arity):
                agenda.pop()
            agenda.append(''.join(representation))

    def __str__(self) -> py.str:
        agenda = []
        if len(self.traversal_) > 0:
            for node in self.traversal_:
                node.str(agenda)
            assert len(agenda) == 1, "PyTreeDef traversal did not yield a singleton."
        return str_cat("PyTreeDef(", agenda, ")")

    def __repr__(self) -> py.str:
        return str(self)

    def __eq__(self, other: PyTreeDef):
        if len(self.traversal_) != len(other.traversal_):
            return False
        for i in range(len(self.traversal_)):
            a: PyTreeDef.Node = self.traversal_[i]
            b: PyTreeDef.Node = other.traversal_[i]
            if a != b:
                return False
        # We don't need to test equality of num_leaves and num_nodes since they
        # are derivable from the other node data.
        return True

    @classmethod
    def get_kind(cls, obj: typing.Any) -> (PyTreeKind, PyTreeTypeRegistry.Registration):
        typ = type(obj)
        registration = PyTreeTypeRegistry.lookup(typ)
        if registration is not None:
            return registration.kind, registration
        elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
            # We can only identify namedtuples heuristically, here by the presence of
            # a _fields attribute.
            return PyTreeKind.kNamedTuple, registration
        else:
            return PyTreeKind.kLeaf, registration

    def flatten_into(self, handle: typing.Any, leaves: typing.List[typing.Any], leaf_predicate: typing.Callable):
        #   Node node;
        node = PyTreeDef.Node()
        #   int start_num_nodes = traversal_.size();
        start_num_nodes = len(self.traversal_)
        #   int start_num_leaves = leaves.size();
        start_num_leaves = len(leaves)
        #   if (leaf_predicate && (*leaf_predicate)(handle).cast<bool>()) {
        #     leaves.push_back(py::reinterpret_borrow<py::object>(handle));
        if leaf_predicate is not None and leaf_predicate(handle):
            leaves.append(handle)
        #   } else {
        else:
            # node.kind = GetKind(handle, &node.custom);
            node.kind, node.custom = self.get_kind(handle)
            # auto recurse = [this, &leaf_predicate, &leaves](py::handle child) {
            #   FlattenInto(child, leaves, leaf_predicate);
            # };
            def recurse(child):
                self.flatten_into(child, leaves, leaf_predicate)
            # switch (node.kind) {
            #   case PyTreeKind::kNone:
            if node.kind == PyTreeKind.kNone:
                # // Nothing to do.
                # break;
                pass
            #   case PyTreeKind::kTuple: {
            elif node.kind == PyTreeKind.kTuple:
                # node.arity = PyTuple_GET_SIZE(handle.ptr());
                # node.arity = len(handle)
                # for (int i = 0; i < node.arity; ++i) {
                #   recurse(PyTuple_GET_ITEM(handle.ptr(), i));
                # }
                # for i in range(node.arity):
                #     recurse(handle[i])
                node.arity = 0
                for x in handle:
                    node.arity += 1
                    self.flatten_into(x, leaves, leaf_predicate)
                # break;
            #   }
            #   case PyTreeKind::kList: {
            elif node.kind == PyTreeKind.kList:
                # node.arity = PyList_GET_SIZE(handle.ptr());
                # node.arity = len(handle)
                # for (int i = 0; i < node.arity; ++i) {
                #   recurse(PyList_GET_ITEM(handle.ptr(), i));
                # }
                # for i in range(node.arity):
                #     recurse(handle[i])
                # [self.flatten_into(x, leaves, leaf_predicate) for x in handle]
                node.arity = 0
                for x in handle:
                    node.arity += 1
                    self.flatten_into(x, leaves, leaf_predicate)
                # break;
            #   }
            #   case PyTreeKind::kDict: {
            elif node.kind == PyTreeKind.kDict:
                # py::dict dict = py::reinterpret_borrow<py::dict>(handle);
                # dict: typing.Dict = handle
                # py::list keys =
                #     py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
                # if (PyList_Sort(keys.ptr())) {
                #   throw std::runtime_error("Dictionary key sort failed.");
                # }
                # keys: typing.List = list(sorted(dict.keys()))
                # for (py::handle key : keys) {
                #   recurse(dict[key]);
                # }
                # for key in keys:
                #     recurse(dict[key])
                # node.arity = dict.size();
                # node.arity = len(dict)
                # node.node_data = std::move(keys);
                # node.node_data = keys
                node.node_data = list(sorted(handle.keys()))
                node.arity = 0
                for k in node.node_data:
                    x = handle[k]
                    node.arity += 1
                    self.flatten_into(x, leaves, leaf_predicate)
                # break;
            #   }
            #   case PyTreeKind::kCustom: {
            elif node.kind == PyTreeKind.kCustom:
                # py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(handle));
                out: typing.Tuple = node.custom.to_iterable(handle)
                # if (out.size() != 2) {
                #   throw std::runtime_error(
                #       "PyTree custom to_iterable function should return a pair");
                # }
                if len(out) != 2:
                    raise RuntimeError("PyTree custom to_iterable function should return a pair")
                # node.node_data = out[1];
                node.node_data = out[1]
                # node.arity = 0;
                node.arity = 0
                # for (py::handle entry : py::cast<py::iterable>(out[0])) {
                #   ++node.arity;
                #   recurse(entry);
                # }
                for entry in out[0]:
                    node.arity += 1
                    self.flatten_into(entry, leaves, leaf_predicate)
                    # recurse(entry)
                # break;
            #   }
            #   case PyTreeKind::kNamedTuple: {
            elif node.kind == PyTreeKind.kNamedTuple:
                # py::tuple tuple = py::reinterpret_borrow<py::tuple>(handle);
                tuple: typing.NamedTuple = handle
                # node.arity = tuple.size();
                node.arity = len(tuple)
                # node.node_data = py::reinterpret_borrow<py::object>(tuple.get_type());
                node.node_data = type(tuple)
                # for (py::handle entry : tuple) {
                #   recurse(entry);
                # }
                for entry in tuple:
                    # recurse(entry)
                    self.flatten_into(entry, leaves, leaf_predicate)
                # break;
            #   }
            #   default:
            else:
                # DCHECK(node.kind == PyTreeKind::kLeaf);
                assert node.kind == PyTreeKind.kLeaf
                # leaves.push_back(py::reinterpret_borrow<py::object>(handle));
                leaves.append(handle)
            # }
        #   }
        #   node.num_nodes = traversal_.size() - start_num_nodes + 1;
        node.num_nodes = len(self.traversal_) - start_num_nodes + 1
        #   node.num_leaves = leaves.size() - start_num_leaves;
        node.num_leaves = len(leaves) - start_num_leaves
        #   traversal_.push_back(std::move(node));
        self.traversal_.append(node)

    @classmethod
    def flatten(cls, x: typing.Any, leaf_predicate: typing.Callable = None) -> (typing.Iterable, PyTreeDef):
        """Flattens a Pytree into a list of leaves and a PyTreeDef.

        Returns references to the flattened objects, which might be temporary
        objects in the case of custom pytype handlers."""
        leaves = []
        tree = cls()
        tree.flatten_into(x, leaves, leaf_predicate)
        return leaves, tree

    @classmethod
    def all_leaves(cls, x: typing.Any) -> bool:
        "Tests whether the given list is a flat list of leaves."
        for v in x:
            kind, registration = cls.get_kind(v)
            if kind != PyTreeKind.kLeaf:
                return False
        return True

    def flatten_up_to(self, xs: typing.Any):
        """Flattens a Pytree up to this PyTreeDef. 'self' must be a tree prefix of
        the tree-structure of 'xs'.

        For example, if we flatten a value [(1, (2, 3)), {"foo": 4}] with a treedef
        [(*, *), *], the result is the list of leaves [1, (2, 3), {"foo": 4}]."""
        #   py::list leaves(num_leaves());
        leaves = resize(None, self.num_leaves)
        #   std::vector<py::object> agenda;
        #   agenda.push_back(py::reinterpret_borrow<py::object>(xs));
        agenda = [xs]
        #   auto it = traversal_.rbegin();
        it = iter(self.traversal_[::-1])
        #   int leaf = num_leaves() - 1;
        leaf = self.num_leaves - 1
        #   while (!agenda.empty()) {
        while len(agenda) > 0:
            # if (it == traversal_.rend()) {
            #   throw std::invalid_argument(absl::StrFormat(
            #       "Tree structures did not match: %s vs %s", py::repr(xs), ToString()));
            # }
            if finished_iterating(it):
                raise ValueError("Tree structures did not match: %s vs %s" % (
                    py.repr(xs), py.str(self)
                ))
            # const Node& node = *it;
            node = next(it)
            # py::object object = agenda.back();
            # agenda.pop_back();
            object = agenda.pop()
            # ++it;
            #
            # switch (node.kind) {
            #   case PyTreeKind::kLeaf:
            if node.kind == PyTreeKind.kLeaf:
                # if (leaf < 0) {
                #   throw std::logic_error("Leaf count mismatch.");
                # }
                assert leaf >= 0, "Leaf count mismatch"
                # leaves[leaf] = py::reinterpret_borrow<py::object>(object);
                leaves[leaf] = object
                # --leaf;
                leaf -= 1
                # break;
            #
            #   case PyTreeKind::kNone:
            #     break;
            elif node.kind == PyTreeKind.kNone:
                pass
            #
            #   case PyTreeKind::kTuple: {
            elif node.kind == PyTreeKind.kTuple:
                # if (!PyTuple_CheckExact(object.ptr())) {
                #   throw std::invalid_argument(
                #       absl::StrFormat("Expected tuple, got %s.", py::repr(object)));
                # }
                if not isinstance(object, py.tuple):
                    raise ValueError("Expected tuple, got %s." % py.repr(object))
                # py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                tuple: py.tuple = object
                # if (tuple.size() != node.arity) {
                #   throw std::invalid_argument(
                #       absl::StrFormat("Tuple arity mismatch: %d != %d; tuple: %s.",
                #                       tuple.size(), node.arity, py::repr(object)));
                # }
                if len(tuple) != node.arity:
                    raise ValueError("Tuple arity mismatch: %d != %d; tuple: %s." % (
                        len(tuple), node.arity, py.repr(object)
                    ))
                # for (py::handle entry : tuple) {
                #   agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                # }
                # for entry in tuple:
                #     agenda.append(entry)
                agenda.extend(tuple)
                # break;
            #   }
            #
            #   case PyTreeKind::kList: {
            elif node.kind == PyTreeKind.kList:
                # if (!PyList_CheckExact(object.ptr())) {
                #   throw std::invalid_argument(
                #       absl::StrFormat("Expected list, got %s.", py::repr(object)));
                # }
                if  not isinstance(object, py.list):
                    raise ValueError("Expected list, got %s." % py.repr(object))
                # py::list list = py::reinterpret_borrow<py::list>(object);
                list: typing.List = object
                # if (list.size() != node.arity) {
                #   throw std::invalid_argument(
                #       absl::StrFormat("List arity mismatch: %d != %d; list: %s.",
                #                       list.size(), node.arity, py::repr(object)));
                # }
                if len(list) != node.arity:
                    raise ValueError("List arity mismatch: %d != %d; list: %s." % (
                        len(list), node.arity, py.repr(object)
                    ))
                # for (py::handle entry : list) {
                #   agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                # }
                # for entry in list:
                #     agenda.append(entry)
                agenda.extend(list)
                # break;
            #   }
            #
            #   case PyTreeKind::kDict: {
            elif node.kind == PyTreeKind.kDict:
                # if (!PyDict_CheckExact(object.ptr())) {
                #   throw std::invalid_argument(
                #       absl::StrFormat("Expected dict, got %s.", py::repr(object)));
                # }
                if  not isinstance(object, py.dict):
                    raise ValueError("Expected dict, got %s." % py.repr(object))
                # py::dict dict = py::reinterpret_borrow<py::dict>(object);
                dict: typing.Dict = object
                # py::list keys =
                #     py::reinterpret_steal<py::list>(PyDict_Keys(dict.ptr()));
                # if (PyList_Sort(keys.ptr())) {
                #   throw std::runtime_error("Dictionary key sort failed.");
                # }
                keys = py.list(py.sorted(dict.keys()))
                # if (keys.not_equal(node.node_data)) {
                #   throw std::invalid_argument(
                #       absl::StrFormat("Dict key mismatch; expected keys: %s; dict: %s.",
                #                       py::repr(node.node_data), py::repr(object)));
                # }
                if not_equal(keys, node.node_data):
                    raise ValueError("Dict key mismatch; expected keys: %s; dict: %s." % (
                        py.repr(node.node_data), py.repr(object)
                    ))
                # for (py::handle key : keys) {
                #   agenda.push_back(dict[key]);
                # }
                for key in keys:
                    agenda.append(dict[key])
                # break;
            #   }
            #
            #   case PyTreeKind::kNamedTuple: {
            elif node.kind == PyTreeKind.kNamedTuple:
                # if (!py::isinstance<py::tuple>(object) ||
                #     !py::hasattr(object, "_fields")) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Expected named tuple, got %s.", py::repr(object)));
                # }
                if not isinstance(object, py.tuple) or not hasattr(object, "_fields"):
                    raise ValueError("Expected named tuple, got %s." % py.repr(object))
                # py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
                tuple: typing.NamedTuple = object
                # if (tuple.size() != node.arity) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Named tuple arity mismatch: %d != %d; tuple: %s.", tuple.size(),
                #       node.arity, py::repr(object)));
                # }
                if len(tuple) != node.arity:
                    raise ValueError("Named tuple arity mismatch: %d != %d; tuple: %s." % (
                        len(tuple), node.arity, py.repr(object)
                    ))
                # if (tuple.get_type().not_equal(node.node_data)) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Named tuple type mismatch: expected type: %s, tuple: %s.",
                #       py::repr(node.node_data), py::repr(object)));
                # }
                if not_equal(py.type(tuple), node.node_data):
                    raise ValueError("Named tuple type mismatch: expected type: %s, tuple: %s." % (
                        py.repr(node.node_data), py.repr(object)
                    ))
                # for (py::handle entry : tuple) {
                #   agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                # }
                # for entry in tuple:
                #     agenda.append(entry)
                agenda.extend(tuple)
                # break;
            #   }
            #
            #   case PyTreeKind::kCustom: {
            elif node.kind == PyTreeKind.kCustom:
                # auto* registration = PyTreeTypeRegistry::Lookup(object.get_type());
                registration = PyTreeTypeRegistry.lookup(py.type(object))
                # if (registration != node.custom) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Custom node type mismatch: expected type: %s, value: %s.",
                #       py::repr(node.custom->type), py::repr(object)));
                # }
                if registration != node.custom:
                    raise ValueError("Custom node type mismatch: expected type: %s, value: %s." % (
                        py.repr(node.custom.type), py.repr(object)
                    ))
                # py::tuple out = py::cast<py::tuple>(node.custom->to_iterable(object));
                out: typing.Tuple = node.custom.to_iterable(object)
                # if (out.size() != 2) {
                #   throw std::runtime_error(
                #       "PyTree custom to_iterable function should return a pair");
                # }
                if len(out) != 2:
                    raise RuntimeError("PyTree custom to_iterable function should return a pair")
                # if (node.node_data.not_equal(out[1])) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Mismatch custom node data: %s != %s; value: %s.",
                #       py::repr(node.node_data), py::repr(out[1]), py::repr(object)));
                # }
                if not_equal(node.node_data, out[1]):
                    raise ValueError("Mismatch custom node data: %s != %s; value: %s." % (
                        py.repr(node.node_data), py.repr(out[1]), py.repr(object)
                    ))
                # int arity = 0;
                # arity = 0
                # for (py::handle entry : py::cast<py::iterable>(out[0])) {
                #   ++arity;
                #   agenda.push_back(py::reinterpret_borrow<py::object>(entry));
                # }
                # for entry in out[0]:
                #     arity += 1
                #     agenda.append(entry)
                arity = len(out[0])
                agenda.extend(out[0])
                # if (arity != node.arity) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Custom type arity mismatch: %d != %d; value: %s.", arity,
                #       node.arity, py::repr(object)));
                # }
                if arity != node.arity:
                    raise ValueError("Custom type arity mismatch: %d != %d; value: %s." % (
                        arity, node.arity, py.repr(object)
                    ))
                # break;
            #   }
            # }
        #   }
        #   if (it != traversal_.rend() || leaf != -1) {
        #     throw std::invalid_argument(absl::StrFormat(
        #         "Tree structures did not match: %s vs %s", py::repr(xs), ToString()));
        #   }
        if not finished_iterating(it) or leaf != -1:
            raise ValueError("Tree structures did not match: %s vs %s" % (
                py.repr(xs), py.str(self)
            ))
        #   return leaves;
        return leaves

    @property
    def num_leaves(self) -> py.int:
        if len(self.traversal_) <= 0:
            return 0
        return self.traversal_[-1].num_leaves

    @property
    def num_nodes(self) -> py.int:
        return len(self.traversal_)

    @staticmethod
    def tuple(defs: typing.Iterable[PyTreeDef]):
        """Makes a Tuple PyTreeDef out of a vector of PyTreeDefs."""
        #   auto out = absl::make_unique<PyTreeDef>();
        out = PyTreeDef()
        #   for (const PyTreeDef& def : defs) {
        #     absl::c_copy(def.traversal_, std::back_inserter(out->traversal_));
        #   }
        for def_ in defs:
            out.traversal_.extend(deepcopy(def_.traversal_))
        #   Node node;
        #   node.kind = PyTreeKind::kTuple;
        #   node.arity = defs.size();
        node = PyTreeDef.Node(kind=PyTreeKind.kTuple, arity=len(defs))
        #   out->traversal_.push_back(node);
        out.traversal_.append(node)
        #   return out;
        return out

    def children(self) -> typing.List[PyTreeDef]:
        #   std::vector<std::unique_ptr<PyTreeDef>> children;
        children = []
        #   if (traversal_.empty()) {
        #     return children;
        #   }
        if len(self.traversal_) <= 0:
            return children
        #   Node const& root = traversal_.back();
        root: PyTreeDef.Node = self.traversal_[-1]
        #   children.resize(root.arity);
        resize(children, root.arity)
        #   int pos = traversal_.size() - 1;
        pos = len(self.traversal_) - 1
        #   for (int i = root.arity - 1; i >= 0; --i) {
        for i in range(root.arity - 1, -1, -1):
            # children[i] = absl::make_unique<PyTreeDef>();
            children[i] = PyTreeDef()
            # const Node& node = traversal_.at(pos - 1);
            node: PyTreeDef.Node = self.traversal_[pos - 1]
            # if (pos < node.num_nodes) {
            #   throw std::logic_error("children() walked off start of array");
            # }
            assert pos >= node.num_nodes, "children() walked off start of array"
            # std::copy(traversal_.begin() + pos - node.num_nodes,
            #           traversal_.begin() + pos,
            #           std::back_inserter(children[i]->traversal_));
            children[i].traversal_.extend(deepcopy(self.traversal_[pos - node.num_nodes:pos]))
            # pos -= node.num_nodes;
            pos -= node.num_nodes
        #   }
        #   if (pos != 0) {
        #     throw std::logic_error("pos != 0 at end of PyTreeDef::Children");
        #   }
        assert pos == 0, "pos != 0 at end of PyTreeDef::Children"
        #   return children;
        return children

    def unflatten(self, leaves: typing.Iterable):
        #   absl::InlinedVector<py::object, 4> agenda;
        agenda = []
        agenda_size = 0
        #   auto it = leaves.begin();
        it = iter(leaves)
        #   int leaf_count = 0;
        leaf_count = 0
        #   for (const Node& node : traversal_) {
        for node in self.traversal_:
            # if (agenda.size() < node.arity) {
            #   throw std::logic_error("Too few elements for TreeDef node.");
            # }
            assert agenda_size >= node.arity, "Too few elements for TreeDef node."
            # switch (node.kind) {
            #   case PyTreeKind::kLeaf:
            if node.kind == PyTreeKind.kLeaf:
                # if (it == leaves.end()) {
                #   throw std::invalid_argument(absl::StrFormat(
                #       "Too few leaves for PyTreeDef; expected %d, got %d", num_leaves(),
                #       leaf_count));
                # }
                if finished_iterating(it):
                    raise ValueError("Too few leaves for PyTreeDef; expected %d, got %d" % (
                        self.num_leaves, leaf_count
                    ))
                # agenda.push_back(py::reinterpret_borrow<py::object>(*it));
                # ++it;
                agenda.append(next(it))
                agenda_size += 1
                # ++leaf_count;
                leaf_count += 1
                # break;
            #
            #   case PyTreeKind::kNone:
            #   case PyTreeKind::kTuple:
            #   case PyTreeKind::kNamedTuple:
            #   case PyTreeKind::kList:
            #   case PyTreeKind::kDict:
            #   case PyTreeKind::kCustom: {
            # elif node.kind in [PyTreeKind.kNone,
            #                    PyTreeKind.kTuple,
            #                    PyTreeKind.kNamedTuple,
            #                    PyTreeKind.kList,
            #                    PyTreeKind.kDict,
            #                    PyTreeKind.kCustom]:
            elif True:
                # const int size = agenda.size();
                # size = len(agenda)
                size = agenda_size
                # absl::Span<py::object> span;
                # if (node.arity > 0) {
                #   span = absl::Span<py::object>(&agenda[size - node.arity], node.arity);
                # }
                span = []
                if node.arity > 0:
                    span = agenda[size - node.arity:size]
                # py::object o = MakeNode(node, span);
                o = self.make_node(node, span)
                # agenda.resize(size - node.arity);
                agenda_size = size - node.arity
                resize(agenda, agenda_size)
                # agenda.push_back(o);
                agenda.append(o)
                agenda_size += 1
                # break;
            #   }
            # }
            else:
                assert False, "Unreachable code."
        #   }
        #   if (it != leaves.end()) {
        #     throw std::invalid_argument(absl::StrFormat(
        #         "Too many leaves for PyTreeDef; expected %d.", num_leaves()));
        #   }
        if not finished_iterating(it):
            raise ValueError("Too many leaves for PyTreeDef; expected %d." % (
                self.num_leaves
            ))
        #   if (agenda.size() != 1) {
        #     throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
        #   }
        # assert len(agenda) == 1, "PyTreeDef traversal did not yield a singleton."
        #   return std::move(agenda.back());
        return agenda[-1]

    @staticmethod
    def make_node(node: PyTreeDef.Node, children: typing.Iterable):
        """Helper that manufactures an instance of a node given its children."""
        #   if (children.size() != node.arity) {
        #     throw std::logic_error("Node arity mismatch.");
        #   }
        assert len(children) == node.arity, "Node arity mismatch."
        #   switch (node.kind) {
        #     case PyTreeKind::kLeaf:
        #       throw std::logic_error("MakeNode not implemented for leaves.");
        assert node.kind != PyTreeKind.kLeaf, "MakeNode not implemented for leaves."
        #
        #     case PyTreeKind::kNone:
        #       return py::none();
        if node.kind == PyTreeKind.kNone:
            return None
        #
        #     case PyTreeKind::kTuple:
        #     case PyTreeKind::kNamedTuple: {
        elif node.kind == PyTreeKind.kTuple or node.kind == PyTreeKind.kNamedTuple:
            #   py::tuple tuple(node.arity);
            tuple = resize(None, node.arity)
            #   for (int i = 0; i < node.arity; ++i) {
            #     tuple[i] = std::move(children[i]);
            #   }
            for i in range(node.arity):
                tuple[i] = children[i]
            #   if (node.kind == PyTreeKind::kNamedTuple) {
            #     return node.node_data(*tuple);
            #   } else {
            #     return std::move(tuple);
            #   }
            if node.kind == PyTreeKind.kNamedTuple:
                return node.node_data(*tuple)
            else:
                return py.tuple(tuple)
            # }
        #
        #     case PyTreeKind::kList: {
        elif node.kind == PyTreeKind.kList:
            #   py::list list(node.arity);
            list: typing.List = resize(None, node.arity)
            #   for (int i = 0; i < node.arity; ++i) {
            #     list[i] = std::move(children[i]);
            #   }
            for i in range(node.arity):
                list[i] = children[i]
            #   return std::move(list);
            return list
            # }
        #
        #     case PyTreeKind::kDict: {
        elif node.kind == PyTreeKind.kDict:
            #   py::dict dict;
            dict: typing.Dict = py.dict()
            #   py::list keys = py::reinterpret_borrow<py::list>(node.node_data);
            keys = node.node_data
            #   for (int i = 0; i < node.arity; ++i) {
            #     dict[keys[i]] = std::move(children[i]);
            #   }
            for i in range(node.arity):
                dict[keys[i]] = children[i]
            #   return std::move(dict);
            return dict
            #   break;
            # }
        #     case PyTreeKind::kCustom: {
        elif node.kind == PyTreeKind.kCustom:
            #   py::tuple tuple(node.arity);
            tuple = resize(None, node.arity)
            #   for (int i = 0; i < node.arity; ++i) {
            #     tuple[i] = std::move(children[i]);
            #   }
            for i in range(node.arity):
                tuple[i] = children[i]
            tuple = py.tuple(tuple)
            #   return node.custom->from_iterable(node.node_data, tuple);
            return node.custom.from_iterable(node.node_data, tuple)
            # }
        #   }
        #   throw std::logic_error("Unreachable code.");
        assert False, "Unreachable code."

    def compose(self, inner: PyTreeDef) -> PyTreeDef:
        """Composes two PyTreeDefs, replacing the leaves of this tree with copies of `inner`."""
        #   auto out = absl::make_unique<PyTreeDef>();
        out = PyTreeDef()
        #   for (const Node& n : traversal_) {
        for n in self.traversal_:
            # if (n.kind == PyTreeKind::kLeaf) {
            #   absl::c_copy(inner.traversal_, std::back_inserter(out->traversal_));
            # } else {
            #   out->traversal_.push_back(n);
            # }
            if n.kind == PyTreeKind.kLeaf:
                out.traversal_.extend(deepcopy(inner.traversal_))
            else:
                out.traversal_.append(deepcopy(n))
        #   }
        #   const auto& root = traversal_.back();
        root = self.traversal_[-1]
        #   const auto& inner_root = inner.traversal_.back();
        inner_root = inner.traversal_[-1]
        #   // TODO(tomhennigan): This should update all nodes in the traversal.
        #   auto& out_root = out->traversal_.back();
        out_root = out.traversal_[-1]
        #   out_root.num_nodes = (root.num_nodes - root.num_leaves) +
        #                        (inner_root.num_nodes * root.num_leaves);
        out_root.num_nodes = (root.num_nodes - root.num_leaves) + \
                             (inner_root.num_nodes * root.num_leaves)
        #   out_root.num_leaves *= inner_root.num_leaves;
        out_root.num_leaves *= inner_root.num_leaves
        #   return out;
        return out

    # py::object PyTreeDef::FromIterableTreeHelper(
    #     py::handle xs,
    #     absl::InlinedVector<PyTreeDef::Node, 1>::const_reverse_iterator* it) const {
    def from_iterable_tree_helper(self, xs, it: iter):
        """Recursive helper used to implement from_iterable_tree()"""
        #   if (*it == traversal_.rend()) {
        #     throw std::invalid_argument("Tree structures did not match.");
        #   }
        if finished_iterating(it):
            raise ValueError("Tree structures did not match.")
        #   const Node& node = **it;
        #   ++*it;
        node = next(it)
        #   if (node.kind == PyTreeKind::kLeaf) {
        #     return py::reinterpret_borrow<py::object>(xs);
        #   }
        if node.kind == PyTreeKind.kLeaf:
            return xs
        #   py::iterable iterable = py::reinterpret_borrow<py::iterable>(xs);
        iterable: typing.Iterable = xs
        #   std::vector<py::object> ys;
        #   ys.reserve(node.arity);
        ys = []
        #   for (py::handle x : iterable) {
        #     ys.push_back(py::reinterpret_borrow<py::object>(x));
        #   }
        for x in iterable:
            ys.append(x)
        #   if (ys.size() != node.arity) {
        #     throw std::invalid_argument("Arity mismatch between trees");
        #   }
        if len(ys) != node.arity:
            raise ValueError("Arity mismatch between trees")
        #   for (int j = node.arity - 1; j >= 0; --j) {
        #     ys[j] = FromIterableTreeHelper(ys[j], it);
        #   }
        for j in range(node.arity - 1, -1, -1):
            ys[j] = self.from_iterable_tree_helper(ys[j], it)
        #
        #   return MakeNode(node, absl::MakeSpan(ys));
        return self.make_node(node, ys)
        # }

    #   pybind11::object Walk(const pybind11::function& f_node,
    #                         pybind11::handle f_leaf,
    #                         pybind11::iterable leaves) const;
    def walk(self, f_node: typing.Callable, f_leaf: typing.Callable, leaves: typing.Iterable):
        """Maps a function over a PyTree structure, applying f_leaf to each leaf, and
        f_node to each container node.

        TODO(phawkins): use flattening everywhere instead and delete this method."""
        #   std::vector<py::object> agenda;
        agenda = []
        #   auto it = leaves.begin();
        it = iter(leaves)
        #   for (const Node& node : traversal_) {
        for node in self.traversal_:
            # switch (node.kind) {
            #   case PyTreeKind::kLeaf: {
            if node.kind == PyTreeKind.kLeaf:
                # if (it == leaves.end()) {
                #   throw std::invalid_argument("Too few leaves for PyTreeDef");
                # }
                if finished_iterating(it):
                    raise ValueError("Too few leaves for PyTreeDef")
                #
                # py::object leaf = py::reinterpret_borrow<py::object>(*it);
                leaf = next(it)
                # agenda.push_back(f_leaf.is_none() ? std::move(leaf)
                #                                   : f_leaf(std::move(leaf)));
                agenda.append(f_leaf(leaf) if f_leaf is not None else leaf)
                # ++it;
                # break;
            #   }
            #
            #   case PyTreeKind::kNone:
            #   case PyTreeKind::kTuple:
            #   case PyTreeKind::kNamedTuple:
            #   case PyTreeKind::kList:
            #   case PyTreeKind::kDict:
            #   case PyTreeKind::kCustom: {
            elif node.kind in [PyTreeKind.kNone,
                               PyTreeKind.kTuple,
                               PyTreeKind.kNamedTuple,
                               PyTreeKind.kList,
                               PyTreeKind.kDict,
                               PyTreeKind.kCustom]:
                # if (agenda.size() < node.arity) {
                #   throw std::logic_error("Too few elements for custom type.");
                # }
                assert len(agenda) >= node.arity, "Too few elements for custom type."
                # py::tuple tuple(node.arity);
                tuple = resize(None, node.arity)
                # for (int i = node.arity - 1; i >= 0; --i) {
                #   tuple[i] = agenda.back();
                #   agenda.pop_back();
                # }
                for i in range(node.arity - 1, -1, -1):
                    tuple[i] = agenda.pop()
                # agenda.push_back(f_node(tuple));
                tuple = py.tuple(tuple)
                agenda.append(f_node(tuple) if f_node is not None else tuple)
            #   }
            # }
        #   }
        #   if (it != leaves.end()) {
        #     throw std::invalid_argument("Too many leaves for PyTreeDef");
        #   }
        if not finished_iterating(it):
            raise ValueError("Too many leaves for PyTreeDef")
        #   if (agenda.size() != 1) {
        #     throw std::logic_error("PyTreeDef traversal did not yield a singleton.");
        #   }
        assert len(agenda) == 1, "PyTreeDef traversal did not yield a singleton."
        #   return std::move(agenda.back());
        return agenda[-1]

    # py::object PyTreeDef::FromIterableTree(py::handle xs) const {
    def from_iterable_tree(self, xs):
        """Given a tree of iterables with the same node/leaf structure as this PyTree,
        build the corresponding PyTree.

        TODO(phawkins): use flattening everywhere instead and delete this method."""
        #   auto it = traversal_.rbegin();
        it = iter(self.traversal_[::-1])
        #   py::object out = FromIterableTreeHelper(xs, &it);
        out = self.from_iterable_tree_helper(xs, it)
        #   if (it != traversal_.rend()) {
        #     throw std::invalid_argument("Tree structures did not match.");
        #   }
        if not finished_iterating(it):
            raise ValueError("Tree structures did not match.")
        #   return out;
        return out
        # }


def resize(l, i):
    if l is None:
        return [None for _ in range(i)]
    n = len(l)
    i = max(i, 0)
    if n < i:
        # for _ in range(i - n):
        #     l.append(None)
        l.extend(range(i - n))
    elif n > i:
        for _ in range(n - i):
            l.pop()
    # assert len(l) == i
    return l


def length_hint(obj: typing.Any, default=0):
    """Return an estimate of the number of items in obj.

    This is useful for presizing containers when building from an
    iterable.

    If the object supports len(), the result will be
    exact. Otherwise, it may over- or under-estimate by an
    arbitrary amount. The result will be an integer >= 0.
    """
    try:
        return len(obj)
    except TypeError:
        try:
            get_hint = type(obj).__length_hint__
        except AttributeError:
            return default
        try:
            hint = get_hint(obj)
        except TypeError:
            return default
        if hint is NotImplemented:
            return default
        if not isinstance(hint, int):
            raise TypeError("Length hint must be an integer, not %r" %
                            type(hint))
        if hint < 0:
            raise ValueError("__length_hint__() should return >= 0")
        return hint


def safe_zip(*args):
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, 'length mismatch: {}'.format(list(map(len, args)))
    return list(zip(*args))


def tree_flatten(tree: typing.Any, is_leaf: Optional[Callable[[Any], bool]] = None) -> (typing.Iterable, PyTreeDef):
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
    return PyTreeDef.flatten(tree, is_leaf)


def tree_unflatten(treedef: PyTreeDef, leaves: typing.Iterable):
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


def tree_leaves(tree: typing.Any) -> typing.Iterable:
    """Gets the leaves of a pytree."""
    return tree_flatten(tree)[0]


def tree_structure(tree: typing.Any) -> PyTreeDef:
    """Gets the treedef for a pytree."""
    return tree_flatten(tree)[1]


def treedef_tuple(treedefs: typing.Iterable[PyTreeDef]) -> PyTreeDef:
    """Makes a tuple treedef from a list of child treedefs."""
    return PyTreeDef.tuple(treedefs)


def treedef_children(treedef: PyTreeDef) -> typing.Iterable[PyTreeDef]:
    return treedef.children()


def treedef_is_leaf(treedef: PyTreeDef) -> py.bool:
    return treedef.num_nodes == 1


def register_pytree_node(nodetype: typing.Type, flatten_func: typing.Callable, unflatten_func: typing.Callable):
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


def register_pytree_node_class(cls: typing.Type):
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


def tree_map(f: typing.Callable, tree: typing.Any) -> PyTreeDef:
    """Maps a function over a pytree to produce a new pytree.

    Args:
      f: function to be applied at each leaf.
      tree: a pytree to be mapped over.

    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x)` where `x` is the value at the corresponding leaf in
      `tree`.
    """
    leaves, treedef = tree_flatten(tree)
    return treedef.unflatten(py.list(map(f, leaves)))


def tree_multimap(f: typing.Callable, tree: typing.Any, *rest: PyTreeDef,
                  is_leaf: Optional[Callable[[Any], bool]] = None) -> typing.Any:
    """Maps a multi-input function over pytree args to produce a new pytree.

    Args:
      f: function that takes `1 + len(rest)` arguments, to be applied at the
        corresponding leaves of the pytrees.
      tree: a pytree to be mapped over, with each leaf providing the first
        positional argument to `f`.
      *rest: a tuple of pytrees, each of which has the same structure as tree or
        or has tree as a prefix.
      is_leaf: an optionally specified function that will be called at each
        flattening step. It should return a boolean, which indicates whether
        the flattening should traverse the current object, or if it should be
        stopped immediately, with the whole subtree being treated as a leaf.
    Returns:
      A new pytree with the same structure as `tree` but with the value at each
      leaf given by `f(x, *xs)` where `x` is the value at the corresponding leaf
      in `tree` and `xs` is the tuple of values at corresponding nodes in
      `rest`.
    """
    leaves, treedef = tree_flatten(tree, is_leaf)
    all_leaves = [leaves] + [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


# TODO(mattjj,phawkins): consider removing this function
def _process_pytree(process_node: typing.Callable, tree: typing.Any):
  leaves, treedef = tree_flatten(tree)
  return treedef.walk(process_node, None, leaves), treedef


def build_tree(treedef: PyTreeDef, xs: typing.Any):
  return treedef.from_iterable_tree(xs)


def tree_transpose_old(outer_treedef: PyTreeDef, inner_treedef: PyTreeDef, pytree_to_transpose: typing.Any):
    flat, treedef = tree_flatten(pytree_to_transpose)
    expected_treedef = outer_treedef.compose(inner_treedef)
    if treedef != expected_treedef:
        raise TypeError("Mismatch\n{}\n != \n{}".format(treedef, expected_treedef))
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    flat_it = iter(flat)
    lol = [[next(flat_it) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = py.list(zip(*lol))
    subtrees = py.list(map(functools.partial(tree_unflatten, outer_treedef), transposed_lol))
    return tree_unflatten(inner_treedef, subtrees)


def tree_transpose(outer_treedef, inner_treedef, pytree_to_transpose):
    flat, treedef = tree_flatten(pytree_to_transpose)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        expected_treedef = outer_treedef.compose(inner_treedef)
        raise TypeError(f"Mismatch\n{treedef}\n != \n{expected_treedef}")
    flat = py.iter(flat)
    lol = [[py.next(flat) for _ in range(inner_size)] for __ in range(outer_size)]
    transposed_lol = py.list(zip(*lol))
    subtrees = py.list(map(partial(tree_unflatten, outer_treedef), transposed_lol))
    return tree_unflatten(inner_treedef, subtrees)


no_initializer = object()

@typing.overload
def tree_reduce(function: typing.Callable[[T, typing.Any], T],
                tree: typing.Any) -> T:
    ...

@typing.overload
def tree_reduce(function: typing.Callable[[T, typing.Any], T],
                tree: typing.Any,
                initializer: T) -> T:
    ...


def tree_reduce(function: typing.Callable[[T, typing.Any], T],
                tree: typing.Any,
                initializer: typing.Any = no_initializer) -> T:
  if initializer is no_initializer:
    return functools.reduce(function, tree_leaves(tree))
  else:
    return functools.reduce(function, tree_leaves(tree), initializer)


def tree_all(tree: typing.Any):
    return py.all(tree_leaves(tree))


def all_leaves(tree: typing.Iterable):
    return PyTreeDef.all_leaves(tree)


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

if __name__ == '__main__':
    import sys, os
    tree_util = sys.modules[__name__]
    print(os.getcwd())
    sys.path += [os.path.realpath(os.path.join(os.getcwd(), '..'))]
    from tests import test_pytreez as test_util
    # test_pytreez.test_standard()
    for tree in test_util.TREES:
        test_util.testTranspose(tree)
    case = test_util.TestCase()
    for arg in (
            (tree_util.Partial(test_util._dummy_func),),
            (tree_util.Partial(test_util._dummy_func, 1, 2),),
            (tree_util.Partial(test_util._dummy_func, x="a"),),
            (tree_util.Partial(test_util._dummy_func, 1, 2, 3, x=4, y=5),),
    ):
        fn = case.testRoundtripPartial
        fn = getattr(fn, '__wrapped', fn)
        fn(*arg)
    special = test_util.Special(2., 3.)
    leaves, treedef = tree_util.tree_flatten(special)
    foo = str(treedef)
    print(foo)