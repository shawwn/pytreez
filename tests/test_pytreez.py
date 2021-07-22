from __future__ import annotations
import pytreez as tree_util
import collections
import re
import pytest


Foo = collections.namedtuple('Foo', 'bar')
ATuple = collections.namedtuple("ATuple", ("foo", "bar"))

class ANamedTupleSubclass(ATuple):
  pass

class AnObject(object):

  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y and self.z == other.z

  def __hash__(self):
    return hash((self.x, self.y, self.z))

  def __repr__(self):
    return "AnObject({},{},{})".format(self.x, self.y, self.z)

tree_util.register_pytree_node(AnObject, lambda o: ((o.x, o.y), o.z),
                               lambda z, xy: AnObject(xy[0], xy[1], z))

@tree_util.register_pytree_node_class
class Special:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __repr__(self):
    return "Special(x={}, y={})".format(self.x, self.y)

  def tree_flatten(self):
    return ((self.x, self.y), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    return cls(*children)

  def __eq__(self, other):
    return type(self) is type(other) and (self.x, self.y) == (other.x, other.y)

@tree_util.register_pytree_node_class
class FlatCache:
  def __init__(self, structured, *, leaves=None, treedef=None):
    if treedef is None:
      leaves, treedef = tree_util.tree_flatten(structured)
    self._structured = structured
    self.treedef = treedef
    self.leaves = leaves

  def __hash__(self):
    return hash(self.structured)

  def __eq__(self, other):
    return self.structured == other.structured

  def __repr__(self):
    return f"FlatCache({self.structured!r})"

  @property
  def structured(self):
    if self._structured is None:
      self._structured = tree_util.tree_unflatten(self.treedef, self.leaves)
    return self._structured

  def tree_flatten(self):
    return self.leaves, self.treedef

  @classmethod
  def tree_unflatten(cls, meta, data):
    if not tree_util.all_leaves(data):
      data, meta = tree_util.tree_flatten(tree_util.tree_unflatten(meta, data))
    return FlatCache(None, leaves=data, treedef=meta)

TREES = (
    (None,),
    ((None,),),
    ((),),
    (([()]),),
    ((1, 2),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    ([3, ATuple(foo=(3, ATuple(foo=3, bar=None)), bar={"baz": 34})],),
    ([AnObject(3, None, [4, "foo"])],),
    (Special(2, 3.),),
    ({"a": 1, "b": 2},),
    (collections.OrderedDict([("foo", 34), ("baz", 101), ("something", -42)]),),
    (collections.defaultdict(dict,
                             [("foo", 34), ("baz", 101), ("something", -42)]),),
    (ANamedTupleSubclass(foo="hello", bar=3.5),),
    (FlatCache(None),),
    (FlatCache(1),),
    (FlatCache({"a": [1, 2]}),),
)


TREE_STRINGS = (
    "PyTreeDef(None)",
    "PyTreeDef((None,))",
    "PyTreeDef(())",
    "PyTreeDef([()])",
    "PyTreeDef((*, *))",
    "PyTreeDef(((*, *), [*, (*, None, *)]))",
    "PyTreeDef([*])",
    "PyTreeDef([*, CustomNode(namedtuple[<class '__main__.ATuple'>], [(*, "
    "CustomNode(namedtuple[<class '__main__.ATuple'>], [*, None])), {'baz': "
    "*}])])",
    "PyTreeDef([CustomNode(<class '__main__.AnObject'>[[4, 'foo']], [*, None])])",
    "PyTreeDef(CustomNode(<class '__main__.Special'>[None], [*, *]))",
    "PyTreeDef({'a': *, 'b': *})",
)

# pytest expects "tree_util_test.ATuple"
STRS = []
for tree_str in TREE_STRINGS:
    tree_str = re.escape(tree_str)
    tree_str = tree_str.replace("__main__", ".*")
    STRS.append(tree_str)
TREE_STRINGS = STRS

LEAVES = (
    ("foo",),
    (0.1,),
    (1,),
    (object(),),
)


def test_basic():
    print('')
    print('----')
    assert tree_util.PyTreeTypeRegistry.lookup(list) is not None
    assert tree_util.PyTreeDef.get_kind([1,2,3])[0] == tree_util.PyTreeKind.kList
    assert tree_util.PyTreeDef.get_kind({'a': 1})[0] == tree_util.PyTreeKind.kDict
    assert tree_util.PyTreeDef.get_kind((1,2,3))[0] == tree_util.PyTreeKind.kTuple
    assert tree_util.PyTreeDef.get_kind(None)[0] == tree_util.PyTreeKind.kNone
    assert tree_util.PyTreeDef.get_kind(1)[0] == tree_util.PyTreeKind.kLeaf
    assert tree_util.PyTreeDef.get_kind(Foo(42))[0] == tree_util.PyTreeKind.kNamedTuple
    assert tree_util.PyTreeDef.all_leaves([1,2,3])
    assert not tree_util.PyTreeDef.all_leaves([1,2,[3]])
    leaves, td = tree_util.PyTreeDef.flatten({'a': [1,2,3]})
    leaves2, td2 = tree_util.PyTreeDef.flatten({'a': [1,2,3]})
    assert td == td2
    assert leaves == leaves2


def test_custom_register():
    class Box:
        def __init__(self, data):
            self.data = data
    tree_util.PyTreeTypeRegistry.register(Box, lambda self: (self.data, None), lambda meta, leaves: Box(leaves))
    leaves, td = tree_util.PyTreeDef.flatten({'box': Box([1,2,Box([3,4])])})
    print(leaves)
    print(td)


def check(tree):
    leaves, td = tree_util.tree_flatten(tree)
    tree2 = td.unflatten(leaves)
    print(tree2)
    assert tree_util.tree_leaves(tree) == tree_util.tree_leaves(tree2)
    assert tree == tree2
    return leaves, td

@tree_util.register_pytree_node_class
class Box:
    def __init__(self, data):
        self.data = data
    def tree_flatten(self):
        leaves, treedef = tree_util.tree_flatten(self.data)
        return (leaves, treedef)
    @classmethod
    def tree_unflatten(cls, treedef: tree_util.PyTreeDef, leaves):
        print('tree_unflatten', leaves, treedef)
        data = treedef.unflatten(leaves)
        return cls(data)
    def __eq__(self, other: Box):
        if not isinstance(other, self.__class__):
            return False
        self_leaves, self_treedef = self.tree_flatten()
        other_leaves, other_treedef = other.tree_flatten()
        if self_treedef != other_treedef:
            return False
        a = self_leaves
        b = other_leaves
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            x, y = a[i], b[i]
            if x != y:
                return False
        return True
    def __str__(self):
        return str(self.data)
    def __repr__(self):
        return self.__class__.__module__ + '.' + self.__class__.__name__ + '(' + str(self) + ')'

def test_register_pytree_node_class():
    check([1,2,(3,4,{'a': 42})])
    check(Box([1,2,(3,4,{'a': 42, 'b': [1,Box([2,{'c': Box(3)}])]})]))

def test_flatten_up_to():
    _, tree = tree_util.tree_flatten([(1, 2), None, ATuple(foo=3, bar=7)])
    out = tree.flatten_up_to([({
        "foo": 7
    }, (3, 4)), None, ATuple(foo=(11, 9), bar=None)])
    assert out == [{"foo": 7}, (3, 4), (11, 9), None]


@pytest.mark.parametrize("inputs", (TREES + LEAVES))
def testRoundtrip(inputs):
    xs, tree = tree_util.tree_flatten(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    assert actual == inputs


@pytest.mark.parametrize("inputs", (TREES + LEAVES))
def testRoundtripWithFlattenUpTo(inputs):
    _, tree = tree_util.tree_flatten(inputs)
    xs = tree.flatten_up_to(inputs)
    actual = tree_util.tree_unflatten(tree, xs)
    assert actual == inputs


@pytest.mark.parametrize("tree", TREES)
def testTranspose(tree):
    outer_treedef = tree_util.tree_structure(tree)
    if not outer_treedef.num_leaves:
        #self.skipTest("Skipping empty tree")
        return
    inner_treedef = tree_util.tree_structure([1, 1, 1])
    nested = tree_util.tree_map(lambda x: [x, x, x], tree)
    actual = tree_util.tree_transpose(outer_treedef, inner_treedef, nested)
    assert actual == [tree, tree, tree]


if __name__ == '__main__':
    test_standard()