import pytest

from main import get_deep_dependencies


@pytest.mark.parametrize("dependencies,result", [
    ({"a": ["b", "c"]}, {"b", "c"}),
    ({"a": ["b; extra == \"x\"", "c"]}, {"b", "c"}),
    ({"a": ["b; extra != \"x\"", "c"]}, {"b", "c"}),
    ({"a": ["b"], "b": ["c; extra == \"x\""]}, {"b"}),
    ({"a": ["b[x]"], "b": ["c; extra == \"y\""]}, {"b"}),
    ({"a": ["b[x]"], "b": ["c; extra == \"x\""]}, {"b", "c"}),
    ({"a": ["b[x]"], "b": ["c[y]; extra == \"x\""], "c": ["d; extra == \"y\""]}, {"b", "c", "d"}),
    ({"a": ["b"], "b": ["c; extra != \"x\""]}, {"b", "c"}),
    ({"a": ["b[x]"], "b": ["c; extra != \"y\""]}, {"b", "c"}),
    ({"a": ["b[x]"], "b": ["c; extra != \"x\""]}, {"b"}),
    ({"a": ["b", "git+https://x.y"]}, {"b", None}),
    ({"a": ["a[x]", "b"]}, {"a", "b"}),
])
def test_deep_dependency_resolver(dependencies, result):
    assert get_deep_dependencies(dependencies, "a") == result
