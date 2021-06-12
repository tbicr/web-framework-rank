import pytest

from main import get_csv_data, get_deep_dependencies, FIELDS, RANK_SUFFIX


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
    ({"a": ["b", "git+https://x.y"]}, {"b"}),
    ({"a": ["a[x]", "b"]}, {"a", "b"}),
])
def test_deep_dependency_resolver(dependencies, result):
    assert get_deep_dependencies(dependencies, "a") == result


def test_get_csv_data_back_compatibility():
    with open("data.csv") as handler:
        content = handler.read()
    data = get_csv_data(content)
    for project_data in data:
        assert sorted(
           field for field in project_data.keys() if not field.endswith(RANK_SUFFIX)
        ) == sorted(FIELDS)
