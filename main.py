import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from html.parser import HTMLParser
from itertools import chain
from typing import Dict, List, Union, Tuple, Optional, Set, Any

import aiohttp
import git
import requests
from github import Github, InputGitAuthor
from pytz import UTC


GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH")
GITHUB_AUTHOR_NAME = os.environ.get("GITHUB_AUTHOR_NAME")
GITHUB_AUTHOR_EMAIL = os.environ.get("GITHUB_AUTHOR_EMAIL")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

WEEK_DAYS = 7

# + and - in metric means use it or not for rank/score calculation
REPO_LINES = "-repo_lines"
REPO_SIZE = "-repo_size"
REPO_COMMITS = "-repo_commits"
REPO_COMMITTERS = "+repo_unique_committers"
REPO_FIRST_COMMIT_AT = "-repo_first_commit"
REPO_LAST_COMMIT_AT = "+repo_last_commit"

REPO_CHANGED_LINES_LAST_MONTH = "+repo_changed_lines_last_month"
REPO_COMMITS_LAST_MONTH = "-repo_commits_last_month"
REPO_COMMITTERS_LAST_MONTH = "+repo_unique_committers_last_month"

GITHUB_STARS = "+github_stars"
GITHUB_WATCHES = "-github_watches"
GITHUB_FORKS = "-github_forks"
GITHUB_SIZE = "-github_size"
GITHUB_CREATED_AT = "-github_first_commit"
GITHUB_UPDATED_AT = "-github_last_commit"

GITHUB_TOPICS_ALL = "-github_topics_all"
GITHUB_TOPICS_LANGUAGE = "-github_topics_language"

STACKOVERFLOW_QUESTIONS = "+stackoverflow_questions"

PYPI_PROJECT_MENTIONS = "-pypi_project_mentions"
PYPI_USED_AS_MAIN_DEPENDENCY = "+pypi_used_as_main_dependency"
PYPI_USED_AS_MAIN_DEPENDENCY_WITH_EXTRA = "-pypi_used_as_main_dependency_with_extra"
PYPI_USED_AS_DEEP_DEPENDENCY = "-pypi_used_as_deep_dependency"
PYPI_USED_AS_DEEP_DEPENDENCY_WITH_EXTRA = "-pypi_used_as_deep_dependency_with_extra"
PYPI_RELEASES = "-pypi_releases"
PYPI_LAST_RELEASE_AT = "-pypi_last_release"

PYPISTATS_DOWNLOADS_LAST_MONTH = "+pypistats_downloads_last_month"

USES_FRAMEWORKS = "uses"
USED_IN_FRAMEWORKS = "used_in"

NAME = "name"
COLLECTED_AT = "-collected_at"
RANK = "rank"
SCORE = "score"

RANK_SUFFIX = "__rank"

PROJECT_LANGUAGE = "language"
PROJECT_REPO = "repo"
PROJECT_GIT = "git"
PROJECT_GITHUB_REPO = "github_repo"
PROJECT_GITHUB_TOPIC = "github_topic"
PROJECT_STACKOVERFLOW_TAG = "stackoverflow_tag"
PROJECT_PYPISTAT_PROJECT = "pypistat_project"
PROJECT_PYPI_PROJECT = "pypi_project"

DATE_FIELDS = {
    REPO_FIRST_COMMIT_AT,
    REPO_LAST_COMMIT_AT,
    GITHUB_CREATED_AT,
    GITHUB_UPDATED_AT,
    PYPI_LAST_RELEASE_AT,
    COLLECTED_AT,
}

FIELDS = [
    NAME,

    RANK,
    SCORE,

    PYPISTATS_DOWNLOADS_LAST_MONTH,

    PYPI_PROJECT_MENTIONS,
    PYPI_USED_AS_MAIN_DEPENDENCY,
    PYPI_USED_AS_MAIN_DEPENDENCY_WITH_EXTRA,
    PYPI_USED_AS_DEEP_DEPENDENCY,
    PYPI_USED_AS_DEEP_DEPENDENCY_WITH_EXTRA,
    PYPI_RELEASES,
    PYPI_LAST_RELEASE_AT,

    STACKOVERFLOW_QUESTIONS,

    GITHUB_TOPICS_ALL,
    GITHUB_TOPICS_LANGUAGE,

    GITHUB_STARS,
    GITHUB_FORKS,
    GITHUB_WATCHES,
    GITHUB_SIZE,
    GITHUB_CREATED_AT,
    GITHUB_UPDATED_AT,

    REPO_LINES,
    REPO_SIZE,
    REPO_COMMITS,
    REPO_COMMITTERS,
    REPO_CHANGED_LINES_LAST_MONTH,
    REPO_COMMITS_LAST_MONTH,
    REPO_COMMITTERS_LAST_MONTH,
    REPO_FIRST_COMMIT_AT,
    REPO_LAST_COMMIT_AT,

    COLLECTED_AT,
]


logger = logging.getLogger()
logger.setLevel(logging.INFO)

session = requests.session()


@dataclass(frozen=True)
class Project:
    name: str
    language: str
    repo: str
    git: str
    github_repo: str
    github_topic: str
    stackoverflow_tag: str
    pypistat_project: str
    pypi_project: str


class GithubWrapper:
    DATA = "data.csv"
    README = "README.md"
    PYPI_CACHE = "pypi_cache.json"
    PROJECTS = "projects.json"

    def __init__(self, dry_run=False, use_local=False):
        self.dry_run = dry_run
        self.use_local = use_local
        g = Github(GITHUB_TOKEN)
        self.repo = g.get_repo(GITHUB_REPO)
        self.author = InputGitAuthor(GITHUB_AUTHOR_NAME, GITHUB_AUTHOR_EMAIL)
        self._state = {}

    def _fetch_content(self, file_name):
        if self.use_local:
            with open(file_name) as handler:
                return handler.read()
        else:
            tree = self.repo.get_git_tree(GITHUB_BRANCH).tree
            sha = [node.sha for node in tree if node.path == file_name][0]
            self._state[file_name] = self.repo.get_git_blob(sha)
            return base64.b64decode(self._state[file_name].content).decode("utf-8")

    def _update_content(self, file_name, commit_message, content):
        if not self.dry_run:
            if self.use_local:
                with open(file_name, "w") as handler:
                    handler.write(content)
            else:
                self.repo.update_file(
                    file_name, commit_message, content,
                    self._state[file_name].sha, branch=GITHUB_BRANCH, author=self.author)
        self._state = {}

    @property
    def data(self):
        return self._fetch_content(self.DATA)

    @data.setter
    def data(self, content):
        self._update_content(self.DATA, "update data", content)

    @property
    def readme(self):
        return self._fetch_content(self.README)

    @readme.setter
    def readme(self, content):
        self._update_content(self.README, "update readme", content)

    @property
    def pypi_cache(self):
        return self._fetch_content(self.PYPI_CACHE)

    @pypi_cache.setter
    def pypi_cache(self, content):
        self._update_content(self.PYPI_CACHE, "update pypi cache", content)

    @property
    def projects(self):
        return self._fetch_content(self.PROJECTS)


def logged(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        projects = [param for param in chain(args, kwargs.values()) if isinstance(param, Project)]
        project = projects[0] if len(projects) == 1 else None
        start = time.time()
        logger.info(
            f"start {func.__name__}" +
            (f" project: {project.name}" if project is not None else ""))
        result = func(*args, **kwargs)
        logger.info(
            f"end {func.__name__}" +
            (f" project: {project.name}" if project is not None else "") +
            f" duration: {round(time.time() - start, 3)} sec")
        return result
    return wrapper


def retry(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for timeout in [3, 10, 30, None]:
            try:
                return func(*args, **kwargs)
            except Exception as err:
                logger.exception(f"fail {func.__name__}")
                if timeout is None:
                    raise err
                time.sleep(timeout)
    return wrapper


def _commit_datetime(commit: git.objects.commit.Commit) -> datetime:
    return commit.committed_datetime.astimezone(UTC).replace(tzinfo=None)


@retry
@logged
def get_repository_stat(project: Project) -> Dict[str, Union[int, str]]:
    if project.git is None:
        return {
            REPO_LINES: 0,
            REPO_SIZE: 0,
            REPO_COMMITS: 0,
            REPO_COMMITTERS: 0,
            REPO_FIRST_COMMIT_AT: "",
            REPO_LAST_COMMIT_AT: "",

            REPO_CHANGED_LINES_LAST_MONTH: 0,
            REPO_COMMITS_LAST_MONTH: 0,
            REPO_COMMITTERS_LAST_MONTH: 0,
        }

    with tempfile.TemporaryDirectory() as dir:
        repo = git.Repo.clone_from(project.git, dir)
        all_commits = list(repo.iter_commits())
        all_committers = {commit.author.email for commit in all_commits}
        month_ago = datetime.utcnow() - timedelta(days=30)
        last_month_commits = [
            commit for commit in all_commits if _commit_datetime(commit) > month_ago
        ]
        last_month_commiters = {commit.author.email for commit in last_month_commits}
        last_month_lines = sum(commit.stats.total["lines"] for commit in last_month_commits)

        total_size = 0
        total_lines = 0
        git_dir = os.path.join(dir, ".git", "")
        for root, _, files in os.walk(dir):
            for file in files:
                filepath = os.path.join(root, file)
                if filepath.startswith(git_dir):
                    continue
                if os.path.islink(filepath):
                    continue
                total_size += os.path.getsize(filepath)
                with open(filepath, "rb") as handler:
                    total_lines += sum(1 for _ in handler)

    return {
        REPO_LINES: total_lines,
        REPO_SIZE: total_size,
        REPO_COMMITS: len(all_commits),
        REPO_COMMITTERS: len(all_committers),
        REPO_FIRST_COMMIT_AT: _commit_datetime(all_commits[-1]).isoformat(),
        REPO_LAST_COMMIT_AT: _commit_datetime(all_commits[0]).isoformat(),

        REPO_CHANGED_LINES_LAST_MONTH: last_month_lines,
        REPO_COMMITS_LAST_MONTH: len(last_month_commits),
        REPO_COMMITTERS_LAST_MONTH: len(last_month_commiters),
    }


@retry
@logged
def get_github_stat(project: Project) -> Dict[str, Union[int, str]]:
    if project.github_repo is None:
        return {
            GITHUB_SIZE: 0,
            GITHUB_STARS: 0,
            GITHUB_WATCHES: 0,
            GITHUB_FORKS: 0,
            GITHUB_CREATED_AT: "",
            GITHUB_UPDATED_AT: "",
        }
    url = f"https://api.github.com/repos/{project.github_repo}"
    response = session.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
    response.raise_for_status()
    data = response.json()
    return {
        GITHUB_SIZE: data["size"],
        GITHUB_STARS: data["stargazers_count"],
        GITHUB_WATCHES: data["subscribers_count"],
        GITHUB_FORKS: data["forks_count"],
        GITHUB_CREATED_AT: data["created_at"].rstrip("Z"),
        GITHUB_UPDATED_AT: data["updated_at"].rstrip("Z"),
    }


TOPIC_PROJECTS_COUNT_PATTERN = re.compile(r"([0-9,]+) public repositories")


@logged
def get_github_topics(project: Project) -> Dict[str, int]:
    """
    There are no API for getting topic, so this metric not robust there
    """
    count_all = 0
    count_language = 0
    if project.github_topic is not None:
        try:
            url = f"https://github.com/topics/{project.github_topic}"
            response = session.get(url)
            response.raise_for_status()
            match = TOPIC_PROJECTS_COUNT_PATTERN.search(response.text)
            if match is not None:
                count_all = int(match.groups()[0].replace(',', ''))

            url = f"https://github.com/topics/{project.github_topic}?l={project.language}"
            response = session.get(url)
            response.raise_for_status()
            match = TOPIC_PROJECTS_COUNT_PATTERN.search(response.text)
            if match is not None:
                count_language = int(match.groups()[0].replace(',', ''))
        except Exception:
            pass

    return {GITHUB_TOPICS_ALL: count_all, GITHUB_TOPICS_LANGUAGE: count_language}


@retry
@logged
def get_stackoverflow_stat(project: Project) -> Dict[str, int]:
    if project.stackoverflow_tag is None:
        return {STACKOVERFLOW_QUESTIONS: 0}
    url = (
        f"https://api.stackexchange.com/2.2/tags/{project.stackoverflow_tag}/info"
        f"?order=desc&sort=popular&site=stackoverflow"
    )
    response = session.get(url)
    response.raise_for_status()
    data = response.json()
    return {STACKOVERFLOW_QUESTIONS: data["items"][0]["count"]}


@retry
@logged
def get_pypistats_stat(project: Project) -> Dict[str, int]:
    url = f"https://pypistats.org/api/packages/{project.pypistat_project}/recent"
    response = session.get(url)
    response.raise_for_status()
    data = response.json()
    return {PYPISTATS_DOWNLOADS_LAST_MONTH: data["data"]["last_month"]}


class SimplePypiIndexHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packages: List[str] = []

    def handle_data(self, data: str):
        data = data.strip()
        if data:
            self.packages.append(data)


@retry
@logged
def get_pypi_index() -> List[str]:
    url = "https://pypi.python.org/simple/"
    data = session.get(url).text
    parser = SimplePypiIndexHTMLParser()
    parser.feed(data)
    return parser.packages


@retry
def _get_pypi_package_meta(package: str) -> Dict:
    response = session.get(f"https://pypi.org/pypi/{package}/json")
    response.raise_for_status()
    return response.json()


extra_pattern = re.compile(r"^extra *(==|!=) *['\"]([0-9a-z-_. ]+)['\"]$")
split_pattern = re.compile(r"[ <!=~>;]")
dependency_pattern = re.compile(
    r"^([0-9a-z-_.]+)(\[([0-9a-z-_.,]+)\])?(; extra (==|!=) \"([0-9a-z-_. ]+)\")?$"
)


def _prettify_dependency(dependency: str) -> str:
    """
    keep only `dependency[option1,...]; extra == "package_option"` dependency pattern
    versions ignored
    next parameters ignored:
    - os_name, sys_platform, platform_system, platform_release, platform_machine
    - implementation_name, platform_python_implementation
    - python_version, python_full_version
    """
    dependency = dependency.replace("(", "").replace(")", "").lower()
    package = split_pattern.split(dependency, maxsplit=1)[0]
    extra = ""
    parts = [
        p3.strip()
        for p1 in dependency.split(";")
        for p2 in p1.split(" and ")
        for p3 in p2.split(" or ")
    ]
    for part in parts:
        match = extra_pattern.match(part)
        if match is not None:
            eq, name = match.groups()
            extra = f"; extra {eq} \"{name}\""

    return package + extra


def _parse_dependency(
        dependency: str
) -> Tuple[Optional[str], List[str], Optional[bool], Optional[str]]:
    if dependency.startswith("git+https://"):
        return None, [], None, None
    match = dependency_pattern.match(dependency)
    if match is None:
        logger.warning(f"dependency in wrong format {dependency}")
        return None, [], None, None
    dependency, _, options, _, eq, extra = match.groups()
    return (
        dependency,
        options.split(",") if options is not None else [],
        eq == "==" if eq is not None else None,
        extra,
    )


async def _get_pypi_package_dependencies(
        session: aiohttp.ClientSession,
        package: str,
) -> Tuple[str, List[str]]:
    async with session.get(f"https://pypi.org/pypi/{package}/json") as response:
        if response.status == 404:
            return package.lower(), []
        response.raise_for_status()
        meta = await response.json()
        if meta is None:
            return package.lower(), []
        return package.lower(), sorted({
            _prettify_dependency(dependency) for dependency in meta["info"]["requires_dist"] or []
        } - {package})


async def _concurrent_pypi_dependencies_fetching(
        packages: List[str],
) -> List[Tuple[str, List[str]]]:
    async with aiohttp.ClientSession() as session:
        tasks = [_get_pypi_package_dependencies(session, package) for package in packages]
        results = await asyncio.gather(*tasks)
    return results


@retry
@logged
def get_and_update_dependencies(
        pypi_index: List[str],
        repo_github: GithubWrapper,
) -> Dict[str, List[str]]:
    data = json.loads(repo_github.pypi_cache)
    uncached = list({package for package in pypi_index if package.lower() not in data})
    logger.info(f"pypi packages for download: {len(uncached)} total: {len(pypi_index)}")
    chunk_size = 10000
    for i in range(0, len(uncached), chunk_size):
        chunk = uncached[i:i + chunk_size]
        for package, dependencies in asyncio.run(_concurrent_pypi_dependencies_fetching(chunk)):
            data[package] = dependencies
        logger.info(f"pypi packages chunk downloaded: {i + len(chunk)} off {len(uncached)}")
    repo_github.pypi_cache = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False).
                              replace("[\n    ", "[").replace("\n   ", "").replace("\n  ]", "]"))
    return data


def get_deep_dependencies(
        data: Dict[str, List[str]],
        package: str,
        extra_options: Optional[List[str]] = None,
        cache: Optional[Set[str]] = None,
) -> Set[str]:
    all_extras = False
    extra_options = extra_options or []
    if cache is None:
        all_extras = True
        extra_options = []
        cache = set()
    for dependency in data.get(package, []):
        parsed_dependency, options, eq, extra = _parse_dependency(dependency)
        if parsed_dependency is None:
            continue
        if parsed_dependency in cache:
            continue
        if not all_extras:
            if extra is not None and eq is True and extra not in extra_options:
                continue
            if eq is False and extra in extra_options:
                continue
        cache.add(parsed_dependency)
        get_deep_dependencies(data, parsed_dependency, options, cache)
    return cache


@logged
def get_deep_dependencies_count(data: Dict[str, List[str]]) -> Dict[str, int]:
    return Counter(
        dependency
        for package in data
        for dependency in get_deep_dependencies(data, package, [], set()) - {package}
    )


@logged
def get_deep_dependencies_count_with_extra(data: Dict[str, List[str]]) -> Dict[str, int]:
    return Counter(
        dependency
        for package in data
        for dependency in get_deep_dependencies(data, package) - {package}
    )


@logged
def get_main_dependencies_count(data: Dict[str, List[str]]) -> Dict[Optional[str], int]:
    return Counter(
        _parse_dependency(dependency)[0]
        for package, dependencies in data.items()
        for dependency in dependencies
        if _parse_dependency(dependency)[0] != package
        and _parse_dependency(dependency)[3] is None
    )


@logged
def get_main_dependencies_count_with_extra(data: Dict[str, List[str]]) -> Dict[Optional[str], int]:
    return Counter(
        _parse_dependency(dependency)[0]
        for package, dependencies in data.items()
        for dependency in dependencies
        if _parse_dependency(dependency)[0] != package
    )


@retry
@logged
def get_pypi_projects_stat(
        project: Project,
        pypi_index: List[str],
        main_dependencies_count: Dict[str, int],
        main_dependencies_count_with_extra: Dict[str, int],
        deep_dependencies_count: Dict[str, int],
        deep_dependencies_count_with_extra: Dict[str, int],
) -> Dict[str, Union[int, str]]:
    project_name = project.pypi_project.lower()

    mentions_count = 0
    for name in pypi_index:
        if project_name in name.lower():
            mentions_count += 1

    meta = _get_pypi_package_meta(project.pypi_project)
    releases_count = len(meta["releases"])
    last_release_at = max((
        bundle["upload_time"]
        for release, bundles in meta["releases"].items()
        for bundle in bundles
    ), default="")

    return {
        PYPI_PROJECT_MENTIONS: mentions_count,
        PYPI_USED_AS_MAIN_DEPENDENCY: main_dependencies_count[project_name],
        PYPI_USED_AS_MAIN_DEPENDENCY_WITH_EXTRA: main_dependencies_count_with_extra[project_name],
        PYPI_USED_AS_DEEP_DEPENDENCY: deep_dependencies_count[project_name],
        PYPI_USED_AS_DEEP_DEPENDENCY_WITH_EXTRA: deep_dependencies_count_with_extra[project_name],
        PYPI_RELEASES: releases_count,
        PYPI_LAST_RELEASE_AT: last_release_at,
    }


def get_usages(
        projects: List[Project],
        dependencies: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    project_pypi_name_mapping = {project.pypi_project: project.name for project in projects}
    uses = defaultdict(list)
    for project in projects:
        project_dependencies = (
            get_deep_dependencies(dependencies, project.pypi_project, [], set()) -
            {project.pypi_project}
        )
        for dependency in project_dependencies:
            if dependency in project_pypi_name_mapping:
                uses[project.name].append(project_pypi_name_mapping[dependency])
    return uses


def get_uses_and_used_by(project: Project, usages: Dict[str, List[str]]) -> Dict[str, List[str]]:
    uses = sorted(usages[project.name])
    used_by = sorted({
        subproject
        for subproject, dependencies in usages.items()
        for dependency in dependencies
        if dependency == project.name
    })
    return {USES_FRAMEWORKS: uses, USED_IN_FRAMEWORKS: used_by}


@logged
def get_rank_and_score(
        projects_data: List[Dict[str, Any]]
) -> Dict[str, Dict[str, int]]:
    fields = {
        field for data in projects_data for field in data.keys()
        if field.startswith("+") and field in FIELDS
    }
    scores: Dict[str, float] = defaultdict(float)
    field_scores = defaultdict(int)
    for field in fields:
        ordered_values = defaultdict(list)
        for data in projects_data:
            name = data[NAME]
            value = data[field]
            if field in DATE_FIELDS:
                if value:
                    date_from = datetime.fromisoformat(value)
                    date_to = datetime.fromisoformat(data[COLLECTED_AT])
                    value = -(date_to - date_from).days // WEEK_DAYS
                else:
                    value = None
            ordered_values[value].append(name)
        current_field_score = len(projects_data)
        if None in ordered_values:
            names = ordered_values.pop(None)
            min_value = min(ordered_values.keys())
            ordered_values[min_value].extend(names)
        for value, names in sorted(ordered_values.items(), reverse=True):
            for name in names:
                scores[name] += current_field_score - sum(range(len(names))) / len(names)
                field_scores[(name, field)] = current_field_score
            current_field_score -= len(names)
    results = {}
    for index, (name, score) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]), start=1):
        results[name] = {
            RANK: index,
            SCORE: round(100 * score / len(fields) / len(projects_data)),
            RANK + RANK_SUFFIX: index,  # small hack for rank tooltip rendering
            SCORE + RANK_SUFFIX: index,  # small hack for score tooltip rendering
        }
    for (name, field), score in field_scores.items():
        results[name][field + RANK_SUFFIX] = len(projects_data) - score + 1
    return results


def rank_and_update(projects_data: List[Dict[str, Any]]):
    rank_and_score = get_rank_and_score(projects_data)
    for data in projects_data:
        data.update(rank_and_score[data[NAME]])


def get_csv_data(content: str) -> List[Dict[str, Any]]:
    handler = io.StringIO(content)
    reader = csv.DictReader(handler, lineterminator="\n")
    name_project_data_mapping = {}
    for data in reader:
        name_project_data_mapping[data[NAME]] = {
            field: (
                int(data.get(field.strip("+-"), "0"))
                if field not in DATE_FIELDS | {NAME} else
                data.get(field.strip("+-"), "")
            )
            for field in FIELDS
        }
    project_data = list(name_project_data_mapping.values())
    rank_and_update(project_data)
    return project_data


def update_csv(result: List[Dict[str, Any]], content: str) -> str:
    handler = io.StringIO(content)
    handler.read()
    writer = csv.DictWriter(handler, fieldnames=FIELDS, lineterminator="\n")
    for data in sorted(result, key=lambda data: data[RANK]):
        writer.writerow({field: value for field, value in data.items() if field in FIELDS})
    return handler.getvalue()


def readme_table_field(
        field: str,
        data: Dict[str, Any],
        prev_data: Dict[str, Any],
):
    value = data[field]
    prev_value = prev_data.get(field)
    rank = data.get(field + RANK_SUFFIX)
    prev_rank = prev_data.get(field + RANK_SUFFIX)

    change_period = "last week"

    rank_change = " "
    if prev_rank is None or rank < prev_rank:
        rank_change = "▲"
    elif rank > prev_rank:
        rank_change = "▼"

    if field == NAME:
        repo = data[PROJECT_REPO]
        tooltip = "; ".join(f"{name}: {value}" for name, value in {
            "first commit": (
                data[REPO_FIRST_COMMIT_AT].split("T")[0] if data[REPO_FIRST_COMMIT_AT] else None
            ),
            "uses": " and ".join(data[USES_FRAMEWORKS]),
            "used by": " and ".join(data[USED_IN_FRAMEWORKS]),
        }.items() if value)
        return f"[<sub>{value}</sub>]({repo} \"{tooltip}\")"
    elif field == RANK:
        change = f"{prev_value - value:+}" if prev_value else "new"
        tooltip = f"{rank_change} {change} {change_period}"
        return f"[<sub>{value}</sub>](# \"{tooltip}\")"
    elif field == SCORE:
        change = f"{value - prev_value:+}" if prev_value else f"{value:+}"
        tooltip = f"{rank_change} {change} {change_period}"
        return f"[<sub>{value}</sub>](# \"{tooltip}\")"
    else:
        if field in DATE_FIELDS:
            if value:
                date_from = datetime.fromisoformat(value)
                date_to = datetime.fromisoformat(data[COLLECTED_AT])
                weeks = (date_to - date_from).days // WEEK_DAYS
                change = f"{weeks + 1} weeks ago" if weeks else "1 week ago"
            else:
                change = ""
            value = data[field].split("T")[0]
        else:
            change = (
                f"{round(100 * (value - prev_value) / prev_value, 2):+}% {change_period}"
                if prev_value else f"+100% {change_period}"
            )
        field_name = field.lstrip("+-").replace("_", " ")
        tooltip = f"{rank_change} #{rank} in {field_name} {change}".rstrip()
        return f"[<sub>{value}</sub>](# \"{tooltip}\")"


def update_readme(
        result: List[Dict[str, Any]],
        prev_result: List[Dict[str, Any]],
        content: str,
) -> str:
    top_part = content.split("---", 1)[0].rsplit("\n", 2)[0]
    bottom_part = content.split("---", 1)[1].split("\n\n", 1)[1]
    header = " | ".join(
        "<sub>" + field.lstrip("+-").replace("_", " ") + "</sub>" for field in FIELDS
        if not field.startswith("-") and field in result[0]
    )
    splitter = " | ".join(
        ":---" if field == NAME else "---:" for field in FIELDS
        if not field.startswith("-") and field in result[0])
    name_prev_data_mapping = {data[NAME]: data for data in prev_result}
    table = [header, splitter] + [
        " | ".join(
            readme_table_field(field, data, name_prev_data_mapping.get(data[NAME], {}))
            for field in FIELDS
            if not field.startswith("-") and field in data
        )
        for data in sorted(result, key=lambda data: data[RANK])
    ]
    return top_part + "\n" + "\n".join(table) + "\n\n" + bottom_part


@retry
@logged
def update_repo(result: List[Dict[str, Any]], repo_github: GithubWrapper):
    prev_result = get_csv_data(repo_github.data)
    repo_github.readme = update_readme(result, prev_result, repo_github.readme)
    repo_github.data = update_csv(result, repo_github.data)


@logged
def lambda_handler(event, context):
    dry_run = event.get("dry_run", False)
    use_local = event.get("use_local", False)
    repo_github = GithubWrapper(dry_run, use_local)
    projects = json.loads(repo_github.projects)
    pypi_index = get_pypi_index()
    dependencies = get_and_update_dependencies(pypi_index, repo_github)
    main_dependencies_count = get_main_dependencies_count(dependencies)
    main_dependencies_count_with_extra = get_main_dependencies_count_with_extra(dependencies)
    deep_dependencies_count = get_deep_dependencies_count(dependencies)
    deep_dependencies_count_with_extra = get_deep_dependencies_count_with_extra(dependencies)
    uses = get_usages([Project(**project_dict) for project_dict in projects], dependencies)
    collected_at = datetime.utcnow().isoformat().split(".")[0].rstrip("Z")
    projects_data = []
    for project_dict in projects:
        project = Project(**project_dict)
        data = {
            COLLECTED_AT: collected_at,
            **project_dict,
            **get_repository_stat(project),
            **get_github_stat(project),
            **get_github_topics(project),
            **get_stackoverflow_stat(project),
            **get_pypistats_stat(project),
            **get_pypi_projects_stat(
                project,
                pypi_index,
                main_dependencies_count,
                main_dependencies_count_with_extra,
                deep_dependencies_count,
                deep_dependencies_count_with_extra,
            ),
            **get_uses_and_used_by(project, uses),
        }
        projects_data.append(data)

    rank_and_update(projects_data)

    update_repo(projects_data, repo_github)

    return projects_data


if __name__ == "__main__":
    lambda_handler({}, None)
