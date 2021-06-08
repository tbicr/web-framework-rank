import asyncio
import csv
import io
import json
import logging
import os
import tempfile
import time
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from html.parser import HTMLParser
from itertools import chain
from typing import Dict, List, Union

import aiohttp
import git
import requests
from github import Github, InputGitAuthor
from pytz import UTC


GITHUB_REPO = os.environ["GITHUB_REPO"]
GITHUB_BRANCH = os.environ["GITHUB_BRANCH"]
GITHUB_AUTHOR_NAME = os.environ["GITHUB_AUTHOR_NAME"]
GITHUB_AUTHOR_EMAIL = os.environ["GITHUB_AUTHOR_EMAIL"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

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

STACKOVERFLOW_QUESTIONS = "+stackoverflow_questions"

PYPI_PROJECT_MENTIONS = "-pypi_project_mentions"
PYPI_USED_AS_MAIN_DEPENDENCY = "+pypi_used_as_main_dependency"
PYPI_RELEASES = "-pypi_releases"
PYPI_LAST_RELEASE_AT = "-pypi_last_release"

PYPISTATS_DOWNLOADS_LAST_MONTH = "+pypistats_downloads_last_month"

NAME = "name"
COLLECTED_AT = "-collected_at"
RANK = "rank"
SCORE = "score"

RANK_SUFFIX = "__rank"

PROJECT_LANGUAGE = "language"
PROJECT_REPO = "repo"
PROJECT_GIT = "git"
PROJECT_GITHUB_REPO = "github_repo"
PROJECT_STACKOVERFLOW_TAG = "stackoverflow_tag"
PROJECT_PYPISTAT_PROJECT = "pypistat_project"
PROJECT_PYPI_PROJECT = "pypi_project"

DATE_FIELDS = {
    REPO_FIRST_COMMIT_AT,
    REPO_LAST_COMMIT_AT,
    GITHUB_CREATED_AT,
    GITHUB_UPDATED_AT,
    COLLECTED_AT,
}

FIELDS = [
    NAME,

    RANK,
    SCORE,

    PYPISTATS_DOWNLOADS_LAST_MONTH,

    PYPI_PROJECT_MENTIONS,
    PYPI_USED_AS_MAIN_DEPENDENCY,
    PYPI_RELEASES,
    PYPI_LAST_RELEASE_AT,

    STACKOVERFLOW_QUESTIONS,

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
    stackoverflow_tag: str
    pypistat_project: str
    pypi_project: str


class GithubWrapper:
    DATA = "data.csv"
    README = "README.md"
    PYPI_CACHE = "pypi_cache.json"
    PROJECTS = "projects.json"

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        g = Github(GITHUB_TOKEN)
        self.repo = g.get_repo(GITHUB_REPO)
        self.author = InputGitAuthor(GITHUB_AUTHOR_NAME, GITHUB_AUTHOR_EMAIL)
        self._state = {}

    def _fetch_content(self, file_name):
        self._state[file_name] = self.repo.get_contents(file_name, ref=GITHUB_BRANCH)
        return self._state[file_name].decoded_content.decode("utf-8")

    def _update_content(self, file_name, commit_message, content):
        if not self.dry_run:
            self.repo.update_file(
                self._state[file_name].path, commit_message, content,
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


def _commit_datetime(commit: git.Commit) -> datetime:
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
def _get_pypi_package_meta(package):
    response = session.get(f"https://pypi.org/pypi/{package}/json")
    response.raise_for_status()
    return response.json()


async def _get_pypi_package_dependencies(session: aiohttp.ClientSession, package: str):
    async with session.get(f"https://pypi.org/pypi/{package}/json") as response:
        if response.status == 404:
            return package.lower(), []
        response.raise_for_status()
        meta = await response.json()
        if meta is None:
            return package.lower(), []
        return package.lower(), sorted({
            dependency.split()[0].split("[")[0].lower()
            for dependency in meta["info"]["requires_dist"] or []
        } - {package})


async def _concurrent_pypi_dependencies_fetching(packages):
    async with aiohttp.ClientSession() as session:
        tasks = [_get_pypi_package_dependencies(session, package) for package in packages]
        results = await asyncio.gather(*tasks)
    return results


@retry
@logged
def get_and_update_dependencies(
        pypi_index: List[str], repo_github: GithubWrapper
) -> Dict[str, List[str]]:
    data = json.loads(repo_github.pypi_cache)
    uncached = list({package for package in pypi_index if package.lower() not in data})
    logger.info(f"pypi packages for download: {len(uncached)} total: {len(pypi_index)}")
    chunk_size = 10000
    for i in range(0, len(uncached), chunk_size):
        chunk = uncached[i:i + chunk_size]
        for package, dependencies in asyncio.run(_concurrent_pypi_dependencies_fetching(chunk)):
            data[package] = dependencies
        logger.info(f"pypi packages chunk downloaded: {i + chunk_size} off {len(uncached)}")
    repo_github.pypi_cache = (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False).
                              replace("[\n    ", "[").replace("\n   ", "").replace("\n  ]", "]"))
    return data


@logged
def get_main_dependencies_count(data: Dict[str, List[str]]) -> Dict[str, int]:
    return Counter(
        dependency
        for package, dependencies in data.items()
        for dependency in dependencies
    )


@retry
@logged
def get_pypi_projects_stat(
        project: Project,
        pypi_index: List[str],
        main_dependencies_count: Dict[str, int],
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
        PYPI_RELEASES: releases_count,
        PYPI_LAST_RELEASE_AT: last_release_at,
    }


@logged
def get_rank_and_score(
        projects_data: List[Dict[str, Union[int, str]]]
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


def rank_and_update(projects_data: List[Dict[str, Union[int, str]]]):
    rank_and_score = get_rank_and_score(projects_data)
    for data in projects_data:
        data.update(rank_and_score[data[NAME]])


def get_csv_data(content: str) -> List[Dict[str, Union[int, str]]]:
    handler = io.StringIO(content)
    reader = csv.DictReader(handler, fieldnames=FIELDS, lineterminator="\n")
    next(reader)
    name_project_data_mapping = {}
    for data in reader:
        name_project_data_mapping[data[NAME]] = {
            field: int(value) if field not in DATE_FIELDS | {NAME} else value
            for field, value in data.items()
        }
    project_data = list(name_project_data_mapping.values())
    rank_and_update(project_data)
    return project_data


def update_csv(result: List[Dict[str, Union[int, str]]], content: str) -> str:
    handler = io.StringIO(content)
    handler.read()
    writer = csv.DictWriter(handler, fieldnames=FIELDS, lineterminator="\n")
    for data in sorted(result, key=lambda data: data[RANK]):
        writer.writerow({field: value for field, value in data.items() if field in FIELDS})
    return handler.getvalue()


def readme_table_field(
        field: str,
        data: Dict[str, Union[int, str]],
        prev_data: Dict[str, Union[int, str]],
):
    value = data[field]
    prev_value = prev_data.get(field)
    rank = data.get(field + RANK_SUFFIX)
    prev_rank = prev_data.get(field + RANK_SUFFIX)

    rank_change = " "
    if prev_rank is None or rank < prev_rank:
        rank_change = "▲"
    elif rank > prev_rank:
        rank_change = "▼"

    if field == NAME:
        repo = data[PROJECT_REPO]
        tooltip = data[PROJECT_LANGUAGE]
        return f"[<sub>{value}</sub>]({repo} \"{tooltip}\")"
    elif field == RANK:
        change = f"{prev_value - value:+}" if prev_value else "new"
        tooltip = f"{rank_change} {change}"
        return f"[<sub>{value}</sub>](# \"{tooltip}\")"
    elif field == SCORE:
        change = f"{value - prev_value:+}" if prev_value else f"{value:+}"
        tooltip = f"{rank_change} {change}"
        return f"[<sub>{value}</sub>](# \"{tooltip}\")"
    else:
        if field in DATE_FIELDS:
            value = data[field].split("T")[0]
            change = ""
        else:
            change = (
                f"{round(100 * (value - prev_value) / prev_value, 2):+}%"
                if prev_value else "+100%"
            )
        field_name = field.lstrip("+-").replace("_", " ")
        tooltip = f"{rank_change} #{rank} in {field_name} {change}".rstrip()
        return f"[<sub>{value}</sub>](# \"{tooltip}\")"


def update_readme(
        result: List[Dict[str, Union[int, str]]],
        prev_result: List[Dict[str, Union[int, str]]],
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
def update_repo(result: List[Dict[str, Union[int, str]]], repo_github: GithubWrapper):
    prev_result = get_csv_data(repo_github.data)
    repo_github.readme = update_readme(result, prev_result, repo_github.readme)
    repo_github.data = update_csv(result, repo_github.data)


@logged
def lambda_handler(event, context):
    dry_run = event.get("dry_run", False)
    repo_github = GithubWrapper(dry_run)
    projects = json.loads(repo_github.projects)
    pypi_index = get_pypi_index()
    dependencies = get_and_update_dependencies(pypi_index, repo_github)
    main_dependencies_count = get_main_dependencies_count(dependencies)
    collected_at = datetime.utcnow().isoformat().split(".")[0].rstrip("Z")
    projects_data = []
    for project_dict in projects:
        project = Project(**project_dict)
        data = {
            COLLECTED_AT: collected_at,
            **project_dict,
            **get_repository_stat(project),
            **get_github_stat(project),
            **get_stackoverflow_stat(project),
            **get_pypistats_stat(project),
            **get_pypi_projects_stat(project, pypi_index, main_dependencies_count),
        }
        projects_data.append(data)

    rank_and_update(projects_data)

    update_repo(projects_data, repo_github)

    return projects_data


if __name__ == "__main__":
    lambda_handler({}, None)
