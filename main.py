import csv
import io
import json
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from html.parser import HTMLParser
from typing import List, Dict, Any

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

PYPI_PROJECTS = "+pypi projects"

PYPISTATS_DOWNLOADS_LAST_MONTH = "+pypistats_downloads_last_month"

NAME = "name"
COLLECTED_AT = "-collected_at"
RANK = "rank"
SCORE = "score"

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

    PYPI_PROJECTS,

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


@dataclass(frozen=True)
class Project:
    name: str
    language: str
    repo: str
    git: str
    github_repo: str
    stackoverflow_tag: str
    pypistat_project: str
    pypi_projects: str


session = requests.session()


def retry(func):
    def wrapper(*args, **kwargs):
        for timeout in [3, 10, 30, None]:
            try:
                return func(*args, **kwargs)
            except Exception as err:
                if timeout is None:
                    raise err
                time.sleep(timeout)
    return wrapper


def _commit_datetime(commit: git.Commit) -> datetime:
    return commit.committed_datetime.astimezone(UTC).replace(tzinfo=None)


@retry
def get_repository_stat(project: Project) -> Dict[str, Any]:
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
        last_month_commits = [commit for commit in all_commits if _commit_datetime(commit) > month_ago]
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
def get_github_stat(project: Project) -> Dict[str, Any]:
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
    response = session.get(url)
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
def get_stackoverflow_stat(project: Project) -> Dict[str, Any]:
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
def get_pypistats_stat(project: Project) -> Dict[str, Any]:
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
            self.packages.append(data.lower())


@cache
def _get_pypi_index() -> List[str]:
    url = "https://pypi.python.org/simple/"
    data = session.get(url).text
    parser = SimplePypiIndexHTMLParser()
    parser.feed(data)
    return parser.packages


@retry
def get_pypi_projects_stat(project: Project) -> Dict[str, Any]:
    index = _get_pypi_index()
    project_name = project.name.lower()
    count = 0
    for name in index:
        if project_name in name:
            count += 1
    return {
        PYPI_PROJECTS: count,
    }


def get_rank_and_score(projects_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    fields = {field for data in projects_data for field in data.keys() if field.startswith("+")}
    scores: Dict[str, float] = defaultdict(int)
    for field in fields:
        ordered_values = defaultdict(list)
        for data in projects_data:
            name = data[NAME]
            value = data[field]
            if isinstance(value, str):
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
            current_field_score -= len(names)
    results = {}
    for index, (name, score) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]), start=1):
        results[name] = {RANK: index, SCORE: round(100 * score / len(fields) / len(projects_data))}
    return results


def update_csv(result: List[Dict[str, Any]], content: str) -> str:
    handler = io.StringIO(content)
    handler.read()
    writer = csv.DictWriter(handler, fieldnames=FIELDS)
    for data in sorted(result, key=lambda data: data[RANK]):
        writer.writerow(data)
    return handler.getvalue()


def update_readme(result: List[Dict[str, Any]], content: str) -> str:
    top_part = content.split("---", 1)[0].rsplit("\n", 2)[0]
    bottom_part = content.split("---", 1)[1].split("\n\n", 1)[1]
    header = " | ".join(
        field.lstrip("+-").replace("_", " ") for field in FIELDS
        if not field.startswith("-") and field in result[0]
    )
    splitter = " | ".join(
        ":---" if field == NAME else "---:" for field in FIELDS
        if not field.startswith("-") and field in result[0])
    table = [header, splitter] + [
        " | ".join(
            data[field].split("T")[0] if field in DATE_FIELDS else str(data[field])
            for field in FIELDS
            if not field.startswith("-") and field in data
        )
        for data in sorted(result, key=lambda data: data[RANK])
    ]
    return top_part + "\n" + "\n".join(table) + "\n\n" + bottom_part


@retry
def update_repo(result: List[Dict[str, Any]]):
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(GITHUB_REPO)
    DATA = "data.csv"
    README = "README.md"
    author = InputGitAuthor(GITHUB_AUTHOR_NAME, GITHUB_AUTHOR_EMAIL)

    readme_file = repo.get_contents(README, ref=GITHUB_BRANCH)
    readme_content = readme_file.decoded_content.decode("utf-8")
    readme_content = update_readme(result, readme_content)
    repo.update_file(
        readme_file.path, "update readme", readme_content, readme_file.sha, branch=GITHUB_BRANCH, author=author)

    data_file = repo.get_contents(DATA, ref=GITHUB_BRANCH)
    data_content = data_file.decoded_content.decode("utf-8")
    data_content = update_csv(result, data_content)
    repo.update_file(
        data_file.path, "update data", data_content, data_file.sha, branch=GITHUB_BRANCH, author=author)


def lambda_handler(event, context):
    with open("projects.json") as handle:
        projects = json.load(handle)
    collected_at = datetime.utcnow().isoformat()
    projects_data = []
    for project_dict in projects:
        project = Project(**project_dict)
        data = {
            NAME: project.name,
            COLLECTED_AT: collected_at,
            **get_repository_stat(project),
            **get_github_stat(project),
            **get_stackoverflow_stat(project),
            **get_pypistats_stat(project),
            **get_pypi_projects_stat(project),
        }
        projects_data.append(data)

    rank_and_score = get_rank_and_score(projects_data)
    for data in projects_data:
        data.update(rank_and_score[data[NAME]])

    update_repo(projects_data)


if __name__ == "__main__":
    lambda_handler(None, None)
