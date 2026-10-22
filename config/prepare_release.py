"""Update release metadata from the GitHub Release event."""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import Version


def prepare_release(root, release):
    version = str(Version(release["tag_name"].removeprefix("v")))
    project_file = root / "pyproject.toml"
    project, count = re.subn(r'^version[ \t]*=[ \t]*"[^"]*"[ \t]*$', f'version = "{version}"', project_file.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError("Expected a project version in pyproject.toml.")

    changelog_file = root / "CHANGELOG.md"
    changelog = changelog_file.read_text(encoding="utf-8")
    date = (release.get("published_at") or datetime.now(timezone.utc).isoformat())[:10]
    body = (release.get("body") or "").strip()
    entry = f"## [{version}] - {date}\n\n{body}\n\n"
    existing = re.compile(rf"^## \[{re.escape(version)}\][^\n]*\n.*?(?=^## \[|\Z)", re.MULTILINE | re.DOTALL)
    if existing.search(changelog):
        changelog = existing.sub(lambda match: entry, changelog, count=1)
    else:
        first_release = re.search(r"^## \[", changelog, re.MULTILINE)
        if first_release is None:
            changelog = changelog.rstrip() + "\n\n" + entry
        else:
            changelog = changelog[:first_release.start()] + entry + changelog[first_release.start():]

    project_file.write_text(project, encoding="utf-8")
    changelog_file.write_text(changelog, encoding="utf-8")
    print(f"Prepared release {version}")


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    prepare_release(Path(__file__).resolve().parent.parent, event["release"])
