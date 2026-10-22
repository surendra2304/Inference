"""M13: the production Docker Compose stack must be internally consistent.

Every assertion here fails against the previous file, which was unrunnable:
  * `nginx/` was bind-mounted but the directory does not exist;
  * `api-1/2/3` published no ports, so with nginx unstartable nothing was
    reachable;
  * `env_file: .env` was mandatory while the repo ships `.env.example`;
  * redis and postgres were provisioned although nothing in app/ connects to
    them and no driver for them is even declared;
  * the obsolete top-level `version` key was present.

Also covers .dockerignore, whose leading UTF-8 BOM made the first pattern
literally "\\ufeff.git" so the git history was never excluded from the image.
"""

import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.production.yml"


def _load() -> dict:
    assert COMPOSE.exists(), "docker-compose.production.yml is missing"
    data = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_compose_is_valid_yaml_with_services():
    data = _load()
    assert data.get("services"), "compose file declares no services"


def test_compose_drops_obsolete_version_key():
    data = _load()
    assert "version" not in data, (
        "top-level `version` is obsolete in Compose v2 and emits a warning"
    )


def test_every_service_path_exists_on_disk():
    """Bind mounts and build contexts must resolve before compose runs."""
    data = _load()
    problems = []
    for name, svc in data["services"].items():
        if "build" in svc:
            build = svc["build"]
            context = build if isinstance(build, str) else build.get("context", ".")
            if not (ROOT / context / "Dockerfile").exists():
                problems.append(f"{name}: no Dockerfile at {context}")
        for mount in svc.get("volumes", []):
            if isinstance(mount, str) and mount.startswith("."):
                host_path = mount.split(":")[0]
                if not (ROOT / host_path).exists():
                    problems.append(f"{name}: missing host path {host_path}")
    assert not problems, "compose references paths that do not exist:\n  " + "\n  ".join(problems)


def test_env_file_is_not_a_hard_requirement():
    """The repo ships .env.example, so .env must be optional at start-up."""
    data = _load()
    for name, svc in data["services"].items():
        for entry in svc.get("env_file", []):
            if isinstance(entry, dict):
                assert entry.get("required") is False, (
                    f"{name}: env_file {entry} would abort compose when absent"
                )
                path = entry.get("path", "")
            else:
                raise AssertionError(
                    f"{name}: env_file {entry!r} is mandatory but .env is not committed"
                )
            assert isinstance(path, str) and path, f"{name}: env_file entry has no path"


def test_built_service_publishes_a_port():
    """The application itself must be reachable, not just a proxy in front of it.

    The previous file gave nginx 80/443 while api-1/2/3 published nothing at
    all - and nginx could not start because ./nginx/ does not exist - so the
    stack had no reachable surface.
    """
    data = _load()
    built = {n: s for n, s in data["services"].items() if "build" in s}
    assert built, "expected at least one service built from this repository"
    unreached = [n for n, s in built.items() if not s.get("ports")]
    assert not unreached, (
        f"built service(s) {unreached} publish no port, so nothing is reachable "
        "without a fronting proxy that this repository does not ship"
    )


def test_no_services_for_architecture_the_app_does_not_use():
    """app/ contains no Redis or Postgres client and declares no driver."""
    data = _load()
    declared = set(data["services"])
    unused = declared & {"redis", "postgres", "nginx"}
    assert not unused, (
        f"provisioning {sorted(unused)} but nothing in app/ connects to them"
    )

    requirements = (ROOT / "requirements.txt").read_text().lower()
    for driver in ("redis", "psycopg", "asyncpg"):
        assert driver not in requirements, f"{driver} driver declared but unused"


def test_persistent_volume_is_declared():
    data = _load()
    named_volumes = set(data.get("volumes") or {})
    for name, svc in data["services"].items():
        for mount in svc.get("volumes", []):
            if isinstance(mount, str) and ":" in mount and not mount.startswith("."):
                volume = mount.split(":")[0]
                assert volume in named_volumes, (
                    f"{name}: volume {volume} used but not declared"
                )


def test_api_service_has_a_healthcheck():
    data = _load()
    assert any(
        "healthcheck" in svc for svc in data["services"].values()
    ), "expected at least one healthcheck"


# ── .dockerignore ─────────────────────────────────────────────────────────────

def test_dockerignore_has_no_byte_order_mark():
    raw = (ROOT / ".dockerignore").read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", (
        "a leading BOM becomes part of the first pattern, so `.git` was never "
        "actually excluded and full history shipped in the image"
    )


def test_dockerignore_excludes_vcs_and_virtualenv():
    lines = [
        ln.strip().lstrip("\ufeff")
        for ln in (ROOT / ".dockerignore").read_text().splitlines()
    ]
    assert ".git" in lines, "git history must not be copied into the image"
    assert ".venv" in lines, "the virtualenv must not be copied into the image"


def test_dockerfile_only_copies_files_that_exist():
    dockerfile = (ROOT / "Dockerfile").read_text()
    for line in dockerfile.splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            parts = line.split()
            # drop flags such as --chown=user:user; last token is the destination
            sources = [tok for tok in parts[1:-1] if not tok.startswith("--")]
            for src in sources:
                if src in (".", "..") or src.startswith("$"):
                    continue
                assert (ROOT / src).exists(), f"Dockerfile copies missing path {src}"
