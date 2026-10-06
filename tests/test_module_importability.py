"""M9: every file under app/ must be reachable through the import system.

The failure this locks down: `app/monitoring/alerts.py` and
`app/optimization/token_optimizer.py` were unreachable because sibling
`app/monitoring.py` and `app/optimization.py` modules occupy those names while
the directories had no `__init__.py`, so Python resolved `app.monitoring` to the
module and the nested files could never be imported. Both turned out to be
orphaned duplicates - `app/token_optimizer.py` was byte-identical and
`app/alerts.py` was a superset of `app/monitoring/alerts.py` and is the version
the admin routes actually use - so they were removed rather than rescued.
"""

import importlib
import pathlib

import app


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts[:-1]) + ([] if path.name == "__init__.py" else [path.stem])
    return ".".join(["app"] + parts) if parts else "app"


def test_every_application_module_is_importable():
    root = pathlib.Path(app.__file__).resolve().parent
    failures: list[str] = []
    checked = 0

    for path in sorted(root.rglob("*.py")):
        name = _module_name(path, root)
        checked += 1
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any failure makes code dead
            failures.append(f"{path.relative_to(root)} -> {name}: {type(exc).__name__}: {exc}")

    assert checked > 100, f"expected a substantial tree, only found {checked} files"
    assert not failures, (
        "these files exist but can never be imported, so they are dead:\n  "
        + "\n  ".join(failures)
    )


def test_no_directory_shares_its_name_with_a_sibling_module():
    """`app/x.py` next to `app/x/` means one of the two is dead weight.

    Without an `__init__.py` the module wins and every file inside the
    directory becomes unreachable - which is exactly how M9 happened.
    """
    root = pathlib.Path(app.__file__).resolve().parent
    collisions = []
    for directory in sorted(root.rglob("*")):
        if not directory.is_dir() or directory.name.startswith("__"):
            continue
        module_file = directory.with_suffix(".py")
        if module_file.exists():
            hidden = sorted(
                str(f.relative_to(root)) for f in directory.rglob("*.py")
            )
            collisions.append(
                f"{directory.relative_to(root)}/ collides with "
                f"{module_file.relative_to(root)} (would hide: {hidden})"
            )
    assert not collisions, (
        "rename or remove one side of each collision:\n  " + "\n  ".join(collisions)
    )


def test_admin_alerts_endpoint_reads_the_live_alert_system():
    """The alerts module the routers use must be the importable one."""
    from app.alerts import alert_system
    from app.routers import admin_analytics

    assert admin_analytics.alert_system is alert_system
    assert alert_system.get_alerts(), "the live alert store should not be empty"
