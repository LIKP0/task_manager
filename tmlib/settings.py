"""tm's own settings: `<root>/tm_config.yaml`, read once at startup.

The division of labour with task lists is strict: a task list says what to run, this
says how tm runs.

Two deliberate absences:

- **No defaults in the code.** Every key must be present in the file. A default
  living in Python means the real value is in two places, and reading the config
  then tells you only what was overridden, not what tm will do. Here the file is
  the whole answer, so it is checked into git and travels with the tool.

- **No command-line override and no re-reading while running.** To change something,
  stop tm, edit the file, start it again. Running tasks live in their own tmux
  sessions, are not part of the scheduler, and survive the restart untouched, so the
  cost is about zero. In return the file always describes the running tm, instead of
  having to reconstruct which flags it was started with.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ConfigError

CONFIG_NAME = "tm_config.yaml"


@dataclass
class Settings:
    poll: float                 # scheduler poll interval, seconds
    device_check: bool          # pre-flight check of trainer.devices in task configs


def config_path(root: Path) -> Path:
    return root / CONFIG_NAME


def load_settings(root: Path) -> Settings:
    """Read `<root>/tm_config.yaml`. Anything missing or malformed is an error.

    Being strict is the point. A silently skipped typo (`pol1: 60`) would leave you
    believing a setting changed when it had not, with nothing to indicate otherwise.
    """
    path = config_path(root)
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise ConfigError(
            f"{path}: not found — tm keeps all of its settings in this file and has no\n"
            f"      built-in defaults. The repo ships one; copy it here if you are\n"
            f"      running with a custom --root or TM_ROOT:\n"
            f"          poll: 10\n"
            f"          device_check: true") from None
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read: {exc}") from exc

    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML\n{exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: expected a mapping of settings, "
                          f"got {type(doc).__name__}")

    known = {"poll", "device_check"}
    unknown = {str(k) for k in doc} - known
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)}. "
                          f"Known keys: {sorted(known)}")
    missing = known - {str(k) for k in doc}
    if missing:
        raise ConfigError(f"{path}: missing key(s) {sorted(missing)}. "
                          f"Every setting must be present; there are no defaults.")

    try:
        poll = float(doc["poll"])
    except (TypeError, ValueError):
        raise ConfigError(f"{path}: 'poll' must be a number of seconds") from None
    if poll <= 0:
        raise ConfigError(f"{path}: 'poll' must be positive")

    if not isinstance(doc["device_check"], bool):
        raise ConfigError(f"{path}: 'device_check' must be true or false")

    return Settings(poll=poll, device_check=doc["device_check"])
