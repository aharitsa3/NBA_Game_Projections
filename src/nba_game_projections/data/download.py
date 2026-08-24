"""Downloads and caches hoopR-nba-data raw files for the Phase 1 season range.

Fetches `player_box`, `team_box`, and `schedules` (one parquet file per
season) plus the single `betting_lines` JSON archive from
`raw.githubusercontent.com/sportsdataverse/hoopR-nba-data`, caching each
file under `data/raw/<folder>/`. Already-cached files are skipped, so
re-running is a no-op except for whatever changed upstream — the source
repo is actively updated in-season.

Season file URLs are built directly from the season integer (e.g.
`nba_schedule_2017.parquet`) rather than by listing the remote directory,
which naturally avoids the malformed two-digit placeholder filenames that
exist alongside the real `schedules` files (PRD §4) without needing an
explicit filter step.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import requests
from tqdm import tqdm

from nba_game_projections import config

logger = logging.getLogger(__name__)

_SEASON_FOLDERS = {
    "player_box": "player_box",
    "team_box": "team_box",
    "schedules": "nba_schedule",
}

_REQUEST_TIMEOUT_SECONDS = 60


def _remote_url(*parts: str) -> str:
    return "/".join((config.HOOPR_BASE_URL, *parts))


def _download_file(url: str, dest: Path, *, force: bool = False) -> bool:
    """Download `url` to `dest` unless already cached. Returns True if a download happened."""
    if dest.exists() and not force:
        logger.debug("skipping cached file: %s", dest)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(dest)
    return True


def download_season_files(
    folder: str, seasons: Iterable[int], *, force: bool = False
) -> list[Path]:
    """Download one season-partitioned folder (`player_box`, `team_box`, or `schedules`).

    Returns the paths that were actually downloaded (excludes cache hits).
    """
    prefix = _SEASON_FOLDERS[folder]
    downloaded = []
    for season in tqdm(list(seasons), desc=f"downloading {folder}"):
        filename = f"{prefix}_{season}.parquet"
        url = _remote_url("nba", folder, "parquet", filename)
        dest = config.DATA_RAW_DIR / folder / filename
        if _download_file(url, dest, force=force):
            downloaded.append(dest)
    return downloaded


def download_betting_lines(*, force: bool = False) -> Path | None:
    """Download the single betting_lines JSON archive. Returns the path if downloaded, else None."""
    url = _remote_url("nba", "betting_lines", "games-archive.json")
    dest = config.DATA_RAW_DIR / "betting_lines" / "games-archive.json"
    return dest if _download_file(url, dest, force=force) else None


def download_all(seasons: Iterable[int] | None = None, *, force: bool = False) -> None:
    """Download all Phase 1 raw data: player_box, team_box, schedules, and betting_lines."""
    seasons = list(seasons) if seasons is not None else list(config.season_range())
    for folder in ("player_box", "team_box", "schedules"):
        download_season_files(folder, seasons, force=force)
    download_betting_lines(force=force)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download_all()
