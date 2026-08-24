from nba_game_projections.data import download


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _patch_get(monkeypatch, requested_urls, content=b"fake-bytes"):
    def fake_get(url, timeout):
        requested_urls.append(url)
        return _FakeResponse(content)

    monkeypatch.setattr(download.requests, "get", fake_get)


def test_player_box_url_construction(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls)
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    download.download_season_files("player_box", [2017, 2018])

    assert requested_urls == [
        "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
        "/nba/player_box/parquet/player_box_2017.parquet",
        "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
        "/nba/player_box/parquet/player_box_2018.parquet",
    ]


def test_schedules_never_construct_malformed_two_digit_filenames(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls)
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    download.download_season_files("schedules", [2017, 2018])

    assert requested_urls == [
        "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
        "/nba/schedules/parquet/nba_schedule_2017.parquet",
        "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
        "/nba/schedules/parquet/nba_schedule_2018.parquet",
    ]
    for url in requested_urls:
        filename = url.rsplit("/", 1)[-1]
        season_token = filename.removeprefix("nba_schedule_").removesuffix(".parquet")
        assert len(season_token) == 4
        assert season_token.isdigit()


def test_skip_if_already_cached(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls)
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    dest = tmp_path / "player_box" / "player_box_2017.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"already-cached")

    downloaded = download.download_season_files("player_box", [2017])

    assert requested_urls == []
    assert downloaded == []
    assert dest.read_bytes() == b"already-cached"


def test_force_redownloads_cached_file(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls, content=b"new-bytes")
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    dest = tmp_path / "player_box" / "player_box_2017.parquet"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"stale-bytes")

    downloaded = download.download_season_files("player_box", [2017], force=True)

    assert requested_urls != []
    assert downloaded == [dest]
    assert dest.read_bytes() == b"new-bytes"


def test_betting_lines_downloads_single_json_archive(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls, content=b"{}")
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    dest = download.download_betting_lines()

    assert requested_urls == [
        "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main"
        "/nba/betting_lines/games-archive.json"
    ]
    assert dest == tmp_path / "betting_lines" / "games-archive.json"
    assert dest.read_bytes() == b"{}"


def test_betting_lines_skips_when_cached(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls)
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    dest = tmp_path / "betting_lines" / "games-archive.json"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"cached")

    result = download.download_betting_lines()

    assert requested_urls == []
    assert result is None


def test_download_all_covers_every_folder(monkeypatch, tmp_path):
    requested_urls = []
    _patch_get(monkeypatch, requested_urls)
    monkeypatch.setattr(download.config, "DATA_RAW_DIR", tmp_path)

    download.download_all(seasons=[2017])

    requested_folders = {url.split("/nba/", 1)[1].split("/", 1)[0] for url in requested_urls}
    assert requested_folders == {"player_box", "team_box", "schedules", "betting_lines"}
