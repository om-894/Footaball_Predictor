"""
Downloads with a local cache, retries and a pause between requests.

A file is only downloaded again if the server says it has changed, using its ETag.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from footy.config import RAW_DIR

log = logging.getLogger(__name__)

USER_AGENT = "footy/2.0 (+https://github.com/om-894/Footaball_Predictor)"

MIN_REQUEST_INTERVAL_S = 1.0 # minimum seconds between requests to the same server


class SourceError(RuntimeError):
    """A source could not be downloaded or read."""


@dataclass
class CachedDownloader:
    """Downloads files to `cache_dir`, reusing the saved copy while the server says it is current.

    Each file's ETag is saved next to it in a .meta.json file and sent with the next
    request, so an unchanged file comes back as a 304 with nothing to download.
    """

    cache_dir: Path = RAW_DIR
    timeout: int = 120
    max_retries: int = 4
    min_interval_s: float = MIN_REQUEST_INTERVAL_S

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_at = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_request_at = time.monotonic()

    def _meta_path(self, target: Path) -> Path:
        return target.with_suffix(target.suffix + ".meta.json")

    def _read_meta(self, target: Path) -> dict:
        meta_path = self._meta_path(target)
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def fetch(self, url: str, filename: str | None = None, *, force: bool = False) -> Path:
        """Local path to the contents of `url`, downloading only if it changed or `force` is set."""
        name = filename or url.rsplit("/", 1)[-1]
        target = self.cache_dir / name
        meta = {} if force else self._read_meta(target)

        headers: dict[str, str] = {}
        if target.exists() and meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                self._throttle()
                with self._session.get(
                    url, headers=headers, timeout=self.timeout, stream=True
                ) as response:
                    if response.status_code == 304 and target.exists():
                        log.info("cache hit (304) %s", name)
                        return target

                    response.raise_for_status()

                    # write to a .part file first, so a broken download never leaves half a CSV
                    tmp = target.with_suffix(target.suffix + ".part")
                    digest = hashlib.sha256()
                    with tmp.open("wb") as fh:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                            digest.update(chunk)
                    tmp.replace(target)

                    self._meta_path(target).write_text(
                        json.dumps({
                            "url": url,
                            "etag": response.headers.get("ETag"),
                            "sha256": digest.hexdigest(),
                            "bytes": target.stat().st_size,
                            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }, indent=2)
                    )
                    log.info("downloaded %s (%.1f MB)", name, target.stat().st_size / 1e6)
                    return target

            except (requests.RequestException, OSError) as exc:
                last_error = exc
                backoff = 2.0**attempt
                log.warning(
                    "fetch failed (%s/%s) for %s: %s, retrying in %.0fs",
                    attempt + 1, self.max_retries, name, exc, backoff,
                )
                time.sleep(backoff)

        # every retry failed, so fall back to the old copy if there is one
        if target.exists():
            log.error("all retries failed for %s; using stale cache", name)
            return target

        raise SourceError(f"Could not fetch {url}: {last_error}") from last_error
