# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
fetch.py - Cached downloads of public government documents, with provenance.

Every number this subpackage produces has to be walkable back to the exact file
it came from, so a download is never just bytes on disk. Each one is stored
beside a small JSON record of where it was asked for, where it actually came
from, when, and its SHA-256, and a later call for the same URL is answered from
that cache without touching the network.

**Where the files come from.** The official hosts (esd.whs.mil for the SARs,
gao.gov for the annual assessments) sit behind bot protection that refuses
scripted clients outright, whatever headers they send. This module does not
try to get round that. It asks the official URL first, and when that is
refused it asks the Internet Archive's Wayback Machine for its copy of the
same URL, the raw bytes it captured (the ``id_`` form, not the Archive's HTML
wrapper). Which of the two answered is recorded in the provenance, so nobody
has to take on trust that an archived copy is what the government published:
the capture timestamp and hash are there to check against.

**Files already on disk.** Anything that is not an ``http://`` or
``https://`` URL is taken as a local path (or a ``file://`` URL) and read in
place: nothing is downloaded or copied, and the record carries the file's
hash the same way. That is how a machine with no route to the internet reads
reports someone else downloaded; see :mod:`cost_core.public.local`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Identifies this library to the servers it asks, with a way to reach the
#: author, as the Internet Archive asks automated clients to do.
USER_AGENT = "cost-core (+https://github.com/MichaelFowler1/cost-risk-toolkit)"

#: Raw-bytes form of a Wayback Machine capture. ``{timestamp}`` may be a full
#: 14-digit capture time or a prefix such as ``2026``; the Archive serves the
#: capture nearest to it.
WAYBACK_RAW = "https://web.archive.org/web/{timestamp}id_/{url}"

#: The Wayback CDX index, used to list what the Archive holds under a prefix.
WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"

#: Seconds to wait between network requests. The Archive is a public service
#: run on donations; a pipeline that pulls a thousand reports should not hit it
#: as fast as the connection allows.
POLITE_DELAY = 1.5

#: A function that takes a URL and returns (status, bytes). Injected by tests
#: so the suite never touches the network.
Opener = Callable[[str], "tuple[int, bytes]"]


class FetchError(RuntimeError):
    """Raised when neither the official host nor the Archive supplies a file."""


@dataclass(frozen=True)
class Fetched:
    """A downloaded file and the record of where it came from.

    Attributes:
        path: The cached file on disk.
        url: The official URL that was asked for.
        served_from: The URL that actually supplied the bytes, which is either
            ``url`` itself or a Wayback Machine capture of it.
        sha256: Hash of the bytes, to compare against any other copy.
        retrieved_at: UTC time of the download, ISO 8601.
        size: Size in bytes.
    """

    path: Path
    url: str
    served_from: str
    sha256: str
    retrieved_at: str
    size: int

    @property
    def archived(self) -> bool:
        """True when the bytes came from the Wayback Machine."""
        return self.served_from != self.url


def cache_dir() -> Path:
    """Where downloads are kept: ``$COST_CORE_CACHE``, else a user cache."""
    env = os.environ.get("COST_CORE_CACHE")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".cache") / "cost-core" / "public"


def _default_opener(url: str) -> "tuple[int, bytes]":
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


_last_request = [0.0]


def _polite(opener: Opener, url: str, delay: float) -> "tuple[int, bytes]":
    wait = _last_request[0] + delay - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    try:
        return opener(url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.info("request to %s failed: %s", url, exc)
        return 0, b""
    finally:
        _last_request[0] = time.monotonic()


def _key(url: str) -> str:
    name = urllib.parse.unquote(url.rsplit("/", 1)[-1]) or "index"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[-80:]
    return f"{hashlib.sha1(url.encode()).hexdigest()[:12]}_{safe}"


def is_local(url: str) -> bool:
    """True when ``url`` names a file on disk rather than a web address."""
    # A Windows drive letter ("C:\\SARs") parses as a one-letter scheme,
    # which this also takes as local.
    return urllib.parse.urlsplit(str(url)).scheme.lower() not in ("http", "https")


def _local(url: str) -> Fetched:
    raw = str(url)
    if raw.lower().startswith("file://"):
        raw = urllib.request.url2pathname(urllib.parse.urlsplit(raw).path)
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FetchError(f"no such file: {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    stamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return Fetched(path=path, url=str(url), served_from=str(url), sha256=digest.hexdigest(),
                   retrieved_at=stamp.isoformat(timespec="seconds"), size=path.stat().st_size)


def fetch(
    url: str,
    *,
    wayback_timestamp: str = "2026",
    cache: Optional[Path] = None,
    refresh: bool = False,
    opener: Optional[Opener] = None,
    delay: float = POLITE_DELAY,
    try_official: bool = True,
) -> Fetched:
    """Download ``url`` once, trying the official host, then the Archive.

    A local path or ``file://`` URL is read in place instead, with no network
    request and no copy in the cache; ``retrieved_at`` is then the file's
    modification time.

    Args:
        url: The official URL of the document, or a local path.
        wayback_timestamp: Capture time to ask the Archive for when the
            official host refuses; a prefix picks the nearest capture.
        cache: Directory to cache in; defaults to :func:`cache_dir`.
        refresh: Download again even if a cached copy exists.
        opener: Replacement for the network call, for tests.
        delay: Minimum seconds between requests.
        try_official: Set False to go straight to the Archive for hosts known
            to refuse scripted clients, which saves a request per file.

    Raises:
        FetchError: Neither source supplied a non-empty file, or a local
            path does not exist.
    """
    if is_local(url):
        return _local(url)
    root = Path(cache) if cache is not None else cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / _key(url)
    meta = path.with_name(path.name + ".json")
    if path.exists() and meta.exists() and not refresh:
        record = json.loads(meta.read_text(encoding="utf-8"))
        record["path"] = path
        return Fetched(**record)

    opener = opener or _default_opener
    tried = []
    candidates = ([url] if try_official else []) + [
        WAYBACK_RAW.format(timestamp=wayback_timestamp, url=url)
    ]
    for source in candidates:
        status, data = _polite(opener, source, delay)
        tried.append(f"{source} -> {status or 'no response'}")
        if status == 200 and data:
            break
    else:
        raise FetchError(f"could not download {url}; tried " + "; ".join(tried))

    path.write_bytes(data)
    fetched = Fetched(
        path=path,
        url=url,
        served_from=source,
        sha256=hashlib.sha256(data).hexdigest(),
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        size=len(data),
    )
    record = asdict(fetched)
    record.pop("path")
    meta.write_text(json.dumps(record, indent=1), encoding="utf-8")
    logger.debug("fetched %s from %s", url, source)
    return fetched


def wayback_listing(
    prefix: str,
    *,
    opener: Optional[Opener] = None,
    delay: float = POLITE_DELAY,
) -> "list[tuple[str, str, int]]":
    """Everything the Archive holds under a URL prefix.

    Returns ``(original_url, capture_timestamp, length)`` for one successful
    capture of each distinct URL (the Archive's first). Not cached, since the point of
    asking is to see captures added since last time.
    """
    query = urllib.parse.urlencode({
        "url": prefix,
        "matchType": "prefix",
        "filter": "statuscode:200",
        "collapse": "urlkey",
        "fl": "original,timestamp,length",
        "limit": "20000",
    })
    status, data = _polite(opener or _default_opener, f"{WAYBACK_CDX}?{query}", delay)
    if status != 200:
        raise FetchError(f"Wayback CDX listing failed for {prefix} ({status or 'no response'})")
    rows = []
    for line in data.decode("utf-8", "replace").splitlines():
        parts = line.split(" ")
        if len(parts) == 3:
            rows.append((parts[0], parts[1], int(parts[2]) if parts[2].isdigit() else 0))
    return rows
