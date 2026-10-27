"""Chunked, cached, read-only access to a Zarr v2 / OME-Zarr volume over HTTP(S), S3 (anonymous) or a local path.

Only the chunks a query touches are fetched. Decoded chunks live in a bounded in-memory LRU; raw chunk bytes can
also be kept in a size-capped disk cache so reruns and the self-test do not download twice. Point sampling is
trilinear and processes points in spatial batches, so memory stays bounded however large the surface is.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

USER_AGENT = "turncoat/1.0 (+https://github.com/gmunhoz0810/turncoat)"


class VolumeError(RuntimeError):
    """The volume cannot be opened or read."""


def _normalise_source(src: str) -> str:
    """s3://bucket/key -> https://bucket.s3.amazonaws.com/key (anonymous public read); strip trailing slashes."""
    src = src.strip()
    if src.startswith("s3://"):
        bucket, _, key = src[5:].partition("/")
        src = f"https://{bucket}.s3.amazonaws.com/{key}"
    return src.rstrip("/")


def _is_remote(src: str) -> bool:
    return src.startswith("http://") or src.startswith("https://")


def _ssl_context() -> ssl.SSLContext:
    """certifi's CA bundle (python.org builds on macOS ship without system CAs), else the system default."""
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


_SSL = _ssl_context()


def _http_get(url: str, retries: int = 5, timeout: float = 120.0) -> bytes | None:
    """GET with retries and backoff; None on 404/403-missing (absent chunk = fill value)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 403):
                return None
            last = exc
        except (OSError, TimeoutError) as exc:
            last = exc
        time.sleep(min(30.0, 1.0 + 2.0 ** attempt))
    raise VolumeError(f"GET failed after {retries} tries: {url}: {last}")


class _Store:
    """Byte access to keys under a zarr root, remote or local, with an optional capped disk cache."""

    def __init__(self, root: str, cache_dir: Path | None, cache_bytes: int):
        self.root = root
        self.remote = _is_remote(root)
        self.cache_dir = None
        self.cache_bytes = cache_bytes
        self._lock = threading.Lock()
        if self.remote and cache_dir is not None and cache_bytes > 0:
            tag = hashlib.sha1(root.encode()).hexdigest()[:16]
            self.cache_dir = Path(cache_dir) / tag
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "SOURCE.txt").write_text(root + "\n")
            self._cached = sum(f.stat().st_size for f in self.cache_dir.glob("c_*"))
        self.bytes_fetched = 0

    def get(self, key: str) -> bytes | None:
        if not self.remote:
            p = Path(self.root) / key
            return p.read_bytes() if p.exists() else None
        if self.cache_dir is not None:
            f = self.cache_dir / ("c_" + key.replace("/", "_"))
            if f.exists():
                data = f.read_bytes()
                return data if data else None
        data = _http_get(f"{self.root}/{key}")
        with self._lock:
            self.bytes_fetched += len(data or b"")
        if self.cache_dir is not None:
            self._put(f, data or b"")
        return data

    def _put(self, f: Path, data: bytes) -> None:
        tmp = f.with_name(f.name + f".tmp{os.getpid()}_{threading.get_ident()}")
        tmp.write_bytes(data)
        os.replace(tmp, f)
        with self._lock:
            self._cached += len(data)
            if self._cached > self.cache_bytes:
                self._evict()

    def _evict(self) -> None:
        files = sorted(self.cache_dir.glob("c_*"), key=lambda p: p.stat().st_mtime)
        target = int(self.cache_bytes * 0.8)
        for p in files:
            if self._cached <= target:
                break
            try:
                size = p.stat().st_size
                p.unlink()
                self._cached -= size
            except FileNotFoundError:
                continue


class Volume:
    """One resolution level of a Zarr v2 array (or an OME-Zarr group, where `level` picks the dataset).

    Coordinates follow the zarr axis order (z, y, x) for boxes; `sample` takes points as (x, y, z) like tifxyz.
    """

    def __init__(self, source: str | Path, level: int | str = 0, cache_dir: str | Path | None = None,
                 cache_gb: float = 4.0, memory_gb: float = 1.5, workers: int = 16):
        src = _normalise_source(str(source))
        if not _is_remote(src) and not Path(src).exists():
            raise VolumeError(f"volume not found: {src}")
        self.source = src
        root = self._resolve_array_root(src, level, cache_dir, cache_gb)
        self.store = _Store(root, Path(cache_dir) if cache_dir else None, int(cache_gb * 1e9))
        raw = self.store.get(".zarray")
        if raw is None:
            raise VolumeError(f"no .zarray under {root}")
        meta = json.loads(raw)
        if int(meta.get("zarr_format", 2)) != 2:
            raise VolumeError("only Zarr v2 arrays are supported")
        if meta.get("order", "C") != "C":
            raise VolumeError("only C-order arrays are supported")
        self.meta = meta
        self.shape = tuple(int(v) for v in meta["shape"])
        self.chunks = tuple(int(v) for v in meta["chunks"])
        if len(self.shape) != 3:
            raise VolumeError(f"expected a 3-D array, got shape {self.shape}")
        self.dtype = np.dtype(meta["dtype"])
        self.fill = meta.get("fill_value") or 0
        self.sep = meta.get("dimension_separator", ".")
        comp = meta.get("compressor")
        filters = meta.get("filters") or []
        if comp or filters:
            import numcodecs
            self.codec = numcodecs.get_codec(comp) if comp else None
            self.filters = [numcodecs.get_codec(f) for f in filters]
        else:
            self.codec, self.filters = None, []
        chunk_bytes = int(np.prod(self.chunks)) * self.dtype.itemsize
        self.max_chunks = max(16, int(memory_gb * 1e9 // chunk_bytes))
        self.workers = workers
        self._mem: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def _resolve_array_root(self, src: str, level, cache_dir, cache_gb) -> str:
        probe = _Store(src, None, 0)
        if probe.get(".zarray") is not None:
            return src
        attrs = probe.get(".zattrs")
        if attrs is None:
            return f"{src}/{level}"
        try:
            ms = json.loads(attrs).get("multiscales")
            paths = [d["path"] for d in ms[0]["datasets"]]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return f"{src}/{level}"
        lvl = str(level)
        if lvl not in paths:
            if isinstance(level, int) and 0 <= level < len(paths):
                lvl = paths[level]
            else:
                raise VolumeError(f"level {level} not in multiscales {paths}")
        return f"{src}/{lvl}"

    # ---- chunks ---------------------------------------------------------------------------------------------
    def _key(self, idx: tuple[int, int, int]) -> str:
        return self.sep.join(str(i) for i in idx)

    def _decode(self, data: bytes | None) -> np.ndarray | None:
        if data is None:
            return None
        buf = self.codec.decode(data) if self.codec else data
        for f in reversed(self.filters):
            buf = f.decode(buf)
        arr = np.frombuffer(buf, dtype=self.dtype)
        if arr.size != int(np.prod(self.chunks)):
            raise VolumeError(f"chunk has {arr.size} values, expected {int(np.prod(self.chunks))}")
        return arr.reshape(self.chunks)

    def _load(self, idx: tuple[int, int, int]) -> np.ndarray | None:
        return self._decode(self.store.get(self._key(idx)))

    def _chunk(self, idx: tuple[int, int, int]) -> np.ndarray | None:
        with self._lock:
            if idx in self._mem:
                self._mem.move_to_end(idx)
                return self._mem[idx]
        arr = self._load(idx)
        self._remember(idx, arr)
        return arr

    def _remember(self, idx, arr) -> None:
        with self._lock:
            self._mem[idx] = arr
            self._mem.move_to_end(idx)
            while len(self._mem) > self.max_chunks:
                self._mem.popitem(last=False)

    def prefetch(self, idxs) -> None:
        """Fetch and decode chunks in parallel into the memory LRU (at most max_chunks are kept)."""
        with self._lock:
            need = [i for i in idxs if i not in self._mem]
        need = need[: self.max_chunks]
        if not need:
            return
        with ThreadPoolExecutor(self.workers) as ex:
            for idx, arr in zip(need, ex.map(self._load, need)):
                self._remember(idx, arr)

    def _chunk_ranges(self, lo, hi):
        rngs = []
        for a in range(3):
            lo_a, hi_a = max(0, int(lo[a])), min(self.shape[a], int(hi[a]))
            if hi_a <= lo_a:
                return None
            rngs.append(range(lo_a // self.chunks[a], (hi_a - 1) // self.chunks[a] + 1))
        return rngs

    # ---- reads ----------------------------------------------------------------------------------------------
    def read_box(self, lo, hi) -> np.ndarray:
        """Dense box [lo, hi) in (z, y, x); outside the array reads as the fill value."""
        lo = [int(v) for v in lo]
        hi = [int(v) for v in hi]
        out = np.full([b - a for a, b in zip(lo, hi)], self.fill, dtype=self.dtype)
        rngs = self._chunk_ranges(lo, hi)
        if rngs is None:
            return out
        idxs = [(a, b, c) for a in rngs[0] for b in rngs[1] for c in rngs[2]]
        self.prefetch(idxs)
        for idx in idxs:
            arr = self._chunk(idx)
            if arr is None:
                continue
            c0 = [i * c for i, c in zip(idx, self.chunks)]
            s_lo = [max(a, c) for a, c in zip(lo, c0)]
            s_hi = [min(h, c + cs, sh) for h, c, cs, sh in zip(hi, c0, self.chunks, self.shape)]
            if any(b <= a for a, b in zip(s_lo, s_hi)):
                continue
            src = tuple(slice(a - c, b - c) for a, b, c in zip(s_lo, s_hi, c0))
            dst = tuple(slice(a - o, b - o) for a, b, o in zip(s_lo, s_hi, lo))
            out[dst] = arr[src]
        return out

    def sample(self, xyz: np.ndarray, order: int = 1, progress: bool = False) -> np.ndarray:
        """Interpolated values at points (..., 3) given as (x, y, z) voxels; NaN-coordinate points give NaN."""
        pts = np.asarray(xyz, dtype=np.float64)
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        out = np.full(len(flat), np.nan, np.float32)
        ok = np.isfinite(flat).all(1)
        if not ok.any():
            return out.reshape(shape)
        idx_ok = np.flatnonzero(ok)
        zyx = flat[idx_ok][:, ::-1]
        ch = np.array(self.chunks)
        key = np.floor(zyx / ch).astype(np.int64)
        order_ = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
        ks = key[order_]
        breaks = np.flatnonzero(np.any(np.diff(ks, axis=0) != 0, axis=1)) + 1
        groups = np.split(order_, breaks)
        batch = max(1, self.max_chunks // 16)
        t0 = time.time()
        for b0 in range(0, len(groups), batch):
            gb = groups[b0:b0 + batch]
            need = set()
            for g in gb:
                q = zyx[g]
                lo = np.floor(q.min(0)).astype(int) - 1
                hi = np.floor(q.max(0)).astype(int) + 2
                rngs = self._chunk_ranges(lo, hi)
                if rngs:
                    need.update((a, b_, c) for a in rngs[0] for b_ in rngs[1] for c in rngs[2])
            self.prefetch(sorted(need))
            for g in gb:
                q = zyx[g]
                lo = np.floor(q.min(0)).astype(int) - 1
                hi = np.floor(q.max(0)).astype(int) + 3
                box = self.read_box(lo, hi).astype(np.float32)
                rel = (q - lo).T
                out[idx_ok[g]] = map_coordinates(box, rel, order=order, mode="nearest")
            if progress and (b0 // batch) % 10 == 0:
                print(f"  sampled {min(b0 + batch, len(groups))}/{len(groups)} chunk groups "
                      f"({self.store.bytes_fetched / 1e9:.2f} GB fetched, {time.time() - t0:.0f}s)", flush=True)
        return out.reshape(shape)
