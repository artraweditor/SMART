"""Download a model file, checking its size and sha256."""

import hashlib
import os
import urllib.request


class DownloadError(Exception):
    pass


def download(url, dest, size=None, sha=None, progress=None, cancelled=None):
    """Download `url` to `dest`. `progress(n, total)` is called per chunk
    (total may be 0 if unknown); `cancelled()` returning True aborts.
    Returns the sha256 hex digest. Raises DownloadError on failure."""
    dest = str(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + '.part'
    h = hashlib.sha256()
    n = 0
    try:
        with urllib.request.urlopen(url) as r, open(part, 'wb') as out:
            total = size or int(r.headers.get('Content-Length') or 0)
            while True:
                if cancelled and cancelled():
                    raise DownloadError('download cancelled')
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                n += len(chunk)
                if progress:
                    progress(n, total)
    except BaseException:
        if os.path.exists(part):
            os.unlink(part)
        raise
    if size is not None and n != size:
        os.unlink(part)
        raise DownloadError(f'expected {size} bytes, got {n}')
    if sha is not None and h.hexdigest() != sha:
        os.unlink(part)
        raise DownloadError('checksum mismatch')
    os.replace(part, dest)
    return h.hexdigest()
