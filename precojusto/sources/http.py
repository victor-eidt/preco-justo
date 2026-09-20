"""One tiny GET with retries. Both APIs are public, keyless, and flaky."""

import json
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "preco-justo/0.1 (academic; slow crawl)"}


def get_json(url, tries=4, timeout=60, on_wait=None):
    """GET ``url`` and parse JSON. Returns ``None`` after ``tries`` failures.

    204 is an empty result, not an error. 429 waits longer than other codes.
    """
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return None
            wait = 90 if e.code == 429 else 12 * (attempt + 1)
            err = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001 - timeouts, bad JSON, resets
            wait = 12 * (attempt + 1)
            err = f"{type(e).__name__}: {e}"
        if attempt == tries - 1:
            return None
        if on_wait:
            on_wait(wait, err)
        time.sleep(wait)
    return None
