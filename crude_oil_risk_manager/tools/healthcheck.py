"""Exit 0 when the dashboard answers on its port (a 401 from the login counts: it is up), else 1. Used by the Docker HEALTHCHECK."""

import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    url = f"http://127.0.0.1:{os.environ.get('APP_PORT', '8050')}/_dash-layout"
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as exc:
        return 0 if exc.code == 401 else 1
    except OSError:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
