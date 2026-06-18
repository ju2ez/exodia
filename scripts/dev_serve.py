#!/usr/bin/env python3
"""Build the site from committed data and serve it locally for preview.

Usage:
    python scripts/dev_serve.py            # build + serve on http://localhost:8000
    PORT=9000 python scripts/dev_serve.py  # custom port

Run `python -m exodia run-all --dry-run` first to populate data/ if it is empty.
"""

from __future__ import annotations

import functools
import http.server
import os
import socketserver

from exodia.config import load_settings
from exodia.logging_setup import setup_logging
from exodia.render import render_site


def main() -> None:
    setup_logging()
    settings = load_settings()
    site_dir = render_site(settings)

    port = int(os.environ.get("PORT", "8000"))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(site_dir))
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving {site_dir} at http://localhost:{port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
