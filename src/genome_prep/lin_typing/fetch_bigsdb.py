"""Fetch an authenticated BIGSdb REST resource (in practice, a scheme's ``profiles_csv``).

MiST ships ``resources/pubmlst/download_bigsdb.py``, but its session-token refresh calls
``OAuth1Session.get`` without ``params``, and rauth >= 0.7.3 then fails with
``TypeError: argument of type 'NoneType' is not iterable``. Once the cached session token expires —
they are short-lived — the shipped script cannot renew it and every download fails. This does the
same OAuth1 flow with that one argument supplied.

Credentials are read from a token directory holding ``client_credentials`` and ``access_tokens`` in
configparser format, keyed by ``--key-name``. **No credential is ever read from, written to, or
logged by this repository** — point ``--token-dir`` at a directory outside the checkout.
"""

from __future__ import annotations

import argparse
import configparser
import logging
import re
import sys
from pathlib import Path

from rauth import OAuth1Session

logger = logging.getLogger(__name__)

BASE_API = {
    "PubMLST": "https://rest.pubmlst.org",
    "Pasteur": "https://bigsdb.pasteur.fr/api",
}
_USER_AGENT = {"User-Agent": "BIGSdb downloader"}


def read_token(token_dir: Path, kind: str, key_name: str) -> tuple[str, str]:
    """Read a ``(token, secret)`` pair from ``<token_dir>/<kind>_tokens``.

    Parameters
    ----------
    token_dir
        Directory holding the credential files. Must be outside the repository.
    kind
        ``"access"`` or ``"session"``.
    key_name
        Section name within the file, e.g. ``"Pasteur"``.
    """
    path = token_dir / f"{kind}_tokens"
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    if not config.has_section(key_name):
        raise SystemExit(f"{path} has no [{key_name}] section (has {config.sections()})")
    return config[key_name]["token"], config[key_name]["secret"]


def read_client(token_dir: Path, key_name: str) -> tuple[str, str]:
    """Read the ``(client_id, client_secret)`` pair for ``key_name``."""
    path = token_dir / "client_credentials"
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    if not config.has_section(key_name):
        raise SystemExit(f"{path} has no [{key_name}] section (has {config.sections()})")
    return config[key_name]["client_id"], config[key_name]["client_secret"]


def db_from_url(url: str) -> str:
    """Extract the BIGSdb database name from a REST URL."""
    match = re.search(r"/db/([^/]+)", url)
    if not match:
        raise SystemExit(f"no /db/<name> segment in {url!r}")
    return match.group(1)


def new_session_token(token_dir: Path, *, key_name: str, site: str, db: str) -> tuple[str, str]:
    """Exchange the long-lived access token for a fresh session token.

    ``params={}`` is the whole point of this function: rauth iterates ``params`` unconditionally, so
    omitting it — as MiST's bundled script does — raises ``TypeError`` instead of renewing.
    """
    client_id, client_secret = read_client(token_dir, key_name)
    access_token, access_secret = read_token(token_dir, "access", key_name)
    session = OAuth1Session(
        client_id, client_secret, access_token=access_token, access_token_secret=access_secret
    )
    url = f"{BASE_API[site]}/db/{db}/oauth/get_session_token"
    response = session.get(url, params={}, headers=_USER_AGENT)
    if response.status_code != 200:
        raise SystemExit(f"session-token request failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    token, secret = payload["oauth_token"], payload["oauth_token_secret"]

    path = token_dir / "session_tokens"
    config = configparser.ConfigParser(interpolation=None)
    config.read(path)
    config[key_name] = {"token": token, "secret": secret}
    with path.open("w") as handle:
        config.write(handle)
    path.chmod(0o600)
    logger.info("refreshed session token for [%s]", key_name)
    return token, secret


def fetch(url: str, token_dir: Path, *, key_name: str, site: str) -> str:
    """GET ``url`` with OAuth1, renewing the session token once if it has expired."""
    client_id, client_secret = read_client(token_dir, key_name)
    db = db_from_url(url)
    try:
        token, secret = read_token(token_dir, "session", key_name)
    except (SystemExit, KeyError):
        token, secret = new_session_token(token_dir, key_name=key_name, site=site, db=db)

    for attempt in (1, 2):
        session = OAuth1Session(client_id, client_secret, access_token=token, access_token_secret=secret)
        response = session.get(url, params={}, headers=_USER_AGENT)
        if response.status_code == 200:
            return response.text
        if response.status_code == 401 and attempt == 1:
            logger.info("session token rejected, renewing")
            token, secret = new_session_token(token_dir, key_name=key_name, site=site, db=db)
            continue
        raise SystemExit(f"GET {url} failed ({response.status_code}): {response.text[:300]}")
    raise SystemExit("unreachable")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="BIGSdb REST URL, e.g. .../schemes/18/profiles_csv")
    p.add_argument("--token-dir", type=Path, required=True, help="Directory with credentials. Keep it out of git.")
    p.add_argument("--key-name", default="Pasteur")
    p.add_argument("--site", default="Pasteur", choices=sorted(BASE_API))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    text = fetch(args.url, args.token_dir, key_name=args.key_name, site=args.site)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    logger.info("wrote %s (%d bytes, %d lines)", args.output, len(text), text.count("\n"))


if __name__ == "__main__":
    main()
