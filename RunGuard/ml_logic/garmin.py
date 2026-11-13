"""Minimal Garmin Personal API client for RunGuard (Python 3.10 compatible).

Adapted from Garmin's official client (Apache-2.0, requires Python >= 3.12):
https://github.com/garmin/connect-personal-api-python

What it does:
1. Reads GARMIN_PERSONAL_API_KEY from a .env file.
2. Exchanges the API key for a bearer token (a JWT) and caches it on disk.
   The cached token is reused until it is within 1 hour of expiry, then refreshed.
3. Fetches all activities in an adjustable date window (default: last 90 days),
   paging through the results.

Setup:
    pip install requests python-dotenv
    # .env (add .env AND the token cache file to .gitignore!)
    GARMIN_PERSONAL_API_KEY=your-key

Usage:
    python garmin.py --days 90
    # or from code:
    from garmin import fetch_activities
    activities = fetch_activities(days=90)
"""

import argparse
import base64
import hashlib
import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# --- Constants (taken from Garmin's constants.py) ---
BASE_URL = "https://personal.connectapi.garmin.com"
TOKEN_PATH = "/personalapitoken-service/personal-api/personal-api-key/token"
ACTIVITIES_PATH = "/activitylist-service/activities/search/activities"
TIMEOUT_SECONDS = 30
REFRESH_WINDOW_SECONDS = 60 * 60  # refresh if the token expires within 1 hour
PAGE_SIZE = 100  # activities per request when paging

# Token cache lives next to this file by default; keep it out of git.
# Override with GARMIN_TOKEN_CACHE (e.g. /tmp/.garmin_token.json on Cloud Run).
TOKEN_CACHE_FILE = Path(os.getenv("GARMIN_TOKEN_CACHE", Path(__file__).parent / ".garmin_token.json"))


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
def load_api_key() -> str:
    """Load the Personal API key from .env (or an already-set env variable)."""
    load_dotenv()
    key = os.getenv("GARMIN_PERSONAL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GARMIN_PERSONAL_API_KEY is missing or empty in .env")
    return key


def _jwt_expiry(token: str) -> float:
    """Read the 'exp' claim (epoch seconds) from a JWT, without verifying it."""
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)  # restore base64 padding
    return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])


def _read_cached_token() -> Optional[dict]:
    """Return the cached token dict, or None if no usable cache exists."""
    try:
        cache = json.loads(TOKEN_CACHE_FILE.read_text(encoding="utf-8"))
        return {"access_token": cache["access_token"], "expires_at": float(cache["expires_at"])}
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _write_cached_token(access_token: str, expires_at: float) -> None:
    """Save the token with owner-only permissions (0600) - it is a credential."""
    fd = os.open(TOKEN_CACHE_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"access_token": access_token, "expires_at": expires_at}, f)


class GarminAuthError(Exception):
    """Garmin rejected the API key (invalid, revoked or not authorised)."""


# Tokens of user-supplied keys: kept in memory only, keyed by a hash of the key,
# so different users never share a token and no key is ever written to disk.
_user_tokens: dict = {}


def _exchange_key(api_key: str) -> str:
    """Exchange an API key for a new bearer token."""
    response = requests.post(
        f"{BASE_URL}{TOKEN_PATH}",
        headers={"Accept": "*/*", "Authorization": f"ApiKey {api_key}"},
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code in (401, 403):
        raise GarminAuthError("Garmin rejected the API key")
    response.raise_for_status()
    return response.text.strip()


def get_token(api_key: Optional[str] = None, force_refresh: bool = False) -> str:
    """Return a valid bearer token, reusing the cache unless it is (nearly) expired.

    api_key=None uses the server's own key from .env / the environment and the
    token cache file. A user-supplied api_key uses an in-memory cache instead.
    Set force_refresh=True to always request a new token (e.g. after a 401).
    """
    if api_key is None:
        cached = None if force_refresh else _read_cached_token()
    else:
        cache_key = hashlib.sha256(api_key.encode()).hexdigest()
        cached = None if force_refresh else _user_tokens.get(cache_key)

    if cached and cached["expires_at"] - time.time() > REFRESH_WINDOW_SECONDS:
        return cached["access_token"]

    access_token = _exchange_key(api_key if api_key is not None else load_api_key())
    expires_at = _jwt_expiry(access_token)
    if api_key is None:
        _write_cached_token(access_token, expires_at)
    else:
        _user_tokens[cache_key] = {"access_token": access_token, "expires_at": expires_at}
    return access_token


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def _get(path: str, params: Optional[dict] = None, api_key: Optional[str] = None):
    """Authenticated GET. On a 401 (e.g. revoked token), refresh once and retry."""
    for attempt in range(2):
        token = get_token(api_key, force_refresh=(attempt == 1))
        response = requests.get(
            f"{BASE_URL}{path}",
            headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
            params=params,
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code == 401:
            if attempt == 0:
                continue  # token rejected -> retry with a fresh one
            raise GarminAuthError("Garmin rejected the token")
        response.raise_for_status()
        return response.json() if response.content else None


def fetch_activities(
    days: int = 90,
    end_date: Optional[date] = None,
    activity_type: Optional[str] = None,
    api_key: Optional[str] = None,
) -> list:
    """Fetch all activities from the `days` days up to and including `end_date`.

    Args:
        days: size of the look-back window (default 90).
        end_date: last day of the window (default today).
        activity_type: optional Garmin activity type filter, e.g. "running".
        api_key: a user's Personal API key (default: the server's own key).

    Returns:
        A list of activity summary dicts, as returned by Garmin.
    """
    end_date = end_date or date.today()
    start_date = end_date - timedelta(days=days)

    activities = []
    start = 0
    while True:
        params = {
            "start": start,
            "limit": PAGE_SIZE,
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
        }
        if activity_type:
            params["activityType"] = activity_type

        page = _get(ACTIVITIES_PATH, params=params, api_key=api_key) or []
        activities.extend(page)
        if len(page) < PAGE_SIZE:  # last page reached
            break
        start += PAGE_SIZE

    return activities


# ---------------------------------------------------------------------------
# Feature building: raw Garmin JSON -> 1 row of the 28 FEATURE_CONTRACT columns
# ---------------------------------------------------------------------------
# Contract section 3, translated from Strava activity types to Garmin typeKeys.
GARMIN_RUN_TYPES = {"running", "trail_running", "treadmill_running", "track_running",
                    "indoor_running", "virtual_run"}
GARMIN_STRENGTH_TYPES = {"strength_training", "hiit"}  # Strava: Workout, Weight Training, Crossfit
GARMIN_ALTERNATIVE_TYPES = {"cycling", "road_biking", "indoor_cycling", "virtual_ride",
                            "mountain_biking", "gravel_cycling", "lap_swimming",
                            "open_water_swimming", "hiking", "walking", "yoga", "pilates"}

# Garmin typeKey -> the Strava "Activity Type" with the same contract group,
# so the Strava pipeline (strava.py) can apply its rules unchanged
_GARMIN_TO_STRAVA_TYPE = {
    **{t: "Run" for t in GARMIN_RUN_TYPES},
    **{t: "Weight Training" for t in GARMIN_STRENGTH_TYPES},
    **{t: "Ride" for t in GARMIN_ALTERNATIVE_TYPES},
}


def process_data(source, event_day=None):
    """Turn Garmin activities into the model input row.

    Selects only the columns the contract needs (start time, type, distance,
    moving and elapsed time), then builds 4 base features x 7 days = 28 columns
    in FEATURE_CONTRACT.md order, using the same rules as the Strava pipeline.

    Args:
        source: the list returned by fetch_activities(), or a path to the JSON
            written by the CLI below.
        event_day: date to predict for (default: the day after the last activity).

    Returns:
        A pandas DataFrame with 1 row (indexed by event day) and 28 columns.
    """
    import pandas as pd

    from RunGuard.ml_logic.strava import CONTRACT_COLUMNS, build_features, prepare_activities

    if isinstance(source, list):
        data = source
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    raw = pd.json_normalize(data)
    if raw.empty:
        return build_features(pd.DataFrame(), event_day)

    # Relevant columns only. GMT matches the Strava export, which is in UTC (decision 1).
    moving = raw["movingDuration"] if "movingDuration" in raw else raw["duration"]
    selected = pd.DataFrame({
        "start": pd.to_datetime(raw["startTimeGMT"]),
        "type": raw["activityType.typeKey"].map(_GARMIN_TO_STRAVA_TYPE),  # unmapped -> excluded
        "distance_m": pd.to_numeric(raw["distance"], errors="coerce").fillna(0),
        "elapsed_s": pd.to_numeric(raw["elapsedDuration"], errors="coerce").fillna(0),
        "moving_s": pd.to_numeric(moving, errors="coerce").fillna(raw["duration"]).fillna(0),
    })

    # Rename to the Strava export layout and reuse its cleaning + feature building
    strava_like = pd.DataFrame({
        "Activity Date": selected["start"].dt.strftime("%b %d, %Y, %I:%M:%S %p"),
        "Activity Type": selected["type"],
        "Distance.1": selected["distance_m"],
        "Elapsed Time": selected["elapsed_s"],
        "Moving Time": selected["moving_s"],
    })
    activities = prepare_activities(strava_like)
    return build_features(activities, event_day)[CONTRACT_COLUMNS]


# ---------------------------------------------------------------------------
# CLI: fetch and save the raw JSON (keeps data lineage reproducible)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Garmin activities")
    parser.add_argument("--days", type=int, default=90, help="look-back window in days")
    parser.add_argument("--type", dest="activity_type", default=None, help='e.g. "running"')
    parser.add_argument("--out", default=None, help="path to save raw JSON")
    args = parser.parse_args()

    acts = fetch_activities(days=args.days, activity_type=args.activity_type)
    print(f"Fetched {len(acts)} activities from the last {args.days} days")

    out = Path(args.out or f"garmin_activities_{date.today().isoformat()}_{args.days}d.json")
    out.write_text(json.dumps(acts, indent=2), encoding="utf-8")
    print(f"Saved raw data to {out}")
