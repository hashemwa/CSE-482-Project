"""Input-source helpers for batch sentiment prediction."""

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen


XQUIK_SEARCH_URL = "https://xquik.com/api/v1/x/tweets/search"
XQUIK_API_CONTRACT = "2026-04-29"
TEXT_COLUMN_HINTS = (
    "review_text",
    "review",
    "tweet",
    "text",
    "content",
    "message",
    "comment",
    "description",
    "body",
)


def detect_text_column(columns):
    """Return the most likely text column, or None when no match is safe."""
    normalized_columns = {str(column).lower().strip(): column for column in columns}

    for hint in TEXT_COLUMN_HINTS:
        if hint in normalized_columns:
            return normalized_columns[hint]

    for column in columns:
        normalized = str(column).lower().strip()
        if any(hint in normalized for hint in TEXT_COLUMN_HINTS):
            return column

    return None


def fetch_xquik_posts(query, api_key, limit=50, open_url=urlopen):
    """Fetch public X posts from Xquik without persisting the API key."""
    normalized_query = query.strip()
    normalized_key = api_key.strip()
    if not normalized_query:
        raise ValueError("Enter an X search query.")
    if not normalized_key:
        raise ValueError("Enter an Xquik API key.")

    query_string = urlencode({
        "q": normalized_query,
        "queryType": "Latest",
        "limit": max(1, min(int(limit), 100)),
    })
    request = Request(
        f"{XQUIK_SEARCH_URL}?{query_string}",
        headers={
            "Accept": "application/json",
            "x-api-key": normalized_key,
            "xquik-api-contract": XQUIK_API_CONTRACT,
        },
    )
    with open_url(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict) or not isinstance(payload.get("tweets"), list):
        raise ValueError("Xquik returned an unexpected response.")
    if any(not isinstance(tweet, dict) for tweet in payload["tweets"]):
        raise ValueError("Xquik returned an invalid tweet record.")

    return payload["tweets"]
