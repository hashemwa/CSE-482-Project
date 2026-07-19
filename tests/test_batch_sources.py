import json
import unittest
from urllib.parse import parse_qs, urlparse

from batch_sources import detect_text_column, fetch_xquik_posts


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class BatchSourcesTest(unittest.TestCase):
    def test_detects_review_columns(self):
        self.assertEqual(
            detect_text_column(["id", "review", "rating"]),
            "review",
        )
        self.assertEqual(
            detect_text_column(["id", "customer_review_body"]),
            "customer_review_body",
        )

    def test_requires_explicit_selection_for_unknown_columns(self):
        self.assertIsNone(detect_text_column(["id", "rating"]))

    def test_fetches_xquik_posts_with_bounded_query_and_private_header(self):
        observed = {}

        def open_url(request, timeout):
            observed["request"] = request
            observed["timeout"] = timeout
            return FakeResponse({"tweets": [{"id": "1", "text": "Useful launch"}]})

        posts = fetch_xquik_posts(" launch news ", " xq_test ", 500, open_url)
        request = observed["request"]
        query = parse_qs(urlparse(request.full_url).query)

        self.assertEqual(posts, [{"id": "1", "text": "Useful launch"}])
        self.assertEqual(
            query,
            {"q": ["launch news"], "queryType": ["Latest"], "limit": ["100"]},
        )
        self.assertEqual(request.get_header("X-api-key"), "xq_test")
        self.assertEqual(request.get_header("Xquik-api-contract"), "2026-04-29")
        self.assertEqual(observed["timeout"], 30)

    def test_rejects_missing_credentials_and_invalid_payloads(self):
        with self.assertRaisesRegex(ValueError, "query"):
            fetch_xquik_posts(" ", "xq_test")
        with self.assertRaisesRegex(ValueError, "API key"):
            fetch_xquik_posts("launch", " ")

        def open_url(_request, timeout):
            self.assertEqual(timeout, 30)
            return FakeResponse({"tweets": "invalid"})

        with self.assertRaisesRegex(ValueError, "unexpected response"):
            fetch_xquik_posts("launch", "xq_test", open_url=open_url)


if __name__ == "__main__":
    unittest.main()
