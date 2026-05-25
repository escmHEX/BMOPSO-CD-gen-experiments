from __future__ import annotations

import unittest

from llm_studio import extract_lm_studio_text, lm_studio_url, normalize_lm_studio_models, usage_tokens


class LmStudioSharedClientHelpersTests(unittest.TestCase):
    def test_builds_native_and_openai_urls(self):
        self.assertEqual(lm_studio_url("http://127.0.0.1:1234/", "native", "/chat"), "http://127.0.0.1:1234/api/v1/chat")
        self.assertEqual(lm_studio_url("http://127.0.0.1:1234", "openai", "/models"), "http://127.0.0.1:1234/v1/models")

    def test_extracts_native_output_text(self):
        payload = {"output_text": " rewritten component "}
        self.assertEqual(extract_lm_studio_text(payload, "native"), "rewritten component")

    def test_extracts_openai_message_text(self):
        payload = {"choices": [{"message": {"content": [{"text": "hello"}, {"text": "world"}]}}]}
        self.assertEqual(extract_lm_studio_text(payload, "openai"), "hello\nworld")

    def test_normalizes_native_models(self):
        payload = {
            "models": [
                {"type": "embedding", "key": "ignore"},
                {
                    "type": "llm",
                    "key": "qwen",
                    "display_name": "Qwen",
                    "loaded_instances": [{"id": "qwen-loaded"}],
                },
            ]
        }
        self.assertEqual(
            normalize_lm_studio_models(payload, "native"),
            [{"id": "qwen-loaded", "label": "Qwen (cargado)", "loaded": True}],
        )

    def test_usage_tokens_accepts_openai_names(self):
        self.assertEqual(
            usage_tokens({"usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}),
            (2, 3, 5),
        )


if __name__ == "__main__":
    unittest.main()
