import asyncio
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_lamsal_covid_corpus import (
    DEFAULT_COOKIES_FILE,
    DEFAULT_FILTERED_OUTPUT,
    build_filtered_corpus,
    build_corpus,
    discover_tweet_ids,
    filter_tweet_text,
    normalize_tweet_text,
    _parse_args,
    _resolve_output_path,
)


class FakeHydrator:
    def __init__(self, tweets):
        self.tweets = tweets
        self.requested = []

    async def fetch_text(self, tweet_id):
        self.requested.append(tweet_id)
        return self.tweets.get(tweet_id)


class SlowHydrator:
    async def fetch_text(self, tweet_id):
        await asyncio.sleep(10)
        return "too late"


class LamsalCovidCorpusBuilderTest(unittest.TestCase):
    def test_cli_defaults_to_local_cookie_file(self):
        args = _parse_args([])

        self.assertEqual(Path(args.cookies_file), DEFAULT_COOKIES_FILE)
        self.assertEqual(Path(args.cookies_file).name, "cookies.txt")

    def test_filtered_cli_uses_filtered_default_output(self):
        args = _parse_args(["--filtered"])

        self.assertEqual(_resolve_output_path(args), DEFAULT_FILTERED_OUTPUT)

    def test_discovers_default_id_columns_and_deduplicates_in_stable_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "ids.csv").write_text(
                "tweet_id,sentiment\n"
                "1234567890123456789,positive\n"
                "9876543210987654321,negative\n"
                "1234567890123456789,positive\n",
                encoding="utf-8",
            )

            ids = discover_tweet_ids(input_dir)

        self.assertEqual(ids, ["1234567890123456789", "9876543210987654321"])

    def test_discovers_custom_id_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "custom.csv").write_text(
                "source_id,label\n"
                "1111111111111111111,neutral\n",
                encoding="utf-8",
            )

            ids = discover_tweet_ids(input_dir, id_column="source_id")

        self.assertEqual(ids, ["1111111111111111111"])

    def test_discovers_ids_inside_zip_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            zip_path = input_dir / "ieee_download.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr(
                    "nested/tweets.csv",
                    "status_id,sentiment\n2222222222222222222,positive\n",
                )

            ids = discover_tweet_ids(input_dir)

        self.assertEqual(ids, ["2222222222222222222"])

    def test_normalizes_text_to_single_line(self):
        text = normalize_tweet_text("  hello\n\tworld  https://t.co/example  ")

        self.assertEqual(text, "hello world https://t.co/example")

    def test_filter_repairs_mojibake_accents(self):
        result = filter_tweet_text("La situaciÃ³n estÃ¡ mejorando con cuidado")

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "La situación está mejorando con cuidado")

    def test_filter_removes_urls_and_hashtags_from_kept_text(self):
        result = filter_tweet_text("Public health update https://t.co/x pic.twitter.com/abc #COVID19 #Health")

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "Public health update")

    def test_filter_discards_hashtag_heavy_text(self):
        result = filter_tweet_text("#COVID19 #StayHome #Pandemic")

        self.assertEqual(result.reason, "hashtag_heavy")

    def test_filter_discards_any_user_mention(self):
        result = filter_tweet_text("Important update from @healthagency about prevention")

        self.assertEqual(result.reason, "mention")

    def test_filter_discards_text_shorter_than_15_characters_after_cleaning(self):
        result = filter_tweet_text("Ok https://t.co/x")

        self.assertEqual(result.reason, "too_short")

    def test_build_corpus_hydrates_with_fake_provider_and_records_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "3333333333333333333\n"
                "4444444444444444444\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "corpus.csv"
            not_hydrated_path = root / "hydrated" / "corpus.not_hydrated.csv"
            report_path = root / "hydrated" / "corpus.report.json"
            hydrator = FakeHydrator(
                {
                    "3333333333333333333": "line one\nline two",
                    "4444444444444444444": None,
                }
            )

            summary = asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=not_hydrated_path,
                    report_path=report_path,
                )
            )

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with not_hydrated_path.open(encoding="utf-8", newline="") as handle:
                missed = list(csv.DictReader(handle))

        self.assertEqual(summary.hydrated, 1)
        self.assertEqual(summary.not_hydrated, 1)
        self.assertEqual(rows, [{"tweetId": "3333333333333333333", "texto": "line one line two"}])
        self.assertEqual(missed[0]["tweetId"], "4444444444444444444")
        self.assertEqual(missed[0]["reason"], "not_found")

    def test_build_corpus_resumes_by_skipping_existing_output_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "5555555555555555555\n"
                "6666666666666666666\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "corpus.csv"
            output_path.parent.mkdir()
            output_path.write_text(
                "tweetId,texto\n"
                "5555555555555555555,already hydrated\n",
                encoding="utf-8",
            )
            hydrator = FakeHydrator({"6666666666666666666": "new text"})

            summary = asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=root / "hydrated" / "corpus.not_hydrated.csv",
                    report_path=root / "hydrated" / "corpus.report.json",
                )
            )

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(hydrator.requested, ["6666666666666666666"])
        self.assertEqual(summary.skipped_existing, 1)
        self.assertEqual(
            rows,
            [
                {"tweetId": "5555555555555555555", "texto": "already hydrated"},
                {"tweetId": "6666666666666666666", "texto": "new text"},
            ],
        )

    def test_build_corpus_resumes_by_skipping_previously_not_hydrated_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "8888888888888888888\n"
                "9999999999999999999\n",
                encoding="utf-8",
            )
            not_hydrated_path = root / "hydrated" / "corpus.not_hydrated.csv"
            not_hydrated_path.parent.mkdir()
            not_hydrated_path.write_text(
                "tweetId,reason\n"
                "8888888888888888888,not_found\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "corpus.csv"
            hydrator = FakeHydrator({"9999999999999999999": "later text"})

            summary = asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=not_hydrated_path,
                    report_path=root / "hydrated" / "corpus.report.json",
                )
            )

        self.assertEqual(hydrator.requested, ["9999999999999999999"])
        self.assertEqual(summary.skipped_existing, 1)

    def test_build_corpus_stops_scanning_input_after_limit_is_reached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "01-valid.csv").write_text(
                "7777777777777777777,0\n",
                encoding="utf-8",
            )
            (input_dir / "02-invalid.csv").write_text(
                "not_a_tweet_id,0\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "corpus.csv"
            hydrator = FakeHydrator({"7777777777777777777": "limited text"})

            summary = asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=root / "hydrated" / "corpus.not_hydrated.csv",
                    report_path=root / "hydrated" / "corpus.report.json",
                    limit=1,
                )
            )

        self.assertEqual(summary.selected, 1)
        self.assertEqual(hydrator.requested, ["7777777777777777777"])

    def test_build_corpus_marks_tweet_timeout_without_blocking_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text("tweet_id\n1212121212121212121\n", encoding="utf-8")
            not_hydrated_path = root / "hydrated" / "corpus.not_hydrated.csv"

            summary = asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=root / "hydrated" / "corpus.csv",
                    hydrator=SlowHydrator(),
                    not_hydrated_path=not_hydrated_path,
                    report_path=root / "hydrated" / "corpus.report.json",
                    limit=1,
                    tweet_timeout_seconds=0.01,
                )
            )

            with not_hydrated_path.open(encoding="utf-8", newline="") as handle:
                missed = list(csv.DictReader(handle))

        self.assertEqual(summary.not_hydrated, 1)
        self.assertEqual(missed, [{"tweetId": "1212121212121212121", "reason": "timeout"}])

    def test_build_filtered_corpus_continues_until_target_valid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "1313131313131313131\n"
                "1414141414141414141\n"
                "1515151515151515151\n"
                "1616161616161616161\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "filtered.csv"
            not_hydrated_path = root / "hydrated" / "filtered.not_hydrated.csv"
            discarded_path = root / "hydrated" / "filtered.discarded.csv"
            hydrator = FakeHydrator(
                {
                    "1313131313131313131": "First normal public health sentence",
                    "1414141414141414141": "Mention from @user should be dropped",
                    "1515151515151515151": None,
                    "1616161616161616161": "Second normal public health sentence",
                }
            )

            summary = asyncio.run(
                build_filtered_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=not_hydrated_path,
                    discarded_path=discarded_path,
                    report_path=root / "hydrated" / "filtered.report.json",
                    target_valid=2,
                    max_attempts=4,
                )
            )

            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with discarded_path.open(encoding="utf-8", newline="") as handle:
                discarded = list(csv.DictReader(handle))
            with not_hydrated_path.open(encoding="utf-8", newline="") as handle:
                missed = list(csv.DictReader(handle))

        self.assertEqual(summary.hydrated, 2)
        self.assertEqual(summary.not_hydrated, 1)
        self.assertEqual(hydrator.requested, [
            "1313131313131313131",
            "1414141414141414141",
            "1515151515151515151",
            "1616161616161616161",
        ])
        self.assertEqual(
            rows,
            [
                {"tweetId": "1313131313131313131", "texto": "First normal public health sentence"},
                {"tweetId": "1616161616161616161", "texto": "Second normal public health sentence"},
            ],
        )
        self.assertEqual(discarded[0]["tweetId"], "1414141414141414141")
        self.assertEqual(discarded[0]["reason"], "mention")
        self.assertEqual(missed[0]["tweetId"], "1515151515151515151")

    def test_build_filtered_corpus_reports_accumulated_resume_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "1717171717171717171\n"
                "1818181818181818181\n"
                "1919191919191919191\n"
                "2020202020202020202\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "filtered.csv"
            not_hydrated_path = root / "hydrated" / "filtered.not_hydrated.csv"
            discarded_path = root / "hydrated" / "filtered.discarded.csv"
            report_path = root / "hydrated" / "filtered.report.json"
            output_path.parent.mkdir()
            output_path.write_text(
                "tweetId,texto\n"
                "1717171717171717171,Existing normal text row\n",
                encoding="utf-8",
            )
            not_hydrated_path.write_text(
                "tweetId,reason\n"
                "1818181818181818181,not_found\n",
                encoding="utf-8",
            )
            discarded_path.write_text(
                "tweetId,reason,cleanedLength\n"
                "1919191919191919191,mention,32\n",
                encoding="utf-8",
            )
            hydrator = FakeHydrator({"2020202020202020202": "New normal public health sentence"})

            summary = asyncio.run(
                build_filtered_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=not_hydrated_path,
                    discarded_path=discarded_path,
                    report_path=report_path,
                    target_valid=2,
                    max_attempts=10,
                )
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(summary.hydrated, 1)
        self.assertEqual(summary.output_rows, 2)
        self.assertEqual(summary.not_hydrated_rows, 1)
        self.assertEqual(summary.discarded_rows, 1)
        self.assertEqual(report["target_valid"], 2)
        self.assertEqual(report["max_attempts"], 10)
        self.assertEqual(report["output_rows"], 2)
        self.assertEqual(report["not_hydrated_rows"], 1)
        self.assertEqual(report["discarded_rows"], 1)


if __name__ == "__main__":
    unittest.main()
