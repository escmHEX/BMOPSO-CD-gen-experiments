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

    def test_cli_accepts_overwrite_flag(self):
        args = _parse_args(["--filtered", "--overwrite"])

        self.assertTrue(args.overwrite)

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

    def test_filter_preserves_real_accents_as_composed_unicode(self):
        result = filter_tweet_text("La situacio\u0301n esta\u0301 mejorando con cuidado")

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "La situación está mejorando con cuidado")

    def test_filter_discards_real_emoji(self):
        result = filter_tweet_text(
            "Welp \U0001f937\U0001f3fd\u200d\u2640\ufe0f y'all had better add this to your bingo cards"
        )

        self.assertEqual(result.reason, "emoji")

    def test_filter_discards_mojibake_emoji_after_repair(self):
        result = filter_tweet_text(
            "Welp \u00f0\u0178\u00a4\u00b7\u00f0\u0178\u008f\u00bd"
            "\u00e2\u20ac\u008d\u00e2\u2122\u20ac\u00ef\u00b8\u008f"
            " y'all had better add this to your bingo cards"
        )

        self.assertEqual(result.reason, "emoji")

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

    def test_filter_discards_at_sign_remaining_after_cleanup(self):
        result = filter_tweet_text("Public update #COVID19 for contact@agency today")

        self.assertEqual(result.reason, "mention")

    def test_filter_strips_wrapping_double_quotes_from_text(self):
        result = filter_tweet_text('"Public health update remains readable today"')

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "Public health update remains readable today")

    def test_filter_removes_internal_double_quotes_from_text(self):
        result = filter_tweet_text('New study finds troubling signs of "brain complications" in severe COVID-19 cases')

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "New study finds troubling signs of brain complications in severe COVID-19 cases")

    def test_filter_removes_commas_from_text(self):
        result = filter_tweet_text("Texas, Florida, and Arizona have fewer COVID restrictions")

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "Texas Florida and Arizona have fewer COVID restrictions")

    def test_filter_discards_text_with_more_than_one_colon_after_cleanup(self):
        result = filter_tweet_text("Breaking: COVID update: cases rising today")

        self.assertEqual(result.reason, "too_many_colons")

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

            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with not_hydrated_path.open(encoding="utf-8-sig", newline="") as handle:
                missed = list(csv.DictReader(handle))

        self.assertEqual(summary.hydrated, 1)
        self.assertEqual(summary.not_hydrated, 1)
        self.assertEqual(rows, [{"tweetId": "3333333333333333333", "texto": "line one line two"}])
        self.assertEqual(missed[0]["tweetId"], "4444444444444444444")
        self.assertEqual(missed[0]["reason"], "not_found")

    def test_build_corpus_writes_utf8_bom_once_when_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "4545454545454545454\n"
                "4646464646464646464\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "corpus.csv"
            not_hydrated_path = root / "hydrated" / "corpus.not_hydrated.csv"
            report_path = root / "hydrated" / "corpus.report.json"

            asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=FakeHydrator({"4545454545454545454": "first normal text"}),
                    not_hydrated_path=not_hydrated_path,
                    report_path=report_path,
                    limit=1,
                )
            )
            asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=FakeHydrator({"4646464646464646464": "second normal text"}),
                    not_hydrated_path=not_hydrated_path,
                    report_path=report_path,
                    limit=1,
                )
            )

            data = output_path.read_bytes()

        self.assertTrue(data.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(data.count(b"\xef\xbb\xbf"), 1)

    def test_build_corpus_does_not_wrap_or_escape_internal_double_quotes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text("tweet_id\n4747474747474747474\n", encoding="utf-8")
            output_path = root / "hydrated" / "corpus.csv"

            asyncio.run(
                build_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=FakeHydrator({"4747474747474747474": 'first part, "quoted" second part'}),
                    not_hydrated_path=root / "hydrated" / "corpus.not_hydrated.csv",
                    report_path=root / "hydrated" / "corpus.report.json",
                    limit=1,
                )
            )

            raw_lines = output_path.read_text(encoding="utf-8-sig").splitlines()

        self.assertEqual(raw_lines[0], "tweetId,texto")
        self.assertEqual(raw_lines[1], '4747474747474747474,first part quoted second part')
        self.assertNotIn(',"', raw_lines[1])
        self.assertNotIn('\\"', raw_lines[1])
        self.assertNotIn("\\,", raw_lines[1])

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

            with output_path.open(encoding="utf-8-sig", newline="") as handle:
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

            with not_hydrated_path.open(encoding="utf-8-sig", newline="") as handle:
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
                "1616161616161616161\n"
                "1717171717171717171\n",
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
                    "1616161616161616161": "Emoji should be dropped \U0001f602 before counting as valid",
                    "1717171717171717171": "Second normal public health sentence",
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
                    max_attempts=5,
                )
            )

            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            with discarded_path.open(encoding="utf-8-sig", newline="") as handle:
                discarded = list(csv.DictReader(handle))
            with not_hydrated_path.open(encoding="utf-8-sig", newline="") as handle:
                missed = list(csv.DictReader(handle))

        self.assertEqual(summary.hydrated, 2)
        self.assertEqual(summary.not_hydrated, 1)
        self.assertEqual(hydrator.requested, [
            "1313131313131313131",
            "1414141414141414141",
            "1515151515151515151",
            "1616161616161616161",
            "1717171717171717171",
        ])
        self.assertEqual(
            rows,
            [
                {"tweetId": "1313131313131313131", "texto": "First normal public health sentence"},
                {"tweetId": "1717171717171717171", "texto": "Second normal public health sentence"},
            ],
        )
        self.assertEqual(discarded[0]["tweetId"], "1414141414141414141")
        self.assertEqual(discarded[0]["reason"], "mention")
        self.assertEqual(discarded[1]["tweetId"], "1616161616161616161")
        self.assertEqual(discarded[1]["reason"], "emoji")
        self.assertEqual(missed[0]["tweetId"], "1515151515151515151")

    def test_build_filtered_corpus_overwrite_removes_previous_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "2121212121212121212\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "filtered.csv"
            not_hydrated_path = root / "hydrated" / "filtered.not_hydrated.csv"
            discarded_path = root / "hydrated" / "filtered.discarded.csv"
            report_path = root / "hydrated" / "filtered.report.json"
            output_path.parent.mkdir()
            output_path.write_text("tweetId,texto\n9999999999999999999,old row\n", encoding="utf-8")
            not_hydrated_path.write_text("tweetId,reason\n8888888888888888888,not_found\n", encoding="utf-8")
            discarded_path.write_text("tweetId,reason,cleanedLength\n7777777777777777777,emoji,20\n", encoding="utf-8")
            report_path.write_text('{"old": true}', encoding="utf-8")

            summary = asyncio.run(
                build_filtered_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=FakeHydrator({"2121212121212121212": "Fresh normal public health sentence"}),
                    not_hydrated_path=not_hydrated_path,
                    discarded_path=discarded_path,
                    report_path=report_path,
                    target_valid=1,
                    max_attempts=1,
                    overwrite=True,
                )
            )

            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(summary.output_rows, 1)
        self.assertEqual(summary.not_hydrated_rows, 0)
        self.assertEqual(summary.discarded_rows, 0)
        self.assertEqual(rows, [{"tweetId": "2121212121212121212", "texto": "Fresh normal public health sentence"}])

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
