import asyncio
import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_lamsal_covid_corpus import (
    CachedTweetTextHydrator,
    DEFAULT_COOKIES_FILE,
    DEFAULT_FILTERED_OUTPUT,
    audit_quality_tsv,
    build_filtered_corpus,
    build_corpus,
    build_quality_corpus,
    build_quality_corpus_from_cache,
    discover_tweet_ids,
    filter_quality_tweet_text,
    filter_tweet_text,
    iter_hydration_skip_cache_paths,
    iter_hydrated_text_cache_records,
    iter_hydrated_text_cache_paths,
    load_hydration_skip_ids,
    load_hydrated_text_cache,
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

    def test_cli_accepts_quality_tsv_flags(self):
        args = _parse_args(["--filtered", "--quality", "--format", "tsv", "--target-valid", "50"])

        self.assertTrue(args.quality)
        self.assertEqual(args.output_format, "tsv")
        self.assertEqual(args.target_valid, 50)

    def test_cli_accepts_quality_cache_only_flags(self):
        args = _parse_args(
            [
                "--filtered",
                "--quality",
                "--format",
                "tsv",
                "--cache-only",
                "--cache-source",
                "cache.tsv",
            ]
        )

        self.assertTrue(args.cache_only)
        self.assertEqual(args.cache_source, ["cache.tsv"])

    def test_cached_hydrator_uses_local_text_before_fallback(self):
        fallback = FakeHydrator({"2222222222222222222": "remote text"})
        hydrator = CachedTweetTextHydrator(
            cache={"1111111111111111111": "cached text"},
            fallback=fallback,
        )

        cached = asyncio.run(hydrator.fetch_text("1111111111111111111"))
        remote = asyncio.run(hydrator.fetch_text("2222222222222222222"))

        self.assertEqual(cached, "cached text")
        self.assertEqual(remote, "remote text")
        self.assertEqual(fallback.requested, ["2222222222222222222"])

    def test_load_hydrated_text_cache_ignores_auxiliary_files_and_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "sample.csv").write_text(
                "tweetId,texto\n"
                "1111111111111111111,CSV text\n",
                encoding="utf-8",
            )
            (cache_dir / "sample.tsv").write_text(
                "tweetId\ttexto\n"
                "2222222222222222222\tTSV text, with comma\n",
                encoding="utf-8",
            )
            (cache_dir / "sample.discarded.tsv").write_text(
                "tweetId\treason\tcleanedLength\n"
                "3333333333333333333\tmention\t20\n",
                encoding="utf-8",
            )
            excluded_output = cache_dir / "output.tsv"
            excluded_output.write_text(
                "tweetId\ttexto\n"
                "4444444444444444444\tExcluded text\n",
                encoding="utf-8",
            )

            paths = list(iter_hydrated_text_cache_paths(cache_dir, exclude_paths=(excluded_output,)))
            cache = load_hydrated_text_cache(paths)

        self.assertEqual(paths, [cache_dir / "sample.csv", cache_dir / "sample.tsv"])
        self.assertEqual(
            cache,
            {
                "1111111111111111111": "CSV text",
                "2222222222222222222": "TSV text, with comma",
            },
        )

    def test_iter_hydrated_text_cache_records_preserves_file_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.tsv"
            cache_path.write_text(
                "tweetId\ttexto\n"
                "1111111111111111111\tFirst text\n"
                "2222222222222222222\tSecond text\n",
                encoding="utf-8",
            )

            records = list(iter_hydrated_text_cache_records([cache_path]))

        self.assertEqual(
            records,
            [
                ("1111111111111111111", "First text"),
                ("2222222222222222222", "Second text"),
            ],
        )

    def test_build_quality_corpus_from_cache_writes_target_valid_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "cache.tsv"
            cache_path.write_text(
                "tweetId\ttexto\n"
                "1111111111111111111\tThis is a normal cached COVID text\n"
                "2222222222222222222\tThis is a normal cached COVID text\n"
                "3333333333333333333\tAnother normal cached COVID text\n",
                encoding="utf-8",
            )
            output_path = tmp_path / "quality.tsv"

            summary = build_quality_corpus_from_cache(
                cache_paths=[cache_path],
                output_path=output_path,
                target_valid=2,
                overwrite=True,
            )
            with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            audit_ok = json.loads(output_path.with_suffix(".audit.json").read_text(encoding="utf-8"))["ok"]

        self.assertEqual(summary.hydrated, 2)
        self.assertEqual(summary.discarded, 1)
        self.assertEqual(summary.output_rows, 2)
        self.assertEqual(rows[0]["tweetId"], "1111111111111111111")
        self.assertEqual(rows[1]["tweetId"], "3333333333333333333")
        self.assertTrue(audit_ok)

    def test_load_hydration_skip_ids_uses_auxiliary_files_except_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            (cache_dir / "old.not_hydrated.csv").write_text(
                "tweetId,reason\n"
                "1111111111111111111,not_found\n",
                encoding="utf-8",
            )
            (cache_dir / "old.discarded.tsv").write_text(
                "tweetId\treason\tcleanedLength\n"
                "2222222222222222222\tmention\t30\n",
                encoding="utf-8",
            )
            (cache_dir / "sample.tsv").write_text(
                "tweetId\ttexto\n"
                "3333333333333333333\tValid cached text\n",
                encoding="utf-8",
            )
            excluded_auxiliary = cache_dir / "current.not_hydrated.tsv"
            excluded_auxiliary.write_text(
                "tweetId\treason\n"
                "4444444444444444444\ttimeout\n",
                encoding="utf-8",
            )

            paths = list(iter_hydration_skip_cache_paths(cache_dir, exclude_paths=(excluded_auxiliary,)))
            skip_ids = load_hydration_skip_ids(paths)

        self.assertEqual(paths, [cache_dir / "old.discarded.tsv", cache_dir / "old.not_hydrated.csv"])
        self.assertEqual(skip_ids, {"1111111111111111111", "2222222222222222222"})

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

    def test_quality_filter_preserves_commas_for_tsv(self):
        result = filter_quality_tweet_text("Texas, Florida, and Arizona have fewer COVID restrictions")

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "Texas, Florida, and Arizona have fewer COVID restrictions")

    def test_quality_filter_discards_thread_markers(self):
        samples = [
            "Useful public health information for everyone 1/2",
            "Useful public health information for everyone 3 / 4",
            "Useful public health information for everyone 1 of 2",
            "Useful public health information part 2 for everyone",
            "Useful public health thread for everyone",
            "Useful public health information continued tomorrow",
            "Useful public health information contd tomorrow",
        ]

        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(filter_quality_tweet_text(sample).reason, "thread_marker")

    def test_quality_filter_discards_truncated_text(self):
        self.assertEqual(filter_quality_tweet_text("Useful public health update...").reason, "truncated")
        self.assertEqual(filter_quality_tweet_text("Useful public health update…").reason, "truncated")

    def test_quality_filter_discards_bad_control_and_delimiter_chars(self):
        samples = [
            "Useful public health\tupdate for everyone",
            "Useful public health\nupdate for everyone",
            "Useful public health \\ update for everyone",
            "Useful public health \x07 update for everyone",
        ]

        for sample in samples:
            with self.subTest(sample=repr(sample)):
                self.assertIsNotNone(filter_quality_tweet_text(sample).reason)

    def test_quality_filter_normalizes_spaces_without_leading_or_trailing_space(self):
        result = filter_quality_tweet_text("  Useful   public health update for everyone  ")

        self.assertIsNone(result.reason)
        self.assertEqual(result.text, "Useful public health update for everyone")

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

    def test_build_quality_corpus_writes_tsv_without_escapes_and_deduplicates_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "ids.csv").write_text(
                "tweet_id\n"
                "2222222222222222222\n"
                "2323232323232323232\n"
                "2424242424242424242\n",
                encoding="utf-8",
            )
            output_path = root / "hydrated" / "quality.tsv"
            not_hydrated_path = root / "hydrated" / "quality.not_hydrated.tsv"
            discarded_path = root / "hydrated" / "quality.discarded.tsv"
            report_path = root / "hydrated" / "quality.report.json"
            audit_path = root / "hydrated" / "quality.audit.json"
            duplicate_text = "Texas, Florida, and Arizona have fewer COVID restrictions"
            hydrator = FakeHydrator(
                {
                    "2222222222222222222": duplicate_text,
                    "2323232323232323232": duplicate_text,
                    "2424242424242424242": "Another normal public health update",
                }
            )

            summary = asyncio.run(
                build_quality_corpus(
                    input_path=input_dir,
                    output_path=output_path,
                    hydrator=hydrator,
                    not_hydrated_path=not_hydrated_path,
                    discarded_path=discarded_path,
                    report_path=report_path,
                    audit_path=audit_path,
                    target_valid=2,
                    max_attempts=3,
                )
            )

            raw_lines = output_path.read_text(encoding="utf-8-sig").splitlines()
            with output_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            with discarded_path.open(encoding="utf-8-sig", newline="") as handle:
                discarded = list(csv.DictReader(handle, delimiter="\t"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(summary.output_rows, 2)
        self.assertEqual(raw_lines[0], "tweetId\ttexto")
        self.assertEqual(raw_lines[1], f"2222222222222222222\t{duplicate_text}")
        self.assertNotIn("\\", "\n".join(raw_lines))
        self.assertNotIn('"', "\n".join(raw_lines))
        self.assertEqual(len(rows), 2)
        self.assertEqual(discarded[0]["reason"], "duplicate_text")
        self.assertTrue(audit["ok"])

    def test_quality_audit_reports_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tsv"
            path.write_text(
                "\ufefftweetId\ttexto\n"
                "2525252525252525252\tBad thread text 1/2\n"
                "2626262626262626262\tBad text with \\ backslash\n",
                encoding="utf-8",
            )

            audit = audit_quality_tsv(path, expected_count=2)

        self.assertFalse(audit.ok)
        self.assertEqual(audit.violations["thread_marker"], 1)
        self.assertEqual(audit.violations["backslash"], 1)


if __name__ == "__main__":
    unittest.main()
