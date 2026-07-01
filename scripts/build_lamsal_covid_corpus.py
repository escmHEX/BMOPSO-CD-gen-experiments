from __future__ import annotations

import argparse
import asyncio
import csv
import io
import itertools
import json
import re
import sys
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol, TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "external" / "covid19_tweets" / "ieee_raw"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "external"
    / "covid19_tweets"
    / "hydrated"
    / "lamsal_covid_tweets_sample.csv"
)
DEFAULT_FILTERED_OUTPUT = (
    ROOT
    / "data"
    / "external"
    / "covid19_tweets"
    / "hydrated"
    / "lamsal_covid_tweets_filtered_sample.csv"
)
DEFAULT_ACCOUNTS_DB = ROOT / "data" / "external" / "covid19_tweets" / "twscrape" / "accounts.db"
DEFAULT_COOKIES_FILE = ROOT / "data" / "external" / "covid19_tweets" / "cookies.txt"
DEFAULT_ACCOUNT_NAME = "thesis_local"
DEFAULT_LIMIT = 100
DEFAULT_TWEET_TIMEOUT_SECONDS = 30.0
DEFAULT_TARGET_VALID = 100
MIN_FILTERED_TEXT_CHARS = 15
HASHTAG_HEAVY_RATIO = 0.5
OUTPUT_FIELDS = ("tweetId", "texto")
NOT_HYDRATED_FIELDS = ("tweetId", "reason")
DISCARDED_FIELDS = ("tweetId", "reason", "cleanedLength")
CSV_ESCAPE_CHAR = "\\"
KNOWN_ID_COLUMNS = ("tweet_id", "tweetid", "tweet id", "id", "status_id", "statusid", "status id")
TWEET_ID_PATTERN = re.compile(r"^\d{5,25}$")
WHITESPACE_PATTERN = re.compile(r"\s+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+", re.IGNORECASE)
HASHTAG_PATTERN = re.compile(r"(?<![\w#])#[^\s#@]+")
MENTION_PATTERN = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{1,20}\b")


class TweetTextHydrator(Protocol):
    async def fetch_text(self, tweet_id: str) -> str | None:
        ...


@dataclass(frozen=True)
class HydrationSummary:
    input_path: str
    output_path: str
    not_hydrated_path: str
    report_path: str | None
    total_discovered: int
    skipped_existing: int
    selected: int
    hydrated: int
    not_hydrated: int
    started_at: str
    completed_at: str
    discarded: int = 0
    target_valid: int | None = None
    max_attempts: int | None = None
    output_rows: int | None = None
    not_hydrated_rows: int | None = None
    discarded_rows: int | None = None


@dataclass(frozen=True)
class FilterResult:
    text: str
    reason: str | None
    cleaned_length: int


class TwscrapeTweetTextHydrator:
    def __init__(self, *, accounts_db: Path, account_name: str, cookies_file: Path) -> None:
        self.accounts_db = accounts_db
        self.account_name = account_name
        self.cookies_file = cookies_file
        self._api = None
        self._ready = False

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            from twscrape import API
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Missing optional dependency 'twscrape'. Install dataset tooling with "
                "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements.dataset.txt'."
            ) from error

        if not self.cookies_file.exists():
            raise FileNotFoundError(f"Cookies file not found: {self.cookies_file}")

        cookies = self.cookies_file.read_text(encoding="utf-8").strip()
        if not cookies:
            raise ValueError(f"Cookies file is empty: {self.cookies_file}")

        self.accounts_db.parent.mkdir(parents=True, exist_ok=True)
        self._api = API(str(self.accounts_db))
        await self._api.pool.add_account_cookies(self.account_name, cookies)
        self._ready = True

    async def fetch_text(self, tweet_id: str) -> str | None:
        await self._ensure_ready()
        tweet = await self._api.tweet_details(int(tweet_id))
        if tweet is None:
            return None
        text = getattr(tweet, "rawContent", None)
        if not text:
            return None
        return str(text)


def normalize_tweet_text(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def filter_tweet_text(text: str) -> FilterResult:
    repaired = normalize_tweet_text(_fix_text(text))
    without_urls = URL_PATTERN.sub("", repaired)
    compact_without_urls = normalize_tweet_text(without_urls)
    if _contains_emoji(compact_without_urls):
        return FilterResult(text="", reason="emoji", cleaned_length=len(compact_without_urls))

    if _is_hashtag_heavy(compact_without_urls):
        return FilterResult(text="", reason="hashtag_heavy", cleaned_length=len(compact_without_urls))

    without_hashtags = HASHTAG_PATTERN.sub("", without_urls)
    cleaned = _remove_csv_unsafe_text_chars(_strip_wrapping_double_quotes(normalize_tweet_text(without_hashtags)))
    if "@" in cleaned or MENTION_PATTERN.search(cleaned):
        return FilterResult(text="", reason="mention", cleaned_length=len(cleaned))
    if cleaned.count(":") > 1:
        return FilterResult(text="", reason="too_many_colons", cleaned_length=len(cleaned))
    if len(cleaned) < MIN_FILTERED_TEXT_CHARS:
        return FilterResult(text="", reason="too_short", cleaned_length=len(cleaned))
    return FilterResult(text=cleaned, reason=None, cleaned_length=len(cleaned))


def _fix_text(text: str) -> str:
    try:
        from ftfy import fix_text
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing optional dependency 'ftfy'. Install dataset tooling with "
            "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements.dataset.txt'."
        ) from error
    return unicodedata.normalize("NFC", fix_text(text))


def _contains_emoji(text: str) -> bool:
    try:
        from emoji import replace_emoji
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Missing optional dependency 'emoji'. Install dataset tooling with "
            "'.\\.venv\\Scripts\\python.exe -m pip install -r requirements.dataset.txt'."
        ) from error
    return replace_emoji(text, replace="") != text


def _strip_wrapping_double_quotes(text: str) -> str:
    stripped = text.strip()
    while len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        stripped = stripped[1:-1].strip()
    return normalize_tweet_text(stripped)


def _remove_csv_unsafe_text_chars(text: str) -> str:
    return normalize_tweet_text(text.translate(str.maketrans({"\"": " ", ",": " "})))


def _is_hashtag_heavy(text: str) -> bool:
    hashtags = HASHTAG_PATTERN.findall(text)
    if not hashtags:
        return False
    visible_chars = len(WHITESPACE_PATTERN.sub("", text))
    hashtag_chars = sum(len(hashtag) for hashtag in hashtags)
    return visible_chars > 0 and hashtag_chars / visible_chars >= HASHTAG_HEAVY_RATIO


def discover_tweet_ids(input_path: Path | str, *, id_column: str | None = None) -> list[str]:
    return list(iter_tweet_ids(input_path, id_column=id_column))


def iter_tweet_ids(input_path: Path | str, *, id_column: str | None = None) -> Iterable[str]:
    path = Path(input_path)
    seen: set[str] = set()

    for csv_source in _iter_csv_sources(path):
        with csv_source.open() as handle:
            yield from _iter_ids_from_csv(handle, source=csv_source.label, id_column=id_column, seen=seen)


async def build_corpus(
    *,
    input_path: Path | str,
    output_path: Path | str,
    hydrator: TweetTextHydrator,
    not_hydrated_path: Path | str | None = None,
    report_path: Path | str | None = None,
    id_column: str | None = None,
    limit: int | None = None,
    tweet_timeout_seconds: float = DEFAULT_TWEET_TIMEOUT_SECONDS,
) -> HydrationSummary:
    started_at = _utc_now()
    input_path = Path(input_path)
    output_path = Path(output_path)
    not_hydrated_path = Path(not_hydrated_path) if not_hydrated_path else output_path.with_suffix(".not_hydrated.csv")
    report_path = Path(report_path) if report_path else None

    existing_ids = _read_existing_output_ids(output_path)
    existing_ids.update(_read_existing_not_hydrated_ids(not_hydrated_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    not_hydrated_path.parent.mkdir(parents=True, exist_ok=True)

    total_discovered = 0
    skipped_existing = 0
    selected = 0
    hydrated = 0
    not_hydrated = 0

    with output_path.open("a", encoding="utf-8", newline="") as output_handle, not_hydrated_path.open(
        "a", encoding="utf-8", newline=""
    ) as miss_handle:
        output_writer = _make_csv_writer(output_handle, OUTPUT_FIELDS)
        miss_writer = _make_csv_writer(miss_handle, NOT_HYDRATED_FIELDS)
        _write_header_if_empty(output_path, output_handle, output_writer)
        _write_header_if_empty(not_hydrated_path, miss_handle, miss_writer)

        tweet_ids = iter_tweet_ids(input_path, id_column=id_column) if limit != 0 else ()
        for tweet_id in tweet_ids:
            total_discovered += 1
            if tweet_id in existing_ids:
                skipped_existing += 1
                continue

            selected += 1
            try:
                text = await asyncio.wait_for(hydrator.fetch_text(tweet_id), timeout=tweet_timeout_seconds)
            except TimeoutError:
                miss_writer.writerow({"tweetId": tweet_id, "reason": "timeout"})
                miss_handle.flush()
                not_hydrated += 1
                if limit is not None and selected >= limit:
                    break
                continue
            except Exception as error:
                miss_writer.writerow({"tweetId": tweet_id, "reason": type(error).__name__})
                miss_handle.flush()
                not_hydrated += 1
                if limit is not None and selected >= limit:
                    break
                continue

            text = _remove_csv_unsafe_text_chars(normalize_tweet_text(text or ""))
            if not text:
                miss_writer.writerow({"tweetId": tweet_id, "reason": "not_found"})
                miss_handle.flush()
                not_hydrated += 1
                if limit is not None and selected >= limit:
                    break
                continue

            output_writer.writerow({"tweetId": tweet_id, "texto": text})
            output_handle.flush()
            hydrated += 1
            if limit is not None and selected >= limit:
                break

    summary = HydrationSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        not_hydrated_path=str(not_hydrated_path),
        report_path=str(report_path) if report_path else None,
        total_discovered=total_discovered,
        skipped_existing=skipped_existing,
        selected=selected,
        hydrated=hydrated,
        not_hydrated=not_hydrated,
        started_at=started_at,
        completed_at=_utc_now(),
    )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


async def build_filtered_corpus(
    *,
    input_path: Path | str,
    output_path: Path | str,
    hydrator: TweetTextHydrator,
    not_hydrated_path: Path | str | None = None,
    discarded_path: Path | str | None = None,
    report_path: Path | str | None = None,
    id_column: str | None = None,
    target_valid: int = DEFAULT_TARGET_VALID,
    max_attempts: int | None = None,
    tweet_timeout_seconds: float = DEFAULT_TWEET_TIMEOUT_SECONDS,
    overwrite: bool = False,
) -> HydrationSummary:
    started_at = _utc_now()
    input_path = Path(input_path)
    output_path = Path(output_path)
    not_hydrated_path = Path(not_hydrated_path) if not_hydrated_path else output_path.with_suffix(".not_hydrated.csv")
    discarded_path = Path(discarded_path) if discarded_path else output_path.with_suffix(".discarded.csv")
    report_path = Path(report_path) if report_path else None
    max_attempts = max_attempts if max_attempts is not None else max(1000, target_valid * 10)

    if overwrite:
        artifacts = [output_path, not_hydrated_path, discarded_path]
        if report_path:
            artifacts.append(report_path)
        _remove_existing_artifacts(*artifacts)

    output_ids = _read_existing_output_ids(output_path)
    existing_ids = set(output_ids)
    existing_ids.update(_read_existing_not_hydrated_ids(not_hydrated_path))
    existing_ids.update(_read_existing_discarded_ids(discarded_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    not_hydrated_path.parent.mkdir(parents=True, exist_ok=True)
    discarded_path.parent.mkdir(parents=True, exist_ok=True)

    target_new_valid = max(0, target_valid - len(output_ids))
    total_discovered = 0
    skipped_existing = 0
    selected = 0
    hydrated = 0
    not_hydrated = 0
    discarded = 0

    with output_path.open("a", encoding="utf-8", newline="") as output_handle, not_hydrated_path.open(
        "a", encoding="utf-8", newline=""
    ) as miss_handle, discarded_path.open("a", encoding="utf-8", newline="") as discard_handle:
        output_writer = _make_csv_writer(output_handle, OUTPUT_FIELDS)
        miss_writer = _make_csv_writer(miss_handle, NOT_HYDRATED_FIELDS)
        discard_writer = _make_csv_writer(discard_handle, DISCARDED_FIELDS)
        _write_header_if_empty(output_path, output_handle, output_writer)
        _write_header_if_empty(not_hydrated_path, miss_handle, miss_writer)
        _write_header_if_empty(discarded_path, discard_handle, discard_writer)

        tweet_ids = iter_tweet_ids(input_path, id_column=id_column) if target_new_valid != 0 else ()
        for tweet_id in tweet_ids:
            total_discovered += 1
            if tweet_id in existing_ids:
                skipped_existing += 1
                continue
            if selected >= max_attempts:
                break

            selected += 1
            try:
                text = await asyncio.wait_for(hydrator.fetch_text(tweet_id), timeout=tweet_timeout_seconds)
            except TimeoutError:
                miss_writer.writerow({"tweetId": tweet_id, "reason": "timeout"})
                miss_handle.flush()
                not_hydrated += 1
                continue
            except Exception as error:
                miss_writer.writerow({"tweetId": tweet_id, "reason": type(error).__name__})
                miss_handle.flush()
                not_hydrated += 1
                continue

            if not text:
                miss_writer.writerow({"tweetId": tweet_id, "reason": "not_found"})
                miss_handle.flush()
                not_hydrated += 1
                continue

            filter_result = filter_tweet_text(text)
            if filter_result.reason is not None:
                discard_writer.writerow(
                    {
                        "tweetId": tweet_id,
                        "reason": filter_result.reason,
                        "cleanedLength": filter_result.cleaned_length,
                    }
                )
                discard_handle.flush()
                discarded += 1
                continue

            output_writer.writerow({"tweetId": tweet_id, "texto": filter_result.text})
            output_handle.flush()
            hydrated += 1
            if hydrated >= target_new_valid:
                break

    summary = HydrationSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        not_hydrated_path=str(not_hydrated_path),
        report_path=str(report_path) if report_path else None,
        total_discovered=total_discovered,
        skipped_existing=skipped_existing,
        selected=selected,
        hydrated=hydrated,
        not_hydrated=not_hydrated,
        started_at=started_at,
        completed_at=_utc_now(),
        discarded=discarded,
        target_valid=target_valid,
        max_attempts=max_attempts,
        output_rows=_count_csv_data_rows(output_path),
        not_hydrated_rows=_count_csv_data_rows(not_hydrated_path),
        discarded_rows=_count_csv_data_rows(discarded_path),
    )
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    return summary


@dataclass(frozen=True)
class _CsvSource:
    label: str
    opener: Callable[[], TextIO]

    def open(self) -> TextIO:
        return self.opener()


def _iter_csv_sources(input_path: Path) -> Iterable[_CsvSource]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*"), key=lambda item: str(item).lower())
    for file_path in files:
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            yield _CsvSource(
                label=str(file_path),
                opener=lambda file_path=file_path: file_path.open("r", encoding="utf-8-sig", newline=""),
            )
        elif suffix == ".zip":
            yield from _iter_zip_csv_sources(file_path)


def _iter_zip_csv_sources(zip_path: Path) -> Iterable[_CsvSource]:
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))

    for name in names:
        label = f"{zip_path}!{name}"

        def opener(zip_path=zip_path, name=name):
            archive = zipfile.ZipFile(zip_path)
            raw = archive.open(name)
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            original_close = text.close

            def close_with_archive() -> None:
                original_close()
                archive.close()

            text.close = close_with_archive
            return text

        yield _CsvSource(label=label, opener=opener)


def _iter_ids_from_csv(
    handle: TextIO,
    *,
    source: str,
    id_column: str | None,
    seen: set[str],
) -> Iterable[str]:
    reader = csv.reader(handle)
    first_row = next(reader, None)
    if first_row is None:
        return

    id_index = _resolve_id_column(first_row, id_column=id_column)
    rows: Iterable[list[str]]
    if id_index is None:
        first_value = _clean_tweet_id(first_row[0] if first_row else "")
        if first_value is None:
            raise ValueError(f"Could not detect tweet ID column in {source}: {first_row}")
        id_index = 0
        rows = itertools.chain([first_row], reader)
    else:
        rows = reader

    for row in rows:
        if id_index >= len(row):
            continue
        tweet_id = _clean_tweet_id(row[id_index])
        if tweet_id is None or tweet_id in seen:
            continue
        seen.add(tweet_id)
        yield tweet_id


def _resolve_id_column(header: list[str], *, id_column: str | None) -> int | None:
    normalized_header = [_normalize_column_name(value) for value in header]
    if id_column:
        wanted = _normalize_column_name(id_column)
        for index, column in enumerate(normalized_header):
            if column == wanted:
                return index
        raise ValueError(f"Column '{id_column}' not found. Available columns: {', '.join(header)}")

    known = {_normalize_column_name(value) for value in KNOWN_ID_COLUMNS}
    for index, column in enumerate(normalized_header):
        if column in known:
            return index
    return None


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _clean_tweet_id(value: str) -> str | None:
    tweet_id = value.strip().strip("'").strip('"')
    if not TWEET_ID_PATTERN.fullmatch(tweet_id):
        return None
    return tweet_id


def _read_existing_output_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            tweet_id
            for row in reader
            if (tweet_id := _clean_tweet_id(str(row.get("tweetId", "")))) is not None
        }


def _read_existing_not_hydrated_ids(not_hydrated_path: Path) -> set[str]:
    if not not_hydrated_path.exists() or not_hydrated_path.stat().st_size == 0:
        return set()
    with not_hydrated_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            tweet_id
            for row in reader
            if (tweet_id := _clean_tweet_id(str(row.get("tweetId", "")))) is not None
        }


def _read_existing_discarded_ids(discarded_path: Path) -> set[str]:
    if not discarded_path.exists() or discarded_path.stat().st_size == 0:
        return set()
    with discarded_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            tweet_id
            for row in reader
            if (tweet_id := _clean_tweet_id(str(row.get("tweetId", "")))) is not None
        }


def _write_header_if_empty(path: Path, handle: TextIO, writer: csv.DictWriter) -> None:
    if not path.exists() or path.stat().st_size == 0:
        handle.write("\ufeff")
        writer.writeheader()
        handle.flush()


def _make_csv_writer(handle: TextIO, fieldnames: Iterable[str]) -> csv.DictWriter:
    return csv.DictWriter(
        handle,
        fieldnames=fieldnames,
        escapechar=CSV_ESCAPE_CHAR,
        lineterminator="\n",
        quotechar=None,
        quoting=csv.QUOTE_NONE,
    )


def _remove_existing_artifacts(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _count_csv_data_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(".report.json")


def _resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    return DEFAULT_FILTERED_OUTPUT if args.filtered else DEFAULT_OUTPUT


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a tweetId,texto CSV from IEEE/DataPort Lamsal COVID-19 tweet ID files."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV/ZIP file or directory containing IEEE raw data.")
    parser.add_argument("--output", default=None, help="Hydrated CSV output path.")
    parser.add_argument("--not-hydrated", default=None, help="CSV path for IDs that could not be hydrated.")
    parser.add_argument("--report", default=None, help="JSON report path. Defaults next to --output.")
    parser.add_argument("--id-column", default=None, help="Override the tweet ID column name.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum new tweet IDs to attempt.")
    parser.add_argument("--filtered", action="store_true", help="Write only tweets that pass text cleanup filters.")
    parser.add_argument("--target-valid", type=int, default=DEFAULT_TARGET_VALID, help="Target valid rows for filtered mode.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum new IDs to attempt in filtered mode.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing filtered output, not_hydrated, discarded, and report files before running.",
    )
    parser.add_argument(
        "--cookies-file",
        default=str(DEFAULT_COOKIES_FILE),
        help="Local file containing auth_token and ct0 cookies.",
    )
    parser.add_argument("--account-name", default=DEFAULT_ACCOUNT_NAME, help="Local twscrape account name.")
    parser.add_argument("--accounts-db", default=str(DEFAULT_ACCOUNTS_DB), help="Local twscrape SQLite account DB.")
    parser.add_argument(
        "--tweet-timeout-seconds",
        type=float,
        default=DEFAULT_TWEET_TIMEOUT_SECONDS,
        help="Maximum seconds to wait for each tweet hydration request.",
    )
    return parser.parse_args(argv)


async def _run_from_args(args: argparse.Namespace) -> HydrationSummary:
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be zero or greater")
    if args.target_valid < 0:
        raise ValueError("--target-valid must be zero or greater")
    if args.max_attempts is not None and args.max_attempts < 0:
        raise ValueError("--max-attempts must be zero or greater")
    if args.overwrite and not args.filtered:
        raise ValueError("--overwrite is only supported with --filtered")

    output_path = _resolve_output_path(args)
    report_path = Path(args.report) if args.report else _default_report_path(output_path)
    hydrator = TwscrapeTweetTextHydrator(
        accounts_db=Path(args.accounts_db),
        account_name=args.account_name,
        cookies_file=Path(args.cookies_file),
    )
    if args.filtered:
        return await build_filtered_corpus(
            input_path=Path(args.input),
            output_path=output_path,
            hydrator=hydrator,
            not_hydrated_path=Path(args.not_hydrated)
            if args.not_hydrated
            else output_path.with_suffix(".not_hydrated.csv"),
            discarded_path=output_path.with_suffix(".discarded.csv"),
            report_path=report_path,
            id_column=args.id_column,
            target_valid=args.target_valid,
            max_attempts=args.max_attempts,
            tweet_timeout_seconds=args.tweet_timeout_seconds,
            overwrite=args.overwrite,
        )
    return await build_corpus(
        input_path=Path(args.input),
        output_path=output_path,
        hydrator=hydrator,
        not_hydrated_path=Path(args.not_hydrated) if args.not_hydrated else output_path.with_suffix(".not_hydrated.csv"),
        report_path=report_path,
        id_column=args.id_column,
        limit=args.limit,
        tweet_timeout_seconds=args.tweet_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = asyncio.run(_run_from_args(args))
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Discovered IDs: {summary.total_discovered}")
    print(f"Skipped existing: {summary.skipped_existing}")
    print(f"Attempted: {summary.selected}")
    print(f"Hydrated: {summary.hydrated}")
    print(f"Not hydrated: {summary.not_hydrated}")
    if summary.discarded:
        print(f"Discarded: {summary.discarded}")
    print(f"Output: {summary.output_path}")
    print(f"Report: {summary.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
