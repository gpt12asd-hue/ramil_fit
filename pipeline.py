#!/usr/bin/env python3
"""YouTube subtitle extraction and Claude analysis pipeline."""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_subtitles(video_id: str) -> tuple[list[dict], str]:
    """Returns (transcript_entries, language_code)."""
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # prefer manual transcripts, then auto-generated
    for priority in ("manual", "generated"):
        for transcript in transcript_list:
            if priority == "manual" and not transcript.is_generated:
                return transcript.fetch(), transcript.language_code
            if priority == "generated" and transcript.is_generated:
                return transcript.fetch(), transcript.language_code

    raise NoTranscriptFound(video_id, [], [])


def format_transcript(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        start = entry.get("start", 0)
        text = entry.get("text", "").strip()
        if text:
            mm, ss = divmod(int(start), 60)
            hh, mm = divmod(mm, 60)
            ts = f"{hh:02d}:{mm:02d}:{ss:02d}" if hh else f"{mm:02d}:{ss:02d}"
            lines.append(f"[{ts}] {text}")
    return "\n".join(lines)


def analyze_with_claude(transcript_text: str, video_url: str) -> str:
    client = anthropic.Anthropic()

    prompt = f"""Analyze the following YouTube video transcript and provide:
1. A concise summary (3-5 sentences)
2. Key topics and themes (bullet list)
3. Main insights or takeaways (bullet list)
4. Sentiment and tone assessment
5. Target audience

Video URL: {video_url}

Transcript:
{transcript_text}"""

    print("Analyzing with Claude...", flush=True)

    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        chunks = []
        for text in stream.text_stream:
            print(text, end="", flush=True)
            chunks.append(text)
        print()  # newline after stream

    return "".join(chunks)


def save_results(output: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {output_path}")


def run(url: str, output_dir: str = "output") -> Path:
    video_id = extract_video_id(url)
    print(f"Video ID: {video_id}")

    print("Fetching subtitles...")
    try:
        entries, lang = fetch_subtitles(video_id)
    except TranscriptsDisabled:
        print("ERROR: Subtitles are disabled for this video.")
        sys.exit(1)
    except NoTranscriptFound:
        print("ERROR: No transcript found for this video.")
        sys.exit(1)

    print(f"Found transcript in language: {lang} ({len(entries)} segments)")
    transcript_text = format_transcript(entries)

    analysis = analyze_with_claude(transcript_text, url)

    result = {
        "url": url,
        "video_id": video_id,
        "language": lang,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transcript": transcript_text,
        "analysis": analysis,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{video_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(result, output_path)
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <youtube_url> [output_dir]")
        print("Example: python pipeline.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
        sys.exit(1)

    youtube_url = sys.argv[1]
    out_directory = sys.argv[2] if len(sys.argv) > 2 else "output"
    run(youtube_url, out_directory)
