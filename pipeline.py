#!/usr/bin/env python3
"""YouTube subtitle extraction and Claude analysis pipeline.

Subtitle backends:
  --backend=api   (default) youtube-transcript-api Python library
  --backend=mcp   kimtaeyoon83/mcp-server-youtube-transcript via subprocess
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

MCP_SERVER_PATH = Path(__file__).parent / "mcp-server-youtube-transcript" / "dist" / "index.js"


# ── helpers ────────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    # bare 11-char ID passed directly
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not extract video ID from: {url}")


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


# ── backend: youtube-transcript-api ───────────────────────────────────────────

def fetch_via_api(video_id: str) -> tuple[str, str]:
    """Returns (formatted_transcript, language_code)."""
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    for priority in ("manual", "generated"):
        for t in transcript_list:
            if priority == "manual" and not t.is_generated:
                return format_transcript(t.fetch()), t.language_code
            if priority == "generated" and t.is_generated:
                return format_transcript(t.fetch()), t.language_code
    raise NoTranscriptFound(video_id, [], [])


# ── backend: MCP server ────────────────────────────────────────────────────────

def _mcp_rpc(request: dict) -> dict:
    """Send one JSON-RPC request to the MCP server over stdio, return response."""
    proc = subprocess.run(
        ["node", str(MCP_SERVER_PATH)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "result" in data or "error" in data:
                return data
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"MCP server returned no parseable response.\nstdout: {proc.stdout}\nstderr: {proc.stderr}")


def fetch_via_mcp(url: str, lang: str = "en") -> tuple[str, str]:
    """Returns (formatted_transcript, language_code)."""
    if not MCP_SERVER_PATH.exists():
        raise FileNotFoundError(
            f"MCP server not built at {MCP_SERVER_PATH}. "
            "Run: cd mcp-server-youtube-transcript && npm install && npm run build"
        )

    # MCP initialize handshake
    _mcp_rpc({
        "jsonrpc": "2.0", "id": 0, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pipeline", "version": "1.0"},
        },
    })

    # call get_transcript tool
    response = _mcp_rpc({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {
            "name": "get_transcript",
            "arguments": {"url": url, "lang": lang, "include_timestamps": True, "strip_ads": True},
        },
    })

    if "error" in response:
        raise RuntimeError(f"MCP error: {response['error']}")

    content_blocks = response.get("result", {}).get("content", [])
    text = "\n".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    return text, lang


# ── Claude analysis ────────────────────────────────────────────────────────────

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
        print()

    return "".join(chunks)


# ── save ───────────────────────────────────────────────────────────────────────

def save_results(output: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {output_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def run(url: str, output_dir: str = "output", backend: str = "api", lang: str = "en") -> Path:
    video_id = extract_video_id(url)
    print(f"Video ID: {video_id}  |  backend: {backend}")

    print("Fetching subtitles...")
    try:
        if backend == "mcp":
            transcript_text, lang_used = fetch_via_mcp(url, lang)
        else:
            transcript_text, lang_used = fetch_via_api(video_id)
    except TranscriptsDisabled:
        print("ERROR: Subtitles are disabled for this video.")
        sys.exit(1)
    except NoTranscriptFound:
        print("ERROR: No transcript found for this video.")
        sys.exit(1)

    print(f"Language: {lang_used}")
    analysis = analyze_with_claude(transcript_text, url)

    result = {
        "url": url,
        "video_id": video_id,
        "language": lang_used,
        "backend": backend,
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
    import argparse

    parser = argparse.ArgumentParser(description="YouTube → subtitles → Claude analysis")
    parser.add_argument("url", help="YouTube URL or video ID")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument(
        "--backend",
        choices=["api", "mcp"],
        default="api",
        help="Subtitle backend: api (youtube-transcript-api) or mcp (kimtaeyoon83 MCP server)",
    )
    parser.add_argument("--lang", default="en", help="Language code for MCP backend (default: en)")
    args = parser.parse_args()

    run(args.url, args.output_dir, args.backend, args.lang)
