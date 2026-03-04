"""
Compare paper selection results between OpenAI and a local LLM.

Reads two output JSON files produced by main.py and prints a side-by-side diff.

Usage:
    python compare_llms.py
    python compare_llms.py --openai out/output.json --local out/output_local.json
    python compare_llms.py --save out/comparison.json
"""

import argparse
import json


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_comparison(openai_results: dict, local_results: dict) -> None:
    all_ids = set(openai_results) | set(local_results)
    both = set(openai_results) & set(local_results)
    only_openai = set(openai_results) - set(local_results)
    only_local = set(local_results) - set(openai_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  OpenAI selected   : {len(openai_results)} papers")
    print(f"  Local LLM selected: {len(local_results)} papers")
    print(f"  Selected by both  : {len(both)}")
    print(f"  Only OpenAI       : {len(only_openai)}")
    print(f"  Only Local LLM    : {len(only_local)}")

    def tag(arxiv_id):
        if arxiv_id in both:
            return "BOTH"
        if arxiv_id in only_openai:
            return " OAI"
        return " LOC"

    print("\n" + "=" * 70)
    print("PER-PAPER BREAKDOWN")
    print("=" * 70)

    for arxiv_id in sorted(all_ids):
        entry = openai_results.get(arxiv_id) or local_results[arxiv_id]
        title = entry["title"][:72]
        print(f"\n[{tag(arxiv_id)}] {arxiv_id}  {title}")

        if arxiv_id in openai_results:
            r = openai_results[arxiv_id]
            print(f"  OpenAI  R={r.get('RELEVANCE', '?')}  {r.get('COMMENT', '')[:120]}")

        if arxiv_id in local_results:
            r = local_results[arxiv_id]
            print(f"  Local   R={r.get('RELEVANCE', '?')}  {r.get('COMMENT', '')[:120]}")

        if arxiv_id in both:
            o_rel = openai_results[arxiv_id].get("RELEVANCE", 0)
            l_rel = local_results[arxiv_id].get("RELEVANCE", 0)
            if o_rel != l_rel:
                print(f"  ↳ Relevance diff: OpenAI={o_rel}  Local={l_rel}")


def save_comparison(openai_results: dict, local_results: dict, path: str) -> None:
    all_ids = sorted(set(openai_results) | set(local_results))
    out = [
        {
            "arxiv_id": arxiv_id,
            "title": (openai_results.get(arxiv_id) or local_results[arxiv_id])["title"],
            "openai": openai_results.get(arxiv_id),
            "local": local_results.get(arxiv_id),
        }
        for arxiv_id in all_ids
    ]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nComparison saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare OpenAI vs local LLM paper selection")
    parser.add_argument("--openai", default="out/output.json",
                        help="OpenAI results JSON (default: out/output.json)")
    parser.add_argument("--local", default="out/output_local.json",
                        help="Local LLM results JSON (default: out/output_local.json)")
    parser.add_argument("--save", default="out/comparison.json",
                        help="Where to save comparison JSON (default: out/comparison.json)")
    args = parser.parse_args()

    openai_results = load_results(args.openai)
    local_results = load_results(args.local)

    print(f"Loaded {len(openai_results)} OpenAI results from {args.openai}")
    print(f"Loaded {len(local_results)} local LLM results from {args.local}")

    print_comparison(openai_results, local_results)
    save_comparison(openai_results, local_results, args.save)
