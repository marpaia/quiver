"""A minimal command line: inspect the example circuit, or run the simulation objective on it."""

from __future__ import annotations

import argparse
import json

from quiver.generate import example_inducible


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quiver", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("example", help="print the example circuit as JSON")
    sub.add_parser("simulate", help="score the example circuit's inducible response (needs the sim extra)")
    args = parser.parse_args(argv)

    if args.command == "example":
        print(json.dumps(example_inducible().to_dict(), indent=2))
        return 0
    if args.command == "simulate":
        from quiver.objective import inducible_response, score_once

        score = score_once(example_inducible(), inducible_response("GFP"))
        print(f"inducible_response(GFP) = {score:.3f}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
