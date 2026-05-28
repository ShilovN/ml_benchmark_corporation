#!/usr/bin/env python3
"""Small CLI for trying agent commands manually."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .executor import AgentContext, CommandExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one LLM command in the local workspace.")
    parser.add_argument("command", help='Command text, e.g. read_file("checker/README.md")')
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", default="salary_prediction")
    parser.add_argument(
        "--model-response",
        action="store_true",
        help="Extract a command from a free-form model response before executing",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    executor = CommandExecutor(AgentContext(workspace=args.workspace, task_id=args.task_id))
    if args.model_response:
        result = executor.execute_model_response(args.command)
    else:
        result = executor.execute_text(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
