"""Command-line entry points for the runnable analytical demo."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Manto time-series analytical workbench")
    commands = parser.add_subparsers(dest="command")
    ui = commands.add_parser("ui", help="Launch the local chat and model-comparison interface")
    ui.add_argument("--port", type=int, default=8501)
    demo = commands.add_parser("demo", help="Run a reproducible synthetic sales experiment")
    demo.add_argument("--size", type=int, default=3)
    demo.add_argument("--pin", action="append", default=[])
    demo.add_argument("--lag", action="append", type=int)
    demo.add_argument("--output", default=os.getenv("MANTO_DATA_DIR", ".manto"))
    spec = commands.add_parser(
        "spec", help="Print a saved conversation specification without running analysis"
    )
    spec.add_argument("--conversation", required=True)
    spec.add_argument("--directory", default=os.getenv("MANTO_DATA_DIR", ".manto"))
    spec.add_argument("--json", action="store_true", dest="as_json")
    fetch = commands.add_parser("fetch-nbp", help="Fetch official monthly FX observations")
    fetch.add_argument("--code", default="EUR")
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", default=".manto/imports")
    args = parser.parse_args()
    if args.command == "spec":
        import sqlite3

        from manto.agent_tools import render_specification

        path = (Path(args.directory) / "dialogue.sqlite").resolve()
        try:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
                row = conn.execute(
                    "SELECT report_json FROM dialogue_reports WHERE thread_id=?",
                    (args.conversation,),
                ).fetchone()
            if not row:
                parser.error("No current report. Ask the agent to show a specification first.")
            report = json.loads(row[0])
            print(
                json.dumps(report, ensure_ascii=False, indent=2)
                if args.as_json
                else render_specification(report)
            )
        except (sqlite3.Error, OSError, ValueError):
            parser.error("The saved specification could not be read.")
        return
    if args.command == "ui":
        app = Path(__file__).with_name("ui.py")
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(app),
                    "--server.address",
                    "127.0.0.1",
                    "--server.port",
                    str(args.port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ]
            )
        )
    if args.command == "demo":
        from manto.analysis import run_analysis
        from manto.data import demo_dataset
        from manto.domain import AnalysisRequest
        from manto.storage import ResultStore

        dataset = demo_dataset()
        request = AnalysisRequest(
            target_id="sales",
            candidate_ids=[item.id for item in dataset.catalog if item.id != "sales"],
            model_size=args.size,
            pinned_ids=args.pin,
            lag_menu=args.lag or [0],
        )
        result = run_analysis(dataset, request)
        path = ResultStore(Path(args.output) / "experiments").save(result, dataset)
        print(
            json.dumps(
                {
                    "status": result.status,
                    "models": len(result.models),
                    "pareto": sum(model.is_pareto for model in result.models),
                    "recommended": result.recommended_id,
                    "champion": result.champion_id,
                    "next_forecast": result.next_forecast,
                    "saved_to": path,
                    "explanation": result.explanation,
                },
                indent=2,
            )
        )
        return
    if args.command == "fetch-nbp":
        from manto.data import fetch_nbp_monthly, snapshot_dataset

        dataset = fetch_nbp_monthly(args.code, args.start, args.end)
        print(snapshot_dataset(dataset, args.output))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
