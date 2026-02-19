#!/usr/bin/env python3
"""
Export OpenAPI specification from CDR FastAPI application.

Generates docs/openapi.json (and optionally openapi.yaml) from the
FastAPI app's live schema — no server required.

Usage:
    python scripts/export_openapi.py                    # → docs/openapi.json
    python scripts/export_openapi.py --format yaml      # → docs/openapi.yaml
    python scripts/export_openapi.py --output spec.json  # custom path
"""

import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cdr.api import create_app


def export_openapi(
    output_path: str | None = None,
    fmt: str = "json",
) -> dict:
    """Export the OpenAPI schema from the CDR app.

    Args:
        output_path: File path to write (default: docs/openapi.{fmt})
        fmt: Output format — 'json' or 'yaml'

    Returns:
        The OpenAPI schema dict
    """
    app = create_app()
    schema = app.openapi()

    if output_path is None:
        docs_dir = os.path.join(os.path.dirname(__file__), "..", "docs")
        os.makedirs(docs_dir, exist_ok=True)
        output_path = os.path.join(docs_dir, f"openapi.{fmt}")

    if fmt == "yaml":
        try:
            import yaml  # type: ignore[import-untyped]

            with open(output_path, "w") as f:
                yaml.dump(schema, f, default_flow_style=False, sort_keys=False)
        except ImportError:
            print("⚠️  PyYAML not installed, falling back to JSON")
            output_path = output_path.replace(".yaml", ".json")
            with open(output_path, "w") as f:
                json.dump(schema, f, indent=2)
    else:
        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)

    print(f"✅ OpenAPI spec exported to: {output_path}")
    print(f"   Title: {schema.get('info', {}).get('title')}")
    print(f"   Version: {schema.get('info', {}).get('version')}")

    paths = schema.get("paths", {})
    print(f"   Endpoints: {len(paths)}")
    for path, methods in sorted(paths.items()):
        for method in methods:
            if method.upper() in ("GET", "POST", "PUT", "DELETE", "PATCH"):
                print(f"     {method.upper():6s} {path}")

    return schema


def main():
    parser = argparse.ArgumentParser(description="Export CDR OpenAPI spec")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: docs/openapi.json)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )
    args = parser.parse_args()

    schema = export_openapi(output_path=args.output, fmt=args.format)

    # Validation summary
    paths = schema.get("paths", {})
    endpoints = sum(
        1
        for methods in paths.values()
        for m in methods
        if m.upper() in ("GET", "POST", "PUT", "DELETE", "PATCH")
    )
    schemas_count = len(schema.get("components", {}).get("schemas", {}))
    print(f"\n📊 Summary: {endpoints} endpoints, {schemas_count} schemas")


if __name__ == "__main__":
    main()
