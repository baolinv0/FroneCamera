from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT


def build_template(source: Path, output: Path) -> Path:
    """Create a compact, content-free template derived from the retained reference."""
    document = Document(source)
    body = document._element.body
    for child in list(body):
        if child.tag != "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sectPr":
            body.remove(child)

    for relationship_id, relationship in list(document.part.rels.items()):
        if relationship.reltype in {RT.IMAGE, RT.CUSTOM_XML}:
            document.part.drop_rel(relationship_id)
    for relationship_id, relationship in list(document.part.package.rels.items()):
        if relationship.reltype.endswith("/metadata/thumbnail"):
            del document.part.package.rels[relationship_id]

    properties = document.core_properties
    properties.title = "Front-camera evidence report template"
    properties.subject = "Reference-derived DOCX layout skeleton"
    properties.author = "FroneCamera"
    properties.last_modified_by = "FroneCamera"
    properties.comments = "Contains layout styles only; no source report content or images."

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_template(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
