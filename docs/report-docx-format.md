# Reference-derived DOCX report format

The report bundle includes a real `.docx` file beside its HTML, JSON and CSV outputs. The DOCX renderer is derived from the retained flagship-phone assessment reference and intentionally keeps its visual language:

- A4 portrait with compact 0.53-inch side margins;
- Noto Sans CJK SC typography;
- dark `#202124`, orange `#F97316`, white and light-gray cards;
- numbered section bands, two-column evidence cards and explicit callouts;
- sample-limited score/evidence graphics;
- scene-by-scene sample montages and observation tables;
- mechanism attribution, failure modes, final verdict and appendix sections;
- a real footer `PAGE` field.

The renderer never invents a quality score. If structured `device_scores` are absent, the scorecard becomes an evidence-coverage view and is labeled as not being a quality ranking. Missing visual assets or mechanism records are also stated explicitly.

Source images are copied into the report directory and represented as relative paths in the JSON payload. This preserves finalization reproducibility without publishing absolute workstation paths.

## Rebuilding the packaged template

The checked-in template contains only styles, theme, page geometry, numbering and footer semantics. It contains no source-report prose or source images. To rebuild it from the retained reference:

```powershell
python scripts/build_report_template.py reference.docx src/portrait_eval/assets/report_reference_template.docx
```

Runtime generation is provided by `portrait_eval.docx_reporting.render_docx_report`. `render_report_bundle` calls it automatically, and authenticated or signed read-only clients can download the sibling DOCX endpoint.
