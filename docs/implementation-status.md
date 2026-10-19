# Implementation status

Updated 2026-10-19. See [implementation results](evaluation_upgrade/2026-10-19/IMPLEMENTATION_RESULTS.md) for tests, usage and limits.

| Capability | Design defined | Code implemented | Runtime connected | Tests | Real samples verified |
|---|---|---|---|---|---|
| Frozen pairing and input hashes | Yes | Yes | API/workflow/worker | Regression | Pending this change |
| ICC/EXIF/SDR rendering | Yes | Core paths | Pipeline/VLM/reports | P3/EXIF/precision fixtures | Pending |
| All-face/native ROI | Yes | Haar/manual correction | Pipeline and reports | Pixel/annotation/HTTP tests | Diverse faces pending |
| Structured proposition review | Yes | Yes | Adjudication | Direction/region/reference | Human calibration pending |
| Family counterexamples | Yes | Two narrow output tendencies | Strategy | Full denominator/counterexamples | Family annotation pending |
| External binding | Yes | Yes | Post-freeze retrieval | Device/version/source isolation | Excerpts remain background |
| Professional report | Yes | Core HTML/DOCX | ReportPayload 0.2 | Export tests and 4-page render | Camera0717 pending |
| Strict core v2 inference | Yes | Models/storage exist | Not fully migrated | Existing storage tests retained | No |
| Skin/lighting/group strategies | Yes | Partial | Collection only | No validity claim | No |
| Perception calibration | Yes | Protocol only | No | Not run | No |
| Editing feedback loop | Yes | Not implemented this change | No | Not run | No |

The system produces an auditable model-assisted report. Rule grades and document finalization are not human validation. Model services, optional search, fonts and datasets remain deployment inputs. Current-run verification is distinct from historical capability claims.
