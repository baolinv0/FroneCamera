from portrait_eval.corroboration import HeuristicCorroborationAdapter
from portrait_eval.models import ExternalEvidence


def test_heuristic_corroboration_rejects_non_front_camera_result() -> None:
    decision = HeuristicCorroborationAdapter().classify(
        claim="This device tends to show a lower near-white clipping ratio.",
        device_name="Phone X",
        evidence=ExternalEvidence(
            title="Phone X rear camera HDR review",
            url="https://example.test/rear-hdr",
            source_domain="example.test",
            snippet="The main rear camera preserves highlights.",
        ),
    )
    assert decision.verdict == "incomparable"
    assert decision.supports_claim is None


def test_heuristic_corroboration_marks_front_camera_evidence_for_human_review() -> None:
    decision = HeuristicCorroborationAdapter().classify(
        claim="This device tends to show a lower near-white clipping ratio.",
        device_name="Phone X",
        evidence=ExternalEvidence(
            title="Phone X selfie HDR review",
            url="https://example.test/selfie-hdr",
            source_domain="example.test",
            snippet="The front camera selfie keeps highlight detail in backlit scenes.",
        ),
    )
    assert decision.verdict == "unresolved"
    assert decision.front_camera_specific is True
