from portrait_eval.research import grade_source


def test_grade_source_distinguishes_official_and_professional_domains() -> None:
    official = grade_source("apple.com", "iPhone technical specifications", "front camera")
    professional = grade_source("dpreview.com", "Phone selfie review", "front camera")
    community = grade_source("random-blog.example", "Phone impressions", "selfie")
    assert official["source_tier"] == 1
    assert professional["source_tier"] == 3
    assert community["source_tier"] == 4
    assert official["front_camera_specific"] is True
