import pytest

from capsleep.annotations import cap_features, parse_annotation_text


TEXT = """RemLogic Event Export
Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation
W\tUnknown\t22:06:03\tSLEEP-S0\t30\tROC-LOC
S1\tUnknown\t22:06:33\tSLEEP-S1\t30\tROC-LOC
S1\tUnknown\t22:07:03\tSLEEP-S1\t30\tROC-LOC
S1\tUnknown\t22:07:33\tSLEEP-S1\t30\tROC-LOC
S1\tUnknown\t22:08:03\tSLEEP-S1\t30\tROC-LOC
S1\tUnknown\t22:08:33\tSLEEP-S1\t30\tROC-LOC
S1\tUnknown\t22:09:03\tSLEEP-S1\t30\tROC-LOC
S1\tUnknown\t22:07:13\tMCAP-A1\t5\tEEG-F2-F4
S1\tUnknown\t22:07:38\tMCAP-A2\t10\tEEG-F2-F4
S1\tUnknown\t22:08:13\tMCAP-A3\t3\tEEG-F2-F4
S1\tUnknown\t22:09:23\tMCAP-A1\t3\tEEG-F2-F4
R\tUnknown\t22:09:33\tSLEEP-REM\t30\tROC-LOC
"""


def test_annotation_alignment_and_cap_sequence_rate():
    events = parse_annotation_text(TEXT, "22.06.03", 240)
    result = cap_features(events, 240)
    assert result["nrem_seconds"] == 180
    assert result["cap_sequence_count"] == 1
    assert result["cap_a_events_in_sequences"] == 3
    assert result["cap_a_orphan_count"] == 1
    assert result["cap_sequence_time_seconds"] == 60
    assert result["cap_a_seconds_in_sequences"] == 18
    assert result["cap_rate_percent"] == pytest.approx(100 * 60 / 180)
    assert result["stage_annotation_coverage_percent"] == 100


def test_annotation_outside_edf_duration_is_rejected():
    with pytest.raises(ValueError, match="exceeds recording duration"):
        parse_annotation_text(TEXT, "22.06.03", 60)


def test_insufficient_stage_coverage_reports_its_measured_fraction():
    events = parse_annotation_text(TEXT, "22.06.03", 360)
    with pytest.raises(ValueError, match="66.7%.*80% is required"):
        cap_features(events, 360)


def test_five_column_annotation_format_and_pre_edf_clock_are_supported():
    lines = []
    for line in TEXT.splitlines():
        if line.startswith("Sleep Stage\t"):
            lines.append(line.replace("Position\t", ""))
        elif line.count("\t") == 5:
            fields = line.split("\t")
            lines.append("\t".join(fields[:1] + fields[2:]))
        else:
            lines.append(line)
    text = "\n".join(lines)
    events = parse_annotation_text(text, "22.56.03", 240)
    assert min(event.start_seconds for event in events) == 0
    assert events[0].annotation_start_offset_seconds == -3000
    result = cap_features(events, 240)
    assert result["cap_sequence_count"] == 1


def test_annotation_clock_rollover_preserves_relative_event_timing():
    text = TEXT.replace("22:06:03", "23:59:03")
    text = text.replace("22:06:33", "23:59:33")
    text = text.replace("22:07:", "00:00:").replace("22:08:", "00:01:")
    text = text.replace("22:09:", "00:02:")
    events = parse_annotation_text(text, "23.59.03", 240)
    result = cap_features(events, 240)
    assert result["cap_sequence_count"] == 1
    assert result["cap_sequence_time_seconds"] == 60


def test_cap_stage_label_mismatch_is_reported_and_hypnogram_controls_nrem():
    mismatched = TEXT.replace("S1\tUnknown\t22:07:13\tMCAP-A1", "W\tUnknown\t22:07:13\tMCAP-A1")
    events = parse_annotation_text(mismatched, "22.06.03", 240)
    result = cap_features(events, 240)
    assert result["cap_sequence_count"] == 1
    assert result["cap_a_stage_label_mismatch_count"] == 1


def test_reference_sequence_parity_does_not_apply_strict_nrem_filter():
    outside_nrem = TEXT.replace(
        "S1\tUnknown\t22:07:03\tSLEEP-S1",
        "W\tUnknown\t22:07:03\tSLEEP-S0",
    )
    events = parse_annotation_text(outside_nrem, "22.06.03", 240)
    result = cap_features(events, 240)
    assert result["cap_sequence_count"] == 0
    assert result["cap_reference_sequence_count"] == 1
    assert result["cap_sequence_excluded_nrem_count"] == 1


def test_cap_rate_rejects_invalid_cap_phase_duration():
    invalid = TEXT.replace("MCAP-A1\t5", "MCAP-A1\t61")
    events = parse_annotation_text(invalid, "22.06.03", 240)
    with pytest.raises(ValueError, match="CAP-A duration"):
        cap_features(events, 240)


def test_cap_rate_reports_short_b_intervals_and_reference_parity():
    short_b = TEXT.replace("22:07:38\tMCAP-A2", "22:07:19\tMCAP-A2")
    events = parse_annotation_text(short_b, "22.06.03", 240)
    result = cap_features(events, 240)
    assert result["cap_short_b_interval_count"] == 1
    assert result["cap_sequence_count"] == 0
    assert result["cap_reference_sequence_count"] == 1


def test_overlapping_cap_events_are_rejected():
    overlap = TEXT.replace("22:07:38\tMCAP-A2", "22:07:12\tMCAP-A2")
    events = parse_annotation_text(overlap, "22.06.03", 240)
    with pytest.raises(ValueError, match="overlap"):
        cap_features(events, 240)
