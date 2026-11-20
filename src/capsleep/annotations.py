from __future__ import annotations

from dataclasses import dataclass
from datetime import time
import re

import numpy as np


@dataclass(frozen=True)
class Event:
    stage: str
    event: str
    start_seconds: float
    duration_seconds: float
    location: str
    annotation_start_offset_seconds: float = 0.0

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds


def _seconds_since_midnight(value: str) -> int:
    parsed = time.fromisoformat(value)
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second


def _recording_start_seconds(value: str) -> int:
    normalized = value.replace(".", ":")
    return _seconds_since_midnight(normalized)


def parse_annotation_text(
    text: str, recording_start: str, recording_duration: float, tolerance_seconds: float = 30.0
) -> list[Event]:
    if not np.isfinite(recording_duration) or recording_duration <= 0:
        raise ValueError("EDF recording duration must be finite and positive")
    start_of_recording = _recording_start_seconds(recording_start)
    columns: dict[str, int] | None = None
    column_count = 0
    annotation_reference_clock: int | None = None
    annotation_start_offset = 0.0
    parsed_events: list[Event] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("Sleep Stage\t"):
            names = [name.strip().lower() for name in line.split("\t")]
            required = {
                "stage": "sleep stage",
                "clock": "time",
                "event": "event",
                "duration": "duration",
                "location": "location",
            }
            columns = {}
            for key, prefix in required.items():
                match = next(
                    (index for index, name in enumerate(names) if name.startswith(prefix)),
                    None,
                )
                if match is None:
                    raise ValueError(f"Malformed annotation header at line {line_number}")
                columns[key] = match
            column_count = len(names)
            continue
        if columns is None or not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != column_count:
            raise ValueError(f"Malformed annotation line {line_number}: unexpected fields")
        try:
            clock = fields[columns["clock"]]
            clock_seconds = _seconds_since_midnight(clock)
            if annotation_reference_clock is None:
                annotation_reference_clock = clock_seconds
                annotation_start_offset = (
                    clock_seconds - start_of_recording + 43200
                ) % 86400 - 43200
            start = (clock_seconds - annotation_reference_clock) % 86400
            duration = float(fields[columns["duration"]])
        except ValueError as error:
            raise ValueError(f"Invalid annotation line {line_number}: {line}") from error
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid event duration at annotation line {line_number}")
        parsed_events.append(
            Event(
                fields[columns["stage"]].strip(),
                fields[columns["event"]].strip(),
                float(start),
                duration,
                fields[columns["location"]].strip(),
                annotation_start_offset,
            )
        )
    if columns is None or not parsed_events:
        raise ValueError("No parseable clinician scoring events found")
    earliest_start = min(event.start_seconds for event in parsed_events)
    latest_end = max(event.end_seconds for event in parsed_events)
    if latest_end - earliest_start > recording_duration + tolerance_seconds:
        raise ValueError(
            f"Annotation time span exceeds recording duration "
            f"{recording_duration:.1f}s"
        )
    events = [
        Event(
            event.stage,
            event.event,
            event.start_seconds - earliest_start,
            event.duration_seconds,
            event.location,
            annotation_start_offset,
        )
        for event in parsed_events
    ]
    events.sort(key=lambda event: (event.start_seconds, event.event))
    return events


def _overlap_duration(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, right) - max(start, left)) for left, right in intervals)


def _cap_sequences(events: list[Event], minimum_b_seconds: float) -> list[list[Event]]:
    sequences: list[list[Event]] = []
    for event in events:
        if not sequences:
            sequences.append([event])
            continue
        gap = event.start_seconds - sequences[-1][-1].end_seconds
        if gap < 0:
            raise ValueError("CAP-A events overlap; sequence timing is invalid")
        if minimum_b_seconds <= gap < 60.0:
            sequences[-1].append(event)
        else:
            sequences.append([event])
    return [sequence for sequence in sequences if len(sequence) >= 3]


def cap_features(events: list[Event], recording_duration: float) -> dict[str, float]:
    if not np.isfinite(recording_duration) or recording_duration <= 0:
        raise ValueError("EDF recording duration must be finite and positive")
    stage_names = {
        "SLEEP-S0": "wake", "SLEEP-S1": "n1", "SLEEP-S2": "n2",
        "SLEEP-S3": "n3", "SLEEP-S4": "n4", "SLEEP-REM": "rem",
        "SLEEP-UNSCORED": "unscored", "SLEEP-MT": "movement",
    }
    stage_events = [
        event for event in events
        if event.event in stage_names
    ]
    if not any(event.event in {"SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4"}
               for event in stage_events):
        raise ValueError("No NREM sleep-stage annotations; CAP rate is undefined")
    resolved_stages: list[tuple[float, float, str, bool]] = []
    imputed_epochs = 0
    for index, event in enumerate(stage_events):
        if abs(event.duration_seconds - 30.0) > 1e-6:
            raise ValueError("Sleep-stage events must represent 30-second epochs")
        if index:
            previous = stage_events[index - 1]
            start_difference = event.start_seconds - previous.start_seconds
            missing_epochs = round(start_difference / 30.0) - 1
            if (
                missing_epochs < 0
                or abs(start_difference - (missing_epochs + 1) * 30.0) > 1e-6
                or missing_epochs > 3
            ):
                raise ValueError("Sleep-stage sequence has an overlap or an unresolvable time gap")
            for gap_index in range(missing_epochs):
                gap_start = previous.end_seconds + gap_index * 30.0
                resolved_stages.append(
                    (gap_start, gap_start + 30.0, previous.event, True)
                )
                imputed_epochs += 1
        resolved_stages.append(
            (event.start_seconds, event.end_seconds, event.event, False)
        )
    explicit_stage_seconds = sum(event.duration_seconds for event in stage_events)
    stage_coverage_percent = 100.0 * explicit_stage_seconds / recording_duration
    if stage_coverage_percent < 80.0:
        raise ValueError(
            f"Sleep-stage annotation covers {stage_coverage_percent:.1f}% of the "
            "EDF duration; 80% is required"
        )
    nrem_names = {"SLEEP-S1", "SLEEP-S2", "SLEEP-S3", "SLEEP-S4"}
    nrem_intervals = [
        (start, end)
        for start, end, stage, _ in resolved_stages
        if stage in nrem_names
    ]
    if not nrem_intervals:
        raise ValueError("NREM sleep-stage annotations are absent after stage alignment")
    cap_events = sorted(
        (event for event in events if re.fullmatch(r"MCAP-A[123]", event.event)),
        key=lambda event: event.start_seconds,
    )
    cap_event_in_nrem: dict[int, bool] = {}
    cap_stage_label_mismatch_count = 0
    cap_outside_nrem_count = 0
    cap_unaligned_count = 0
    for event in cap_events:
        if not 2.0 <= event.duration_seconds <= 60.0:
            raise ValueError(
                f"CAP-A duration outside 2-60 seconds: {event.duration_seconds:g}s"
            )
        onset_stage = next(
            (
                stage for start, end, stage, _ in resolved_stages
                if start <= event.start_seconds < end
            ),
            None,
        )
        if onset_stage is None:
            cap_unaligned_count += 1
            cap_event_in_nrem[id(event)] = False
        else:
            is_nrem = onset_stage in nrem_names
            cap_event_in_nrem[id(event)] = is_nrem
            if not is_nrem:
                cap_outside_nrem_count += 1
            expected_label = onset_stage.removeprefix("SLEEP-")
            if event.stage != expected_label:
                cap_stage_label_mismatch_count += 1
    candidate_sequences = _cap_sequences(cap_events, minimum_b_seconds=2.0)
    candidate_reference_sequences = _cap_sequences(cap_events, minimum_b_seconds=0.0)
    valid_sequences = [
        sequence
        for sequence in candidate_sequences
        if all(cap_event_in_nrem[id(event)] for event in sequence)
    ]
    reference_sequences = candidate_reference_sequences
    valid_a = [event for sequence in valid_sequences for event in sequence]
    reference_a = [event for sequence in reference_sequences for event in sequence]
    sequence_time = sum(
        sequence[-1].start_seconds - sequence[0].start_seconds
        for sequence in valid_sequences
    )
    reference_sequence_time = sum(
        sequence[-1].start_seconds - sequence[0].start_seconds
        for sequence in reference_sequences
    )
    nrem_seconds = sum(end - start for start, end in nrem_intervals)
    reference_nrem_seconds = sum(
        end - start
        for start, end, stage, _ in resolved_stages
        if stage in nrem_names | {"SLEEP-MT"}
    )
    stage_intervals: dict[str, list[tuple[float, float]]] = {}
    for start, end, stage_event, _ in resolved_stages:
        stage = stage_names[stage_event]
        stage_intervals.setdefault(stage, []).append((start, end))
    stage_total = explicit_stage_seconds
    short_b_intervals = sum(
        0 <= current.start_seconds - previous.end_seconds < 2.0
        for previous, current in zip(cap_events, cap_events[1:])
    )
    output = {
        "nrem_seconds": float(nrem_seconds),
        "nrem_seconds_physionet_reference": float(reference_nrem_seconds),
        "cap_sequence_time_seconds": float(sequence_time),
        "cap_rate_percent": float(100.0 * sequence_time / nrem_seconds),
        "cap_rate_physionet_reference_percent": float(
            100.0 * reference_sequence_time / reference_nrem_seconds
        ),
        "cap_sequence_time_physionet_reference_seconds": float(reference_sequence_time),
        "cap_a_seconds_in_sequences": float(sum(e.duration_seconds for e in valid_a)),
        "cap_a_event_count": float(len(cap_events)),
        "cap_a_events_in_sequences": float(len(valid_a)),
        "cap_sequence_count": float(len(valid_sequences)),
        "cap_a_orphan_count": float(len(cap_events) - len(valid_a)),
        "cap_a_stage_label_mismatch_count": float(cap_stage_label_mismatch_count),
        "cap_a_outside_nrem_count": float(cap_outside_nrem_count),
        "cap_a_unaligned_count": float(cap_unaligned_count),
        "cap_sequence_excluded_nrem_count": float(
            len(candidate_sequences) - len(valid_sequences)
        ),
        "cap_short_b_interval_count": float(short_b_intervals),
        "cap_reference_sequence_count": float(len(reference_sequences)),
        "stage_annotation_imputed_epoch_count": float(imputed_epochs),
        "annotation_start_offset_seconds": float(
            events[0].annotation_start_offset_seconds if events else 0.0
        ),
        "stage_annotation_coverage_percent": float(
            min(100.0, 100.0 * stage_total / recording_duration)
        ),
    }
    for stage, intervals in stage_intervals.items():
        seconds = sum(end - start for start, end in intervals)
        output[f"sleep_{stage}_seconds"] = float(seconds)
        output[f"sleep_{stage}_percent_recording"] = float(
            100.0 * seconds / recording_duration
        )
    return output
