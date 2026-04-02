"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import csv
import json
from collections import namedtuple, defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple, Optional

import torch
from tg4perfetto import TraceGenerator
import pandas as pd


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    """
    Decode a profiler tag into (block_idx, group_idx, event_idx, event_type, sm_id).
    Tag layout:
      bits 0-1: event_type
      bits 2-11: event_idx
      bits 12-23: block_group_idx
      bits 24-31: sm_id
    """
    sm_id = (tag >> 24) & 0xFF
    block_group_idx = (tag >> 12) & 0xFFF
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    block_idx = block_group_idx // num_groups
    group_idx = block_group_idx % num_groups
    return block_idx, group_idx, event_idx, event_type, sm_id


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    event_names: List[str],
    file_name: str,
) -> None:
    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)
    tgen = TraceGenerator(file_name)

    # First pass: collect sm_id for each block_idx
    block_to_sm = {}
    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        block_idx, group_idx, event_idx, event_type, sm_id = decode_tag(
            tag, num_blocks, num_groups
        )
        if block_idx not in block_to_sm:
            block_to_sm[block_idx] = sm_id

    sm_pid_map = {}      # sm_id -> perfetto group
    track_map: Dict[Tuple[int, int], Any] = {}  # (block_idx, group_idx) -> track

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type, sm_id = decode_tag(
            tag, num_blocks, num_groups
        )

        # Group by SM, one track per (block, group) — all events on same line
        if sm_id not in sm_pid_map:
            sm_pid_map[sm_id] = tgen.create_group(f"SM_{sm_id:03d}")
        sm_group = sm_pid_map[sm_id]

        event = event_names[event_idx]
        tkey = (block_idx, group_idx)
        if tkey not in track_map:
            track_map[tkey] = sm_group.create_track(f"blk{block_idx}_g{group_idx}")
        track = track_map[tkey]

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()


def analyze_sm_performance(
    profiler_buffer: torch.Tensor,
    task_info: Dict[str, Dict],
    event_names: List[str] = None,
) -> Optional["pd.DataFrame"]:
    """
    Analyze SM performance by matching profiler events with task information.

    Args:
        profiler_buffer: Profiler buffer tensor
        task_info: Task information dict from BatchAttention._extract_task_info()
        event_names: List of event names (e.g., ["prefill", "decode", "reduction"])

    Returns:
        DataFrame with columns: decode_kv_len, decode_qo_len, prefill_kv_len, prefill_qo_len, sm_time
    """

    if event_names is None:
        event_names = ["prefill", "decode", "reduction"]

    assert profiler_buffer.dtype == torch.uint64
    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    # Map event names to task indices
    # Assuming: decode=0, prefill=1 based on typical event ordering
    task_event_map = {}
    for idx, name in enumerate[str](event_names):
        if name == "prefill":
            task_event_map[0] = idx
        elif name == "decode":
            task_event_map[1] = idx

    # Collect events by (sm_id, block_idx, group_idx, event_idx)
    events = []
    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type, sm_id = decode_tag(
            tag, num_blocks, num_groups
        )
        events.append((sm_id, block_idx, group_idx, event_idx, event_type, timestamp))

    # Group events by (sm_id, block_idx, group_idx, event_idx) and calculate durations
    event_times: defaultdict[Any, dict[str, int | None]] = defaultdict(
        lambda: {"begin": None, "end": None}
    )
    for sm_id, block_idx, group_idx, event_idx, event_type, timestamp in events:
        key = (sm_id, block_idx, group_idx, event_idx)
        if event_type == EventType.kBegin.value:
            event_times[key]["begin"] = timestamp  # type: ignore
        elif event_type == EventType.kEnd.value:
            event_times[key]["end"] = timestamp  # type: ignore

    # Group events by (sm_id, block_idx) and store as list of dicts
    sm_block_event_durations = defaultdict[Any, defaultdict[Any, list]](
        lambda: defaultdict[Any, list](list)
    )
    for (sm_id, block_idx, group_idx, event_idx), times in event_times.items():  # noqa
        if event_idx >= len(event_names):
            continue

        sm_block_event_durations[sm_id][block_idx].append(
            {
                "event": event_idx,
                "duration": (times["end"] or 0) - (times["begin"] or 0),  # type: ignore
                "begin": times["begin"],  # type: ignore
                "end": times["end"],  # type: ignore
            }
        )

    # Sort events by time within each block
    for sm_id in sm_block_event_durations:
        for block_idx in sm_block_event_durations[sm_id]:
            sm_block_event_durations[sm_id][block_idx].sort(key=lambda x: x["begin"])

    # Calculate total time per SM, aggregated by task
    sm_task_times: defaultdict[Any, dict[str, float]] = defaultdict(
        lambda: {"decode": 0.0, "prefill": 0.0}
    )
    sm_task_lengths = defaultdict[Any, dict[str, int]](
        lambda: {
            "decode_kv_len": 0,
            "decode_qo_len": 0,
            "prefill_kv_len": 0,
            "prefill_qo_len": 0,
        }
    )

    # For each SM, process blocks and measure overlap time
    for sm_id, block_events in sm_block_event_durations.items():  # noqa
        # Get all block indices for this SM
        block_indices = list[Any](block_events.keys())

        if len(block_indices) < 2:
            # Only one block, process all events
            for block_idx, events in block_events.items():  # noqa
                for event_info in events:
                    event_idx = event_info["event"]  # type: ignore
                    duration = event_info["duration"]  # type: ignore
                    # Map event_idx to task
                    for task_name, task_idx in [("prefill", 0), ("decode", 1)]:
                        if task_event_map.get(task_idx) == event_idx:
                            sm_task_times[sm_id][task_name] += duration  # type: ignore
                            break
        else:
            # Two blocks: find which one lasts longer and filter overlap
            block0_idx, block1_idx = block_indices[0], block_indices[1]
            events0 = block_events[block0_idx]
            events1 = block_events[block1_idx]

            # Find last event end time for each block
            last_end0 = max(e["end"] for e in events0) if events0 else 0
            last_end1 = max(e["end"] for e in events1) if events1 else 0

            # Determine which block lasts longer
            if last_end0 >= last_end1:
                longer_block_idx = block0_idx
                shorter_block_idx = block1_idx
                shorter_last_end = last_end1
            else:
                longer_block_idx = block1_idx
                shorter_block_idx = block0_idx
                shorter_last_end = last_end0

            # Process shorter block: all events count
            for event_info in block_events[shorter_block_idx]:
                event_idx = event_info["event"]  # type: ignore
                duration = event_info["duration"]  # type: ignore
                for task_name, task_idx in [("prefill", 0), ("decode", 1)]:
                    if task_event_map.get(task_idx) == event_idx:
                        sm_task_times[sm_id][task_name] += duration  # type: ignore
                        break

            # Process longer block: filter out events that start after shorter block ends
            # For events that overlap, clip duration to overlap period
            for event_info in block_events[longer_block_idx]:
                if event_info["begin"] <= shorter_last_end:  # type: ignore
                    event_idx = event_info["event"]  # type: ignore
                    # Clip duration to overlap period (don't count time after shorter block ends)
                    duration = min(
                        event_info["duration"],
                        shorter_last_end - event_info["begin"],  # type: ignore
                    )
                    for task_name, task_idx in [("prefill", 0), ("decode", 1)]:
                        if task_event_map.get(task_idx) == event_idx:
                            sm_task_times[sm_id][task_name] += duration  # type: ignore
                            break

    # Now aggregate lengths using block_idx to index into work_indptr
    for task_name, task_idx in [("prefill", 0), ("decode", 1)]:
        if task_name not in task_info:
            continue

        info = task_info[task_name]
        work_indptr = info["work_indptr"]
        q_len = info["q_len"]
        kv_len = info["kv_len"]

        # Get the event_idx for this task
        task_event_idx = task_event_map.get(task_idx)
        if task_event_idx is None:
            continue

        # For each SM and block_idx, get work items and aggregate lengths
        for sm_id, block_events in sm_block_event_durations.items():
            for block_idx, events in block_events.items():  # noqa
                # Check if this block has events for this task
                has_task_event = any(e["event"] == task_event_idx for e in events)  # type: ignore
                if not has_task_event:
                    continue

                # Use block_idx directly to index into work_indptr
                if block_idx >= len(work_indptr) - 1:
                    continue

                work_start = int(work_indptr[block_idx].item())
                work_end = int(work_indptr[block_idx + 1].item())

                # Sum lengths for this block
                block_qo_len = 0
                block_kv_len = 0
                for work_idx in range(work_start, work_end):
                    if work_idx < len(q_len) and work_idx < len(kv_len):
                        block_qo_len += int(q_len[work_idx].item())
                        block_kv_len += int(kv_len[work_idx].item())

                sm_task_lengths[sm_id][f"{task_name}_kv_len"] += block_kv_len
                sm_task_lengths[sm_id][f"{task_name}_qo_len"] += block_qo_len

    # Create dataframe
    rows = []
    for sm_id in sorted(sm_task_times.keys()):
        sm_times = sm_task_times[sm_id]
        lengths = sm_task_lengths[sm_id]

        # Convert timestamps to milliseconds (assuming timestamp is in some unit)
        # Adjust conversion factor based on actual timestamp unit
        # For now, assuming timestamps are in nanoseconds
        decode_time_ms = sm_times.get("decode", 0) / 1e6
        prefill_time_ms = sm_times.get("prefill", 0) / 1e6
        total_time_ms = decode_time_ms + prefill_time_ms

        rows.append(
            {
                "decode_kv_len": lengths.get("decode_kv_len", 0),
                "decode_qo_len": lengths.get("decode_qo_len", 0),
                "prefill_kv_len": lengths.get("prefill_kv_len", 0),
                "prefill_qo_len": lengths.get("prefill_qo_len", 0),
                "decode_time_ms": decode_time_ms,
                "prefill_time_ms": prefill_time_ms,
                "sm_time": total_time_ms,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df
