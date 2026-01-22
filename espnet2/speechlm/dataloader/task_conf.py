#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unified task configuration definitions for SpeechLM.

This module provides centralized task configurations including:
- Supported data entries (audio1-10, text1-10, dialogue, speaker)
- Valid roles for chat templates (user, assistant, system)
- Task templates defining the structure of each task type
- Helper functions to access configuration data
"""

# List of supported entries for task configurations
SUPPORTED_ENTRIES = (
    [f"audio{i}" for i in range(1, 11)]  # audio1 to audio10
    + [f"text{i}" for i in range(1, 11)]  # text1 to text10
    + ["dialogue", "speaker"]
)

# Valid roles for chat templates
VALID_ROLES = ["assistant", "user", "system"]


# Unified task configuration with templates
# Template format: List of (role, entry) tuples, or special string ("dynamic", "dialogue")
TASK_CONFIGS = {
    "text_to_audio": {
        "template": [("user", "text1"), ("assistant", "audio1")],
    },
    "audio_to_text": {
        "template": [("user", "audio1"), ("assistant", "text1")],
    },
    "text_only": {
        "template": [("assistant", "text1")],
    },
    "dialogue": {
        "template": "dialogue",  # Special: entire data is dialogue format
    },
    "audio_to_text_interleave": {
        "template": "dynamic",  # Dynamic at runtime, as the number of audio-text pairs is not fixed
    },
}


def get_required_entries(task_name):
    """Derive required entries from task template.

    Args:
        task_name: Name of the task

    Returns:
        List of required entry names, or "dynamic" for dynamic tasks
    """
    template = TASK_CONFIGS[task_name]["template"]
    if template == "dynamic":
        return "dynamic"
    elif template == "dialogue":
        return ["dialogue"]
    else:
        return [entry for _, entry in template]


def get_template(task_name):
    """Get task template.

    Args:
        task_name: Name of the task

    Returns:
        Task template (list of tuples or special string)
    """
    return TASK_CONFIGS[task_name]["template"]


# Sanity check: validate task configurations at import time
def _validate_task_configs():
    """Validate that all entries and roles in TASK_CONFIGS are supported."""
    for task_name, config in TASK_CONFIGS.items():
        template = config.get("template")
        if template in ("dynamic", "dialogue"):
            continue  # Skip special templates

        for role, entry in template:
            if role not in VALID_ROLES:
                raise ValueError(
                    f"Invalid role '{role}' in task '{task_name}'. "
                    f"Must be one of: {VALID_ROLES}"
                )
            if entry not in SUPPORTED_ENTRIES:
                raise ValueError(
                    f"Invalid entry '{entry}' in task '{task_name}'. "
                    f"Must be one of: {SUPPORTED_ENTRIES}"
                )


_validate_task_configs()
