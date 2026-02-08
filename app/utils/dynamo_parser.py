"""DynamoDB JSON format parser.

DynamoDB exports data with type markers:
- {"S": "value"} -> "value" (String)
- {"N": "123"} -> 123 (Number)
- {"L": [{"S": "a"}, {"S": "b"}]} -> ["a", "b"] (List)
- {"M": {...}} -> {...} (Map/Dict)
- {"NULL": true} -> None
- {"BOOL": true} -> True (Boolean)
- {"SS": ["a", "b"]} -> ["a", "b"] (String Set)
- {"NS": ["1", "2"]} -> [1, 2] (Number Set)

This parser recursively flattens these type markers into native Python types.
"""

from typing import Any, Dict, List


class DynamoDBParser:
    """Parser for DynamoDB JSON format."""

    @staticmethod
    def parse(data: Any) -> Any:
        """
        Recursively parse DynamoDB typed JSON into native Python types.

        Args:
            data: DynamoDB JSON data (dict, list, or primitive)

        Returns:
            Native Python representation
        """
        if data is None:
            return None

        if isinstance(data, dict):
            # Check if this is a DynamoDB type wrapper
            if len(data) == 1:
                key = list(data.keys())[0]
                value = data[key]

                # String
                if key == "S":
                    return str(value)

                # Number (int or float)
                if key == "N":
                    num_str = str(value)
                    if "." in num_str:
                        return float(num_str)
                    return int(num_str)

                # Boolean
                if key == "BOOL":
                    return bool(value)

                # Null
                if key == "NULL":
                    return None

                # List
                if key == "L":
                    return [DynamoDBParser.parse(item) for item in value]

                # Map (nested dict)
                if key == "M":
                    return {k: DynamoDBParser.parse(v) for k, v in value.items()}

                # String Set
                if key == "SS":
                    return list(value)

                # Number Set
                if key == "NS":
                    return [float(n) if "." in str(n) else int(n) for n in value]

                # Binary (base64 string)
                if key == "B":
                    return value  # Keep as string

                # Binary Set
                if key == "BS":
                    return list(value)

            # Not a type wrapper, recursively parse all values
            return {k: DynamoDBParser.parse(v) for k, v in data.items()}

        if isinstance(data, list):
            return [DynamoDBParser.parse(item) for item in data]

        # Primitive type, return as-is
        return data

    @staticmethod
    def extract_nested_value(data: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        """
        Safely extract a nested value from parsed DynamoDB data.

        Args:
            data: Parsed data dictionary
            *keys: Sequence of keys to traverse
            default: Default value if path doesn't exist

        Returns:
            Value at the nested path or default
        """
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    @staticmethod
    def extract_list_of_strings(data: Dict[str, Any], *keys: str) -> List[str]:
        """
        Extract a list of strings from nested path.

        Handles both:
        - Direct list: ["a", "b"]
        - DynamoDB list: [{"S": "a"}, {"S": "b"}]

        Args:
            data: Parsed data dictionary
            *keys: Sequence of keys to traverse

        Returns:
            List of strings (empty list if path doesn't exist)
        """
        value = DynamoDBParser.extract_nested_value(data, *keys)

        if value is None:
            return []

        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    result.append(item)
                elif isinstance(item, dict) and "S" in item:
                    result.append(item["S"])
            return result

        if isinstance(value, str):
            return [value]

        return []


def parse_song_metadata(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single song item from DynamoDB export.

    Extracts and structures:
    - Core fields (id, title, prompt, etc.)
    - Technical metadata (bpm, key)
    - Classification (genre, mood, context)
    - Extended metadata (tags, instruments, etc.)

    Args:
        raw_item: Raw DynamoDB item

    Returns:
        Structured song data ready for database insertion
    """
    parser = DynamoDBParser()

    # Parse the entire item first
    item = parser.parse(raw_item)

    # Handle search_metadata which may still need parsing
    search_meta = item.get("search_metadata", {})
    if search_meta:
        search_meta = parser.parse(search_meta)

    # Extract technical metadata
    technical = search_meta.get("technical", {})
    if isinstance(technical, dict):
        technical = parser.parse(technical) if "M" in technical else technical

    # Extract core attributes
    core_attrs = search_meta.get("core_attributes", {})
    if isinstance(core_attrs, dict):
        core_attrs = parser.parse(core_attrs) if "M" in core_attrs else core_attrs

    # Extract genre info
    genre_info = core_attrs.get("genre", {})
    primary_genre = genre_info.get("primary_genre") if isinstance(genre_info, dict) else None

    # Extract mood info
    mood_info = core_attrs.get("mood", {})
    primary_mood = mood_info.get("primary_mood") if isinstance(mood_info, dict) else None

    # Extract context info
    context_info = core_attrs.get("context", {})
    primary_context = (
        context_info.get("primary_context") if isinstance(context_info, dict) else None
    )

    # Extract vocals info
    vocals_info = core_attrs.get("vocals", {})
    vocal_gender_list = vocals_info.get("vocal_gender", []) if isinstance(vocals_info, dict) else []
    vocal_gender = vocal_gender_list[0] if vocal_gender_list else None

    # Build extended metadata (everything else goes here)
    extended_metadata = {
        "algo_extra_tags": search_meta.get("algo_extra_tags", []),
        "all_tags": search_meta.get("all_tags", []),
        "instruments": core_attrs.get("instruments", {}),
        "mood_details": mood_info,
        "genre_details": genre_info,
        "context_details": context_info,
        "vocals_details": vocals_info,
        "additional_elements": core_attrs.get("additional_elements", []),
        "additional_tags": core_attrs.get("additional_tags", []),
        "sfx_descriptors": core_attrs.get("sfx_descriptors", []),
        "sfx_tags": core_attrs.get("sfx_tags", []),
        "sfx_category": core_attrs.get("sfx_category"),
    }

    # Parse BPM (handle string or int)
    bpm = technical.get("bpm")
    if isinstance(bpm, str):
        try:
            bpm = int(bpm)
        except ValueError:
            bpm = None

    return {
        # Core fields
        "id": item.get("id"),
        "title": item.get("title", "Untitled"),
        "prompt": item.get("prompt"),
        "lyrics": item.get("lyrics") or item.get("lyrics_1"),
        # Embedding source
        "acoustic_prompt_descriptive": search_meta.get("acoustic_prompt_descriptive"),
        # Promoted fields (indexed)
        "bpm": bpm,
        "musical_key": technical.get("key"),
        "primary_genre": primary_genre,
        "primary_mood": primary_mood,
        "format": search_meta.get("format"),
        "primary_context": primary_context,
        "vocal_gender": vocal_gender,
        # Extended metadata (JSONB)
        "extended_metadata": extended_metadata,
        # Audio outputs
        "outputs": [
            {
                "output_number": 1,
                "audio_url": item.get("conversion_path_1", ""),
                "sounds_description": item.get("sounds_1"),
            },
            {
                "output_number": 2,
                "audio_url": item.get("conversion_path_2", ""),
                "sounds_description": item.get("sounds_2"),
            },
        ],
    }
