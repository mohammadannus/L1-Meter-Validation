from __future__ import annotations

import re
from collections import Counter
from typing import Any


class UniversalOCRConsolidator:
    """
    Model-independent OCR result consolidator.

    Combines Serial Number and IMEI candidates from
    multiple OCR regions.

    The consolidator handles:

        - exact duplicates
        - case differences
        - truncated OCR candidates
        - prefix/suffix relationships
        - cross-region voting
        - IMEI length preference

    Example:

        tile_1     -> U5020
        full_image -> U5020434
        tile_2     -> U5020434

    Final:

        U5020434
    """

    # =========================================================
    # Public API
    # =========================================================

    def consolidate(
        self,
        extracted_results: dict[
            str,
            dict[str, Any],
        ],
    ) -> dict[str, Any]:

        serial_candidates: list[str] = []
        imei_candidates: list[str] = []

        for result in extracted_results.values():

            serial = result.get(
                "serial_number",
                "",
            )

            imei = result.get(
                "imei",
                "",
            )

            if serial:
                serial_candidates.append(
                    serial.strip()
                )

            if imei:
                imei_candidates.append(
                    imei.strip()
                )

        return {
            "serial_number": (
                self._select_best_identifier(
                    serial_candidates
                )
            ),
            "imei": (
                self._select_best_imei(
                    imei_candidates
                )
            ),
        }

    # =========================================================
    # General identifier consolidation
    # =========================================================

    def _select_best_identifier(
        self,
        candidates: list[str],
    ) -> str:

        if not candidates:
            return ""

        cleaned = [
            value.strip()
            for value in candidates
            if value
            and value.strip()
        ]

        # Filter out frequency tokens, cellular band specs, or tokens with slashes
        valid_cleaned = []
        for val in cleaned:
            u_val = val.upper()
            if "/" in val or "\\" in val:
                continue
            if re.search(r"(?:MHZ|GHZ|KHZ|\d+HZ)\b", u_val):
                continue
            if any(term in u_val for term in ["CELLULAR", "BAND", "GPRS", "GSM", "LTE"]):
                continue
            valid_cleaned.append(val)

        if valid_cleaned:
            cleaned = valid_cleaned

        if not cleaned:
            return ""

        # -----------------------------------------------------
        # Normalize only for comparison.
        # Preserve the original candidate for output.
        # -----------------------------------------------------

        normalized: list[str] = [
            self._normalize(
                value
            )
            for value in cleaned
        ]

        # -----------------------------------------------------
        # Exact-frequency voting.
        # -----------------------------------------------------

        exact_counts = Counter(
            normalized
        )

        # -----------------------------------------------------
        # Build candidate groups.
        #
        # This allows:
        #
        # U5020
        # U5020434
        #
        # to belong to the same cluster.
        # -----------------------------------------------------

        groups: list[
            list[int]
        ] = []

        used: set[int] = set()

        for i, value_a in enumerate(
            normalized
        ):

            if i in used:
                continue

            group = [
                i
            ]

            used.add(i)

            for j in range(
                i + 1,
                len(normalized),
            ):

                if j in used:
                    continue

                value_b = normalized[j]

                if self._are_related(
                    value_a,
                    value_b,
                ):

                    group.append(j)

                    used.add(j)

            groups.append(
                group
            )

        # -----------------------------------------------------
        # Score each group.
        # -----------------------------------------------------

        best_group: list[int] | None = None

        best_score = float(
            "-inf"
        )

        for group in groups:

            group_values = [
                normalized[i]
                for i in group
            ]

            # Number of OCR regions supporting
            # this identifier family.
            region_support = len(
                group
            )

            # Prefer longer candidates because
            # shorter candidates are often truncated.
            longest_length = max(
                len(value)
                for value in group_values
            )

            # Frequency of the strongest exact candidate.
            strongest_exact_frequency = max(
                exact_counts[value]
                for value in group_values
            )

            # Prefer groups containing a candidate
            # with stronger repeated evidence.
            score = (
                region_support * 100
                + strongest_exact_frequency * 25
                + longest_length * 2
            )

            if score > best_score:

                best_score = score

                best_group = group

        if best_group is None:
            return ""

        # -----------------------------------------------------
        # Pick the longest candidate in the winning group.
        #
        # This is critical for:
        #
        # U5020
        # U5020434
        #
        # because U5020 is a truncated version.
        # -----------------------------------------------------

        best_indices = sorted(
            best_group,
            key=lambda i: (
                len(cleaned[i]),
                exact_counts[
                    normalized[i]
                ],
            ),
            reverse=True,
        )

        return cleaned[
            best_indices[0]
        ]

    # =========================================================
    # Determine whether two OCR candidates are related
    # =========================================================

    def _are_related(
        self,
        a: str,
        b: str,
    ) -> bool:

        if not a or not b:
            return False

        if a == b:
            return True

        # -----------------------------------------------------
        # Prefix relationship.
        #
        # Example:
        #
        # U5020
        # U5020434
        # -----------------------------------------------------

        if (
            len(a) >= 4
            and len(b) >= 4
        ):

            if a.startswith(b):
                return True

            if b.startswith(a):
                return True

        # -----------------------------------------------------
        # Suffix relationship.
        #
        # Useful for occasional OCR where the beginning
        # of an identifier is missed.
        # -----------------------------------------------------

        if (
            len(a) >= 6
            and len(b) >= 6
        ):

            if a.endswith(b):
                return True

            if b.endswith(a):
                return True

        # -----------------------------------------------------
        # Small edit-distance tolerance.
        #
        # Only use this for reasonably long identifiers.
        # This prevents unrelated short values from being
        # clustered together.
        # -----------------------------------------------------

        if (
            len(a) >= 7
            and len(b) >= 7
        ):

            distance = (
                self._levenshtein_distance(
                    a,
                    b,
                )
            )

            max_length = max(
                len(a),
                len(b),
            )

            # Allow approximately 10% OCR difference,
            # with a minimum tolerance of 1.
            allowed = max(
                1,
                int(
                    max_length * 0.10
                ),
            )

            if distance <= allowed:
                return True

        return False

    # =========================================================
    # IMEI consolidation
    # =========================================================

    def _select_best_imei(
        self,
        candidates: list[str],
    ) -> str:

        if not candidates:
            return ""

        cleaned = [
            value.strip()
            for value in candidates
            if value
            and value.strip()
        ]

        if not cleaned:
            return ""

        # -----------------------------------------------------
        # Normalize digits only.
        # -----------------------------------------------------

        normalized = [
            self._normalize_imei(
                value
            )
            for value in cleaned
        ]

        # -----------------------------------------------------
        # Group related IMEIs.
        # -----------------------------------------------------

        groups: list[
            list[int]
        ] = []

        used: set[int] = set()

        for i, value_a in enumerate(
            normalized
        ):

            if i in used:
                continue

            group = [
                i
            ]

            used.add(i)

            for j in range(
                i + 1,
                len(normalized),
            ):

                if j in used:
                    continue

                value_b = normalized[j]

                if self._are_imei_related(
                    value_a,
                    value_b,
                ):

                    group.append(j)

                    used.add(j)

            groups.append(
                group
            )

        # -----------------------------------------------------
        # Select strongest IMEI group.
        # -----------------------------------------------------

        best_group: list[int] | None = None

        best_score = float(
            "-inf"
        )

        for group in groups:

            group_values = [
                normalized[i]
                for i in group
            ]

            support = len(
                group
            )

            longest_length = max(
                len(value)
                for value in group_values
            )

            standard_imei_count = sum(
                1
                for value in group_values
                if len(value) == 15
            )

            score = (
                support * 100
                + standard_imei_count * 50
                + longest_length * 2
            )

            if score > best_score:

                best_score = score

                best_group = group

        if best_group is None:
            return ""

        # Prefer:
        #
        # 1. 15 digits
        # 2. longest value
        #
        best_indices = sorted(
            best_group,
            key=lambda i: (
                len(normalized[i]) == 15,
                len(normalized[i]),
            ),
            reverse=True,
        )

        return normalized[
            best_indices[0]
        ]

    # =========================================================
    # IMEI relationship
    # =========================================================

    def _are_imei_related(
        self,
        a: str,
        b: str,
    ) -> bool:

        if a == b:
            return True

        if not a or not b:
            return False

        # Prefix/suffix OCR truncation.
        if (
            len(a) >= 10
            and len(b) >= 10
        ):

            if a.startswith(b):
                return True

            if b.startswith(a):
                return True

            if a.endswith(b):
                return True

            if b.endswith(a):
                return True

        # Small OCR errors.
        if (
            len(a) >= 14
            and len(b) >= 14
        ):

            distance = (
                self._levenshtein_distance(
                    a,
                    b,
                )
            )

            if distance <= 2:
                return True

        return False

    # =========================================================
    # Normalization
    # =========================================================

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:

        return "".join(
            character
            for character in value.upper()
            if character.isalnum()
        )

    @staticmethod
    def _normalize_imei(
        value: str,
    ) -> str:

        return "".join(
            character
            for character in value
            if character.isdigit()
        )

    # =========================================================
    # Levenshtein distance
    # =========================================================

    @staticmethod
    def _levenshtein_distance(
        a: str,
        b: str,
    ) -> int:

        if a == b:
            return 0

        if not a:
            return len(b)

        if not b:
            return len(a)

        previous_row = list(
            range(
                len(b) + 1
            )
        )

        for i, char_a in enumerate(
            a,
            start=1,
        ):

            current_row = [
                i
            ]

            for j, char_b in enumerate(
                b,
                start=1,
            ):

                insertions = (
                    current_row[j - 1]
                    + 1
                )

                deletions = (
                    previous_row[j]
                    + 1
                )

                substitutions = (
                    previous_row[j - 1]
                    + (
                        char_a
                        != char_b
                    )
                )

                current_row.append(
                    min(
                        insertions,
                        deletions,
                        substitutions,
                    )
                )

            previous_row = (
                current_row
            )

        return previous_row[-1]