from __future__ import annotations

import re
from typing import Any


class UniversalOCRExtractor:
    """
    Model-independent OCR extractor for electricity-meter OCR.

    Extracts:
        - serial_number
        - imei

    Designed to work with OCR output from:
        - PaddleOCR

    Important design principle:

        We only accept a serial number when there is
        sufficient textual evidence that the value is
        actually associated with a serial-number label.

    Supported examples:

        SL NO U5028359
        SL. NO. U5028359

        Sr. No.
        U5028359

        Sr. No.
        8652933

        SL
        NO
        U5028359

        SL.
        U5028359

        NO. U5023881

        NO.
        U5023881

        Serial Number: U5028359

        Meter No MTR998877

        S/N
        U5028359

    IMEI examples:

        IMEI NO: 884524071203313

        IMEI
        NO
        884524071203313

        IMEI NO :
        884524071203313
    """

    # =========================================================
    # Serial-number labels (Tiered by certainty)
    # =========================================================

    EXPLICIT_SERIAL_LABEL_PATTERNS = [
        # Serial Number / Serial No
        re.compile(
            r"^\s*serial\s+(?:number|no\.?)[:\s#-]*$",
            re.IGNORECASE,
        ),

        # Sr. No / Sr No / Sr.
        re.compile(
            r"^\s*sr\.?\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*s\.?r\.?\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*sr\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        # Meter Number / Meter No / Meter ID
        re.compile(
            r"^\s*meter\s+(?:number|no\.?|id)[:\s#-]*$",
            re.IGNORECASE,
        ),

        # S/N / S.N.
        re.compile(
            r"^\s*s\s*/\s*n[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*s\.?\s*n\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        # SL NO / SL.NO / SI.NO / S1.NO
        re.compile(
            r"^\s*s[l|1iI]\.?\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        # SL. / SI. / S1.
        re.compile(
            r"^\s*s[l|1iI]\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        # S. NO. / S.NO. / S NO
        re.compile(
            r"^\s*s\.?\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        # SO. / SO NO (OCR reading L as O, e.g. SO.25385584)
        re.compile(
            r"^\s*s[o0]\.?\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*s[o0]\.?[:\s#-]*$",
            re.IGNORECASE,
        ),
    ]

    AMBIGUOUS_SERIAL_LABEL_PATTERNS = [
        # Standalone NO / NO.
        re.compile(
            r"^\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),
    ]

    SERIAL_LABEL_PATTERNS = EXPLICIT_SERIAL_LABEL_PATTERNS + AMBIGUOUS_SERIAL_LABEL_PATTERNS

    DEVICE_ID_LABEL_PATTERNS = [
        # Device ID / Device No / Device I0
        re.compile(
            r"^\s*device\s*(?:id|i0|no\.?)[:\s#-]*$",
            re.IGNORECASE,
        ),
    ]

    # =========================================================
    # IMEI labels
    # =========================================================

    IMEI_LABEL_PATTERNS = [

        re.compile(
            r"^\s*i?mei[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*i?mei\s*no\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*i?meino\.?[:\s#-]*$",
            re.IGNORECASE,
        ),

        re.compile(
            r"^\s*i?mei\s+number[:\s#-]*$",
            re.IGNORECASE,
        ),
    ]

    # =========================================================
    # Identifier patterns
    # =========================================================

    # Typical meter serial numbers.
    # Meter serial numbers never contain slashes or backslashes.
    SERIAL_VALUE_PATTERN = re.compile(
        r"\b[A-Z0-9][A-Z0-9\-]{4,24}\b",
        re.IGNORECASE,
    )

    # Standard IMEI is 15 digits.
    #
    # We allow 14-17 digits because OCR can occasionally
    # lose or duplicate a digit.
    IMEI_VALUE_PATTERN = re.compile(
        r"(?<!\d)\d{14,17}(?!\d)"
    )

    # =========================================================
    # Words that are clearly not serial numbers
    # =========================================================

    SERIAL_EXCLUSIONS = {
        "PHASE",
        "FPHASE",
        "WIRE",
        "METER",
        "SMART",
        "STATIC",
        "WATTHOUR",
        "MANUFACTURED",
        "MANUFACTURER",
        "NUMBER",
        "NO",
        "SL",
        "SERIAL",
        "IMEI",
        "DATE",
        "DIAL",
        "NET",
        "DATA",
        "AC",
        "KW",
        "KWH",
        "KVA",
        "KVAH",
        "VOLT",
        "VOLTAGE",
        "CURRENT",
        "FREQUENCY",
        "LATITUDE",
        "LONGITUDE",
        "WAN",
        "AURORA",
        "AUROR",
        "AUNOR",
        "SCHNEIDER",
        "ELECTRIC",
        "INDIA",
        "PRIVATE",
        "LIMITED",
        "MYSURU",
        "CONTRACT",
        "AWARD",
        "TYPE",
        "DLMS",
        "CAT",
        "WARRANTY",
        "PERIOD",
        "PROPERTY",
        "HESCOM",
        "BESCOM",
        "MESCOM",
        "GESCOM",
        "CESC",
        "DISCOM",
        "SEE",
        "LOI",
        "DWA",
        "TENDER",
        "ORDER",
        "SCROLL",
        "BATT",
        "OPTICAL",
        "PORT",
        "DISPLAY",
        "COMMUNICATION",
        "MODULE",
        "COVER",
        "SEAL",
        "TERMINAL",
        "BLOCK",
        "GUARANTEE",
        "STATUS",
        "CATEGORY",
        "DIRECT",
        "CONNECTED",
        "PRINCIPAL",
        "LN-KWN",
        "LNT",
        "SHARTMEER",
        "ACCOUNT",
        "ACC0UNT",
        "ACC1D",
        "SANLOAD",
        "CONSUMER",
        "PURCHASED",
        # Frequency / RF / Communication terms
        "MHZ",
        "GHZ",
        "KHZ",
        "HZ",
        "CELLULAR",
        "BAND",
        "BANDS",
        "GSM",
        "GPRS",
        "LTE",
        "WIFI",
        "BLUETOOTH",
        "BLE",
        "RF",
        "SAKSHAM",
        "GENUS",
        "BATTERY",
    }

    # Words which indicate that a standalone NO belongs
    # to something other than the meter serial number.
    NON_SERIAL_NO_CONTEXTS = {
        "IMEI",
        "CELLULAR",
        "WAN",
        "COMMUNICATION",
        "FREQUENCY",
        "MHZ",
        "GHZ",
        "BAND",
        "BANDS",
        "CONTRACT",
        "AWARD",
        "TENDER",
        "ORDER",
        "MODEL",
        "TYPE",
        "PART",
        "DATE",
        "PHONE",
        "MOBILE",
        "STANDARD",
        "IS",
        "DLMS",
        "CAT",
        "CLASS",
        "IMP",
        "KWH",
        "VOLT",
        "PHASE",
    }

    def __init__(
        self,
        context_window: int = 1,
    ) -> None:

        self.context_window = context_window

    # =========================================================
    # Public API
    # =========================================================

    def extract(
        self,
        text: str | None,
    ) -> dict[str, Any]:

        if not text:

            return {
                "serial_number": "",
                "imei": "",
            }

        lines = self._prepare_lines(
            text
        )

        serial_candidates = (
            self._extract_serial_candidates(
                lines
            )
        )

        imei_candidates = (
            self._extract_imei_candidates(
                lines
            )
        )

        return {
            "serial_number": (
                serial_candidates[0]
                if serial_candidates
                else ""
            ),
            "imei": (
                imei_candidates[0]
                if imei_candidates
                else ""
            ),
        }

    # =========================================================
    # Prepare OCR text
    # =========================================================

    def _prepare_lines(
        self,
        text: str,
    ) -> list[str]:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        raw_lines = text.split(
            "\n"
        )

        lines: list[str] = []

        for line in raw_lines:

            line = line.strip()

            if not line:
                continue

            # Remove common Markdown formatting
            # generated by VLM OCR.
            line = re.sub(
                r"[*_`#]+",
                " ",
                line,
            )

            # Normalize whitespace.
            line = re.sub(
                r"\s+",
                " ",
                line,
            ).strip()

            if line:
                lines.append(line)

        return lines

    # =========================================================
    # Serial extraction
    # =========================================================

    def _extract_serial_candidates(
        self,
        lines: list[str],
    ) -> list[str]:

        explicit_candidates: list[str] = []
        fallback_candidates: list[str] = []
        device_candidates: list[str] = []

        # -----------------------------------------------------
        # PASS 1: Same-line labels.
        #
        # Examples:
        #   SL NO U5028359
        #   Serial Number: ABC12345
        #   Meter No MTR998877
        #   Meter SI.No.U7143223
        #   SO.25385584
        #   Sr.No. 9044259
        # -----------------------------------------------------

        for line in lines:
            explicit_cand, fallback_cand = self._extract_serial_from_same_line_tiered(line)
            if explicit_cand:
                explicit_candidates.append(explicit_cand)
            elif fallback_cand:
                fallback_candidates.append(fallback_cand)
            else:
                dev_candidate = self._extract_device_id_from_same_line(line)
                if dev_candidate:
                    device_candidates.append(dev_candidate)

        # -----------------------------------------------------
        # PASS 2: Split / Multi-line labels.
        #
        # Examples:
        #   SL \n NO \n U5028359
        #   Sr.No. \n 9044259
        #   SL. \n U5023881
        # -----------------------------------------------------

        for index in range(len(lines)):
            line = lines[index]

            # -----------------------------------------------
            # Two-line label: SL \n NO or SR \n NO
            # -----------------------------------------------
            if index + 1 < len(lines):
                combined_label = f"{line} {lines[index + 1]}"
                if self._is_combined_serial_label(combined_label):
                    value_index = index + 2
                    if value_index < len(lines):
                        candidate = self._find_serial_value(lines[value_index])
                        if candidate:
                            explicit_candidates.append(candidate)

            # -----------------------------------------------
            # Single-line explicit label: Sr.No. \n 9044259
            # -----------------------------------------------
            if self._is_explicit_serial_label(line):
                if self._is_imei_related_label(lines, index):
                    continue

                found_forward = False
                for offset in (1, 2, 3):
                    value_index = index + offset
                    if value_index >= len(lines):
                        break

                    line_val = lines[value_index].strip()
                    if line_val in {":", "-", "--", ".", "#", ":-"}:
                        continue

                    candidate = self._find_serial_value(line_val)
                    if candidate:
                        explicit_candidates.append(candidate)
                        found_forward = True
                        break

                # Backward lookup if forward failed (for labels printed below serial)
                if not found_forward:
                    for offset in (1, 2):
                        value_index = index - offset
                        if value_index < 0:
                            break
                        line_val = lines[value_index].strip()
                        if line_val in {":", "-", "--", ".", "#", ":-"}:
                            continue
                        candidate = self._find_serial_value(line_val)
                        if candidate:
                            explicit_candidates.append(candidate)
                            break

            # -----------------------------------------------
            # Single-line ambiguous label: standalone NO. / NO
            # (Only low-priority fallback; must not be non-serial context)
            # -----------------------------------------------
            elif self._is_ambiguous_serial_label(line):
                if self._is_imei_related_label(lines, index) or self._is_non_serial_no(lines, index):
                    continue

                for offset in (1, 2, 3):
                    value_index = index + offset
                    if value_index >= len(lines):
                        break

                    line_val = lines[value_index].strip()
                    if line_val in {":", "-", "--", ".", "#", ":-"}:
                        continue

                    candidate = self._find_serial_value(line_val)
                    if candidate:
                        fallback_candidates.append(candidate)
                        break

            # -----------------------------------------------
            # Device ID label (fallback only)
            # -----------------------------------------------
            elif self._is_device_id_label(line):
                for offset in (1, 2):
                    value_index = index + offset
                    if value_index >= len(lines):
                        break
                    line_val = lines[value_index].strip()
                    if line_val in {":", "-", "--", ".", "#", ":-"}:
                        continue
                    candidate = self._find_serial_value(line_val)
                    if candidate:
                        device_candidates.append(candidate)
                        break

        # Priority resolution:
        # Tier 1 (Explicit serial labels) > Tier 2 (Ambiguous NO) > Tier 3 (Device ID)
        if explicit_candidates:
            selected = explicit_candidates
        elif fallback_candidates:
            selected = fallback_candidates
        else:
            selected = device_candidates

        return self._unique(selected)

    # =========================================================
    # Same-line serial extraction
    # =========================================================

    def _extract_serial_from_same_line_tiered(
        self,
        line: str,
    ) -> tuple[str, str]:
        """
        Extract serial candidate from a single line.
        Returns (explicit_candidate, fallback_candidate).
        """
        if re.search(r"\bimei\b", line, re.IGNORECASE):
            return "", ""

        explicit_patterns = [
            # Meter SI.No. / Meter Sl.No. / Sl.No / SI.No / S.No
            re.compile(r"\b(?:meter\s+)?s[l|1iI]\s*\.?\s*no\.?\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
            # Sr.No / Sr No / S.R. No
            re.compile(r"\bs\.?r\.?\s*no\.?\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
            # Serial Number / Serial No
            re.compile(r"\bserial\s+(?:number|no\.?)\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
            # Meter No / Meter Number / Meter ID / Meter: / Meter<digits>
            re.compile(r"\bmeter\s*(?:no\.?|number|id)?\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
            # S/N
            re.compile(r"\bs\s*/\s*n\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
            # S.NO.
            re.compile(r"\bs\.?\s*no\.?\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
            # SO. / SO NO (OCR reading L as O, e.g. SO.25385584)
            re.compile(r"\bs[o0]\.?\s*(?:no\.?)?\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
        ]

        ambiguous_patterns = [
            # Standalone line starting with NO. <serial>
            re.compile(r"^\s*no\s*\.?\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE),
        ]

        for pattern in explicit_patterns:
            match = pattern.search(line)
            if not match:
                continue
            candidate = self._find_serial_value(match.group(1))
            if candidate:
                return candidate, ""

        for pattern in ambiguous_patterns:
            match = pattern.search(line)
            if not match:
                continue
            candidate = self._find_serial_value(match.group(1))
            if candidate:
                return "", candidate

        return "", ""

    def _extract_serial_from_same_line(
        self,
        line: str,
    ) -> str:
        explicit_cand, fallback_cand = self._extract_serial_from_same_line_tiered(line)
        return explicit_cand or fallback_cand

    def _extract_device_id_from_same_line(
        self,
        line: str,
    ) -> str:
        if re.search(r"\bimei\b", line, re.IGNORECASE):
            return ""

        pattern = re.compile(r"\bdevice\s*(?:id|i0|no\.?)\s*[:#-]?\s*([A-Z0-9\-]{5,24})\b", re.IGNORECASE)
        match = pattern.search(line)
        if match:
            return self._find_serial_value(match.group(1))
        return ""

    # =========================================================
    # Label checks
    # =========================================================

    def _is_explicit_serial_label(
        self,
        text: str,
    ) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip())
        for pattern in self.EXPLICIT_SERIAL_LABEL_PATTERNS:
            if pattern.fullmatch(normalized):
                return True
        return False

    def _is_ambiguous_serial_label(
        self,
        text: str,
    ) -> bool:
        normalized = re.sub(r"\s+", " ", text.strip())
        for pattern in self.AMBIGUOUS_SERIAL_LABEL_PATTERNS:
            if pattern.fullmatch(normalized):
                return True
        return False

    def _is_serial_label(
        self,
        text: str,
    ) -> bool:
        return self._is_explicit_serial_label(text) or self._is_ambiguous_serial_label(text)

    def _is_device_id_label(
        self,
        text: str,
    ) -> bool:
        normalized = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )
        for pattern in self.DEVICE_ID_LABEL_PATTERNS:
            if pattern.fullmatch(normalized):
                return True
        return False

    def _is_combined_serial_label(
        self,
        text: str,
    ) -> bool:

        normalized = re.sub(
            r"\s+",
            " ",
            text.strip(),
        )

        normalized = normalized.upper()
        normalized = re.sub(r"[:#-]+$", "", normalized).strip()

        valid = {
            "SL NO",
            "SL NO.",
            "SL. NO",
            "SL. NO.",
            "SL.NO.",
            "SL.NO",
            "SR NO",
            "SR NO.",
            "SR. NO",
            "SR. NO.",
            "SR.NO.",
            "SR.NO",
            "S NO",
            "S NO.",
            "S. NO",
            "S. NO.",
            "S.NO.",
            "S.NO",
            "SO NO",
            "SO NO.",
            "SO. NO",
            "SO. NO.",
            "SO.NO.",
            "SO.NO",
            "SERIAL NUMBER",
            "SERIAL NO",
            "SERIAL NO.",
            "METER NUMBER",
            "METER NO",
            "METER NO.",
            "DEVICE ID",
            "DEVICE NO",
            "DEVICE NO.",
        }

        return normalized in valid

    # =========================================================
    # Context checks for ambiguous / IMEI labels
    # =========================================================

    def _is_non_serial_no(
        self,
        lines: list[str],
        index: int,
    ) -> bool:
        """
        Check if a standalone NO/NO. line is in a non-serial context
        (e.g., Cellular (850... NO ...MHZ), Contract Award NO, Type NO, IMEI NO).
        """
        current = lines[index].strip()
        if not re.fullmatch(r"no\.?[:\s#-]*", current, re.IGNORECASE):
            return False

        # Check surrounding lines (2 lines before, 2 lines after)
        start = max(0, index - 2)
        end = min(len(lines), index + 3)
        for i in range(start, end):
            if i == index:
                continue
            line_upper = lines[i].upper()
            if any(ctx in line_upper for ctx in self.NON_SERIAL_NO_CONTEXTS):
                return True
            # If followed within 2 lines by an IMEI (14-17 digits), it's part of IMEI NO
            if i > index and re.search(r"(?<!\d)\d{14,17}(?!\d)", lines[i]):
                return True
        return False

    def _is_imei_related_label(
        self,
        lines: list[str],
        index: int,
    ) -> bool:

        current = lines[index].strip()
        if re.search(r"\bimei\b", current, re.IGNORECASE):
            return True

        # Check preceding lines
        for offset in (1, 2):
            if index - offset >= 0:
                if re.search(r"\bimei\b", lines[index - offset], re.IGNORECASE):
                    return True

        # If followed within 2 lines by an IMEI number, it belongs to IMEI
        for offset in (1, 2):
            if index + offset < len(lines):
                if re.search(r"(?<!\d)\d{14,17}(?!\d)", lines[index + offset]):
                    return True

        return False

    # =========================================================
    # Find serial value
    # =========================================================

    def _find_serial_value(
        self,
        text: str,
    ) -> str:

        if not text:
            return ""

        # Never extract IMEI as serial.
        if re.search(
            r"\bimei\b",
            text,
            re.IGNORECASE,
        ):
            return ""

        cleaned = text.strip()

        # Remove common separators.
        cleaned = re.sub(
            r"^[\s:;,\-./]+",
            "",
            cleaned,
        )

        matches = (
            self.SERIAL_VALUE_PATTERN.findall(
                cleaned
            )
        )

        for value in matches:

            normalized = (
                self._normalize_identifier(
                    value
                )
            )

            if not normalized:
                continue

            # Meter serial numbers never contain slashes or backslashes
            if "/" in normalized or "\\" in normalized:
                continue

            if len(normalized) < 5 or len(normalized) > 24:
                continue

            # A meter serial number MUST contain digits.
            if sum(c.isdigit() for c in normalized) < 2:
                continue

            upper_value = (
                normalized.upper()
            )

            # Reject MTCTE marks and OCR corruptions
            if re.search(r"M[IT]C?TE|CCTEZ|OCTEZ|WICIE|ICTE", upper_value):
                continue

            # Reject frequency units (e.g. 2300MHZ, 2500MHZ, 50HZ, 2.4GHZ, 1800MHZ)
            if re.search(r"(?:MHZ|GHZ|KHZ|\d+HZ)\b", upper_value):
                continue

            # Reject cellular / RF / communication keywords
            if any(term in upper_value for term in ["CELLULAR", "BAND", "GPRS", "GSM", "LTE"]):
                continue

            # Reject known non-serial words.
            if (
                upper_value in self.SERIAL_EXCLUSIONS
                or any(part in self.SERIAL_EXCLUSIONS for part in re.split(r"[\/\-_]", upper_value) if part)
            ):
                continue

            # Reject dates (e.g. 05/26, 11/2025, 23.12.2024, 07126).
            if re.search(r"^\d{1,2}[./-]\d{2,4}$|^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$", normalized):
                continue

            # Reject electrical ratings (e.g. 5-30A, 10-60A, 240V, 3X240V, 50HZ, 27C, 1P2W, 3P4W)
            if re.search(r"^\d+-\d+[A-Z]$|^\d+V$|^\d+X\d+V$|^\d+P\d+W$|^\d+HZ$|^\d+C$", upper_value):
                continue

            # Reject impulse ratings (e.g. 3200IMP, 1600IMP)
            if "IMP" in upper_value:
                continue

            # Reject tender/contract/award prefixes
            if any(term in upper_value for term in ["BESCOM", "HESCOM", "MESCOM", "GESCOM", "DISCOM", "DLMS"]):
                continue

            # Reject pure short alphabetic words.
            if (
                normalized.isalpha()
                and len(normalized) < 6
            ):
                continue

            # Reject short numeric values
            if normalized.isdigit():
                if len(normalized) < 6:
                    continue

            return normalized

        return ""

    # =========================================================
    # IMEI extraction
    # =========================================================

    def _extract_imei_candidates(
        self,
        lines: list[str],
    ) -> list[str]:

        candidates: list[str] = []

        # -----------------------------------------------------
        # PASS 1
        #
        # IMEI on same line.
        #
        # IMEI NO: 884524071203313
        # -----------------------------------------------------

        for line in lines:

            if not re.search(
                r"\b(?:i?mei|imeino)\b",
                line,
                re.IGNORECASE,
            ):

                continue

            candidate = (
                self._find_imei_value(
                    line
                )
            )

            if candidate:

                candidates.append(
                    candidate
                )

        # -----------------------------------------------------
        # PASS 2
        #
        # Split IMEI.
        #
        # IMEI
        # NO
        # 884524071203313
        #
        # IMEI NO
        # 884524071203313
        #
        # IMEI NO
        # Cellular (850/900...
        # 860376088548717
        # -----------------------------------------------------

        for index, line in enumerate(
            lines
        ):

            if not re.search(
                r"\b(?:i?mei|imeino)\b",
                line,
                re.IGNORECASE,
            ):

                continue

            for offset in range(
                1,
                5,
            ):

                value_index = (
                    index + offset
                )

                if (
                    value_index
                    >= len(lines)
                ):

                    break

                subsequent_line = lines[value_index].strip()

                # Stop if another major section begins
                if self._is_serial_label(subsequent_line):
                    break

                candidate = (
                    self._find_imei_value(
                        subsequent_line
                    )
                )

                if candidate:

                    candidates.append(
                        candidate
                    )

                    break

        # -----------------------------------------------------
        # PASS 3
        #
        # Strict Fallback: scan for any 15-digit cellular IMEI starting
        # with standard TAC prefixes (86, 35, 84).
        # Slashes and non-IMEI contexts (Account, DWA, Tender) are rejected.
        # -----------------------------------------------------
        if not candidates:
            for line in lines:
                if "/" in line:
                    continue
                upper = line.upper()
                if any(
                    k in upper
                    for k in [
                        "ACCOUNT", "ACC0UNT", "ACC1D", "SANLOAD",
                        "CONTRACT", "DWA", "TENDER", "AWARD",
                        "MHZ", "GHZ", "CELLULAR", "WAN", "PHONE",
                        "MOBILE", "LATITUDE", "LONGITUDE",
                    ]
                ):
                    continue
                clean = re.sub(r"[\s\-]+", "", line)
                matches = self.IMEI_VALUE_PATTERN.findall(clean)
                for cand in matches:
                    if len(cand) == 15 and cand.startswith(("86", "35", "84")):
                        candidates.append(cand)
                        break

        return self._unique(
            candidates
        )

    # =========================================================
    # Find IMEI value
    # =========================================================

    def _find_imei_value(
        self,
        text: str,
    ) -> str:

        if not text or "/" in text:
            return ""

        upper = text.upper()
        if any(
            k in upper
            for k in [
                "ACCOUNT", "ACC0UNT", "ACC1D", "SANLOAD",
                "CONTRACT", "DWA", "TENDER", "AWARD",
                "MHZ", "GHZ", "CELLULAR", "WAN",
            ]
        ):
            if "IMEI" not in upper:
                return ""

        # Remove spaces and hyphens only, NEVER slashes
        clean = re.sub(r"[\s\-]+", "", text)
        matches = self.IMEI_VALUE_PATTERN.findall(clean)

        if not matches:
            return ""

        # Prefer standard 15-digit IMEI.
        fifteen_digit = [
            value
            for value in matches
            if len(value) == 15
        ]

        if fifteen_digit:

            return fifteen_digit[0]

        return matches[0]

    # =========================================================
    # Utility methods
    # =========================================================

    @staticmethod
    def _normalize_identifier(
        value: str,
    ) -> str:

        value = value.strip()

        value = re.sub(
            r"[.,:;]+$",
            "",
            value,
        )

        return value

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in values:

            key = value.upper()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                value
            )

        return result