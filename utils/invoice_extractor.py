"""
Invoice document extractor — extracts Application No. and Account Id
from BESCOM (Bangalore Electricity Supply Company) procurement letters
and similar utility invoice documents.

Designed to work with raw OCR text output from PaddleOCR.

Supported document patterns:

    (2) Application. No:833291721618 , Account Id : 0350226046
    (2) Application Reg No. 1000983387 Account ID 2815613333
    Application No 833291721618
    Application. No:833291721618
    Application No.: 1000983387
    Account Id : 0350226046
    Account ID 2815613333
    Reg No. 1000983387
    vide Application. No: 1000983387
"""

from __future__ import annotations

import re
from typing import Any


class InvoiceExtractor:
    """
    Extracts Application No., Account Id, and other fields from
    invoice / procurement letter OCR text.
    """

    # =========================================================
    # Application Number patterns
    # =========================================================

    # Pattern priority: most specific → least specific

    APPLICATION_NO_PATTERNS = [
        # (2) Application. No:833291721618
        # (2) Application Reg No. 1000983387
        # (2) Application Reg N. 100983387
        # ApplicationNo764194469619
        # Application.No:833291721618
        # Application No.: 1000983387
        re.compile(
            r"application[\s\.\-_]*(?:reg[\s\.\-_]*)?(?:no|n)[\s\.\-_:#]*(\d{6,18})",
            re.IGNORECASE,
        ),

        # vide Application. No: 1000983387
        re.compile(
            r"vide[\s\.\-_]+application[\s\.\-_]*(?:reg[\s\.\-_]*)?(?:no|n)[\s\.\-_:#]*(\d{6,18})",
            re.IGNORECASE,
        ),

        # Reg. No. 1000983387 / Reg N. 100983387
        re.compile(
            r"(?:application[\s\.\-_]+)?reg[\s\.\-_]*(?:no|n)[\s\.\-_:#]*(\d{6,18})",
            re.IGNORECASE,
        ),
    ]

    # =========================================================
    # Account ID patterns
    # =========================================================

    ACCOUNT_ID_PATTERNS = [
        # Account Id : 0350226046
        # Account ID 2815613333
        # Account ld: 7432805454 (OCR reads 'ld' with lowercase 'l')
        # Account Id0350226046
        # Acunt D 281561333
        # Acct Id 0350226046
        re.compile(
            r"(?:account|acct|accnt|acunt)[\s\.\-_]*(?:id|ld|1d|ud|d)[\s\.\-_:#]*(\d{6,18})",
            re.IGNORECASE,
        ),
    ]

    # =========================================================
    # Reference Number patterns (BESCOM / SESCOM reference)
    # =========================================================

    REFERENCE_NO_PATTERNS = [
        # No:BESCOM/MTR-PROCURE1/3662399508/
        # N9:SESCOM/MTR-PROCURE1/5296165575/
        re.compile(
            r"(?:no|n[0-9]|ref)[\s\.\-_:#]*([A-Z0-9]ESCOM/[A-Z0-9\-/]+)",
            re.IGNORECASE,
        ),

        # Ref: (1) BESCOM/SEE-IT&MT/IT1/1291/24-25/2025-26/CYS-5259
        re.compile(
            r"ref[\s\.\-_:#]*\(?1\)?[\s\.\-_:#]*([A-Z0-9]ESCOM/[^\s,]+)",
            re.IGNORECASE,
        ),
    ]

    def __init__(self) -> None:
        pass

    # =========================================================
    # Public API
    # =========================================================

    def extract(self, text: str | None) -> dict[str, Any]:
        """
        Extract invoice fields from raw OCR text.

        Parameters
        ----------
        text : str or None
            Raw OCR text (newline-separated lines).

        Returns
        -------
        dict
            Keys: application_no, account_id, reference_no, raw_text
        """
        if not text:
            return {
                "application_no": "",
                "account_id": "",
                "reference_no": "",
                "raw_text": "",
            }

        # Normalize text
        normalized = self._normalize(text)

        # Extract fields
        application_no = self._extract_first_match(
            normalized, self.APPLICATION_NO_PATTERNS
        )
        account_id = self._extract_first_match(
            normalized, self.ACCOUNT_ID_PATTERNS
        )
        reference_no = self._extract_first_match(
            normalized, self.REFERENCE_NO_PATTERNS
        )

        # If we found application_no in multi-line context, also try
        # scanning line-by-line for better accuracy
        if not application_no:
            application_no = self._extract_multiline_application_no(text)

        if not account_id:
            account_id = self._extract_multiline_account_id(text)

        return {
            "application_no": application_no,
            "account_id": account_id,
            "reference_no": reference_no,
            "raw_text": text,
        }

    # =========================================================
    # Internal helpers
    # =========================================================

    def _normalize(self, text: str) -> str:
        """Normalize OCR text for pattern matching."""
        # Replace common OCR artifacts
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Normalize multiple spaces but keep newlines
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line.strip())
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _extract_first_match(
        self,
        text: str,
        patterns: list[re.Pattern],
    ) -> str:
        """Return the first matching group from the first matching pattern."""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_multiline_application_no(self, text: str) -> str:
        """
        Handle cases where 'Application' and the number are on separate lines.

        Example:
            Application
            No:833291721618
        
        Or:
            Application. No:
            833291721618
        """
        lines = self._prepare_lines(text)

        for i, line in enumerate(lines):
            # Check if this line contains "Application" label
            if re.search(r"application", line, re.IGNORECASE):
                # Check if number is on the same line (after label end)
                match = re.search(r"no\.?\s*[:.\-#]?\s*(\d{6,18})", line, re.IGNORECASE)
                if match:
                    return match.group(1)

                # Check next few lines for a standalone number
                for offset in range(1, 4):
                    if i + offset >= len(lines):
                        break
                    next_line = lines[i + offset].strip()

                    # Line starts with No: or No.
                    no_match = re.match(
                        r"no\.?\s*[:.\-#]?\s*(\d{6,18})",
                        next_line,
                        re.IGNORECASE,
                    )
                    if no_match:
                        return no_match.group(1)

                    # Line is just a number
                    num_match = re.fullmatch(r"\d{6,18}", next_line)
                    if num_match:
                        return num_match.group(0)

        return ""

    def _extract_multiline_account_id(self, text: str) -> str:
        """
        Handle cases where 'Account Id' and the number are on separate lines.
        """
        lines = self._prepare_lines(text)

        for i, line in enumerate(lines):
            if re.search(r"account\s*id", line, re.IGNORECASE):
                # Check same line
                match = re.search(r"id\s*[:.\-#]?\s*(\d{6,18})", line, re.IGNORECASE)
                if match:
                    return match.group(1)

                # Check next lines
                for offset in range(1, 3):
                    if i + offset >= len(lines):
                        break
                    next_line = lines[i + offset].strip()
                    num_match = re.fullmatch(r"\d{6,18}", next_line)
                    if num_match:
                        return num_match.group(0)

        return ""

    def _prepare_lines(self, text: str) -> list[str]:
        """Split text into cleaned lines."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                # Normalize whitespace
                line = re.sub(r"\s+", " ", line)
                lines.append(line)
        return lines
