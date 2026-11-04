"""Text cleaning and normalization utilities."""

import unicodedata


class TextCleaner:
    """Clean extracted text while preserving its document structure."""

    def clean(self, text: str) -> str:
        """
        Normalize extracted text.

        Preserves:
        - Paragraph boundaries
        - Headings and line breaks
        - Markdown structure
        - Meaningful indentation

        Removes:
        - Null bytes
        - Invalid control characters
        - Excessive blank lines
        - Unnecessary whitespace
        """

        if not isinstance(text, str):
            raise TypeError("TextCleaner.clean() expects a string.")

        if not text:
            return ""

        # Normalize Unicode characters.
        text = unicodedata.normalize("NFC", text)

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Convert form-feed characters into line breaks.
        text = text.replace("\f", "\n")

        cleaned_lines = []

        for line in text.split("\n"):
            cleaned_chars = []

            for char in line:
                category = unicodedata.category(char)

                # Remove control and surrogate characters.
                if category in {"Cc", "Cs"} and char != "\t":
                    continue

                cleaned_chars.append(char)

            line = "".join(cleaned_chars)

            # Normalize horizontal whitespace without removing indentation.
            line = line.replace("\t", "    ").rstrip()

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Allow at most one empty line between content blocks.
        lines = text.split("\n")
        result = []
        blank_count = 0

        for line in lines:
            if not line.strip():
                blank_count += 1

                if blank_count <= 1:
                    result.append("")
            else:
                blank_count = 0
                result.append(line)

        return "\n".join(result).strip()

    def clean_text(self, text: str) -> str:
        """Alias for clean()."""
        return self.clean(text)