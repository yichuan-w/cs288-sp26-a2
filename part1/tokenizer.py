"""
BPE Tokenizer implementation compatible with GPT-2 / tiktoken.
"""

from __future__ import annotations

import regex as re
from typing import Iterator


class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer compatible with GPT-2.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of BPE merge pairs (bytes, bytes)
            special_tokens: List of special token strings
        """
        self.vocab = vocab  # id -> bytes
        self.inverse_vocab = {v: k for k, v in vocab.items()}  # bytes -> id (also used as rank)
        self.merges = merges
        # Note: We use inverse_vocab for BPE ranking, not the merges list.
        # In GPT-2/tiktoken, the token ID serves as the rank - lower ID = higher priority.
        # This is different from naive BPE which uses merge order.
        
        # Handle special tokens
        self.special_tokens = special_tokens or []
        # Sort special tokens by length (descending) for longest-match-first
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        
        # Build special token to ID mapping
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.inverse_vocab:
                self.special_token_ids[token] = self.inverse_vocab[token_bytes]
        
        # GPT-2 regex pattern for pre-tokenization
        # This splits text into chunks that are tokenized independently
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Get all adjacent pairs of tokens."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE to a single token (sequence of bytes).
        Returns a list of merged byte sequences.
        
        Uses vocab ranks (token IDs) to determine merge priority.
        Lower token ID = higher priority (more common/earlier merge).
        
        Algorithm:
            1. Start with individual bytes as tokens
            2. While there are pairs that can be merged:
               a. Find the pair whose merged result has the lowest vocab rank
               b. Merge all occurrences of that pair
            3. Return final token list
        """
        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]
        
        if len(tokens) <= 1:
            return tokens
        
        while True:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break
            
            # Find the pair whose merged result has the lowest vocab rank (token ID)
            best_pair = None
            best_rank = float('inf')
            for pair in pairs:
                merged = pair[0] + pair[1]
                if merged in self.inverse_vocab:
                    rank = self.inverse_vocab[merged]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair
            
            if best_pair is None:
                break
            
            # Merge all occurrences of best_pair
            first, second = best_pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens

    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text by special tokens, preserving them.
        Returns list of (substring, is_special) tuples.
        """
        if not self.special_tokens_sorted:
            return [(text, False)] if text else []
        
        result = []
        remaining = text
        
        while remaining:
            # Find the earliest occurring special token
            earliest_pos = len(remaining)
            earliest_token = None
            
            for special in self.special_tokens_sorted:
                pos = remaining.find(special)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
                    earliest_token = special
            
            if earliest_token is None:
                # No special token found, add remaining text
                if remaining:
                    result.append((remaining, False))
                break
            else:
                # Add text before the special token
                if earliest_pos > 0:
                    result.append((remaining[:earliest_pos], False))
                # Add the special token
                result.append((earliest_token, True))
                remaining = remaining[earliest_pos + len(earliest_token):]
        
        return result

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encode a text chunk (without special tokens) to token IDs.
        
        Algorithm:
            1. Use regex pattern (self.pat) to split text into pre-tokens
            2. For each pre-token:
               a. Convert to bytes
               b. Apply BPE to get list of byte sequences
               c. Convert each byte sequence to token ID using inverse_vocab
               d. Handle unknown tokens by falling back to individual bytes
        """
        if not text:
            return []
        
        ids = []
        # Use regex pattern to split text into pre-tokens
        for match in self.pat.finditer(text):
            pre_token = match.group()
            token_bytes = pre_token.encode("utf-8")
            # Apply BPE
            bpe_tokens = self._bpe(token_bytes)
            for bt in bpe_tokens:
                if bt in self.inverse_vocab:
                    ids.append(self.inverse_vocab[bt])
                else:
                    # Fall back to individual bytes
                    for b in bt:
                        ids.append(self.inverse_vocab[bytes([b])])
        
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode a string to a list of token IDs.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        ids = []
        
        # Split by special tokens first
        parts = self._split_with_special_tokens(text)
        
        for part, is_special in parts:
            if is_special:
                # Add special token ID
                ids.append(self.special_token_ids[part])
            else:
                # Encode regular text
                ids.extend(self._encode_chunk(part))
        
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs to a string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        
        Algorithm:
            1. For each token_id, look up corresponding bytes in self.vocab
            2. Concatenate all byte chunks
            3. Decode as UTF-8 with errors="replace"
        """
        if not ids:
            return ""
        
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                byte_chunks.append(b"")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        Yields token IDs one at a time without loading entire input into memory.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        # Buffer for handling text that spans multiple lines
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # Process complete portions, keeping potential partial special tokens
            # Find the last safe split point
            safe_end = self._find_safe_split_point(buffer)
            
            if safe_end > 0:
                to_process = buffer[:safe_end]
                buffer = buffer[safe_end:]
                
                for token_id in self.encode(to_process):
                    yield token_id
        
        # Process remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _find_safe_split_point(self, text: str) -> int:
        """
        Find a safe point to split text for streaming encoding.
        We need to be careful not to split in the middle of:
        1. A potential special token
        2. A whitespace sequence (to preserve tokens like '\\n\\n')
        """
        if not text:
            return 0
        
        # Check if any special token could be starting at the end
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        
        # We need to keep at least max_special_len - 1 characters in buffer
        # to avoid splitting a special token
        min_keep = max_special_len - 1 if max_special_len > 0 else 0
        
        if len(text) <= min_keep:
            return 0
        
        safe_end = len(text)
        
        # Check for partial special token matches at the end
        for special in self.special_tokens:
            # Check if any prefix of special token matches end of text
            for prefix_len in range(1, len(special)):
                prefix = special[:prefix_len]
                if text.endswith(prefix):
                    safe_end = min(safe_end, len(text) - prefix_len)
        
        # Don't split in the middle of trailing whitespace
        # This prevents breaking up tokens like '\n\n'
        if safe_end > 0:
            # Find the last non-whitespace character
            last_non_ws = safe_end - 1
            while last_non_ws >= 0 and text[last_non_ws].isspace():
                last_non_ws -= 1
            
            # If there's trailing whitespace, don't include it in this chunk
            # unless the entire text is whitespace
            if last_non_ws >= 0 and last_non_ws < safe_end - 1:
                safe_end = last_non_ws + 1
        
        return safe_end


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Create a tokenizer from vocabulary and merge rules.
    
    Args:
        vocab: Mapping from token ID to bytes
        merges: List of BPE merge pairs
        special_tokens: Optional list of special token strings
        
    Returns:
        Tokenizer instance
    """
    return Tokenizer(vocab, merges, special_tokens)