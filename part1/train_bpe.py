"""
BPE (Byte Pair Encoding) training implementation.

This module implements the BPE algorithm for learning a tokenizer vocabulary
from a text corpus, compatible with GPT-2 style tokenization.
"""

from __future__ import annotations

import regex as re
from collections import Counter
from pathlib import Path
from typing import Iterator


# GPT-2 pre-tokenization pattern
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word (tuple of byte tokens)."""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of a pair in a word."""
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    """
    Pre-tokenize text using GPT-2 pattern, preserving special tokens.
    
    Special tokens are yielded as-is (not split by the regex pattern).
    """
    special_tokens = special_tokens or []
    
    if not special_tokens:
        # No special tokens, just use the pattern
        for match in GPT2_PAT.finditer(text):
            yield match.group()
        return
    
    # Sort special tokens by length (longest first) for greedy matching
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    # Build a pattern that matches special tokens
    import re as std_re
    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"
    
    # Split text by special tokens
    parts = std_re.split(split_pattern, text)
    
    for part in parts:
        if part in special_tokens:
            # Special token - yield as-is, but it won't be BPE-encoded
            # (we skip special tokens in the word frequency counting)
            continue
        elif part:
            # Regular text - apply GPT-2 pre-tokenization
            for match in GPT2_PAT.finditer(part):
                yield match.group()


def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a text file.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include (e.g., ["<|endoftext|>"])
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict mapping token_id (int) -> token (bytes)
        - merges: list of merge pairs in order they were learned [(bytes, bytes), ...]
    
    Algorithm Overview:
        BPE iteratively merges the most frequent pair of adjacent tokens until
        the vocabulary reaches the target size.
    
    Detailed Steps:
    
    1. VOCABULARY INITIALIZATION
       The initial vocabulary is built in this exact order:
       - First: Add special tokens (in the order provided)
       - Then: Add all 256 single-byte values (0x00 to 0xFF)
       
       Example with special_tokens=["<|endoftext|>"]:
         vocab = {
             0: b"<|endoftext|>",   # Special token first
             1: b"\\x00",           # Byte 0
             2: b"\\x01",           # Byte 1
             ...
             256: b"\\xff",         # Byte 255
         }
       
       So the initial vocab size = len(special_tokens) + 256
    
    2. WORD FREQUENCY COUNTING
       - Pre-tokenize the corpus using pre_tokenize(text, special_tokens)
       - For each pre-token, convert to bytes and represent as tuple of single bytes
       - Skip any word containing a "forbidden substring" (prefix of a special token)
       
       Example: "hello" -> (b'h', b'e', b'l', b'l', b'o')
       
       word_freqs is a Counter mapping: tuple[bytes, ...] -> frequency
    
    3. PAIR FREQUENCY COUNTING  
       Count how often each adjacent pair appears across ALL words, weighted by
       word frequency.
       
       Example: If word (b'h', b'e', b'l', b'l', b'o') appears 10 times:
         - pair (b'h', b'e') gets +10
         - pair (b'e', b'l') gets +10
         - pair (b'l', b'l') gets +10
         - pair (b'l', b'o') gets +10
    
    4. MERGE LOOP (repeat until vocab_size is reached)
       
       a. SELECT BEST PAIR (DETERMINISTIC TIE-BREAKING):
          Find the pair with highest frequency. If multiple pairs have the same
          frequency, select the lexicographically smallest pair.
          
          Lexicographic comparison on (bytes, bytes) tuples:
            - Compare first element as bytes
            - If equal, compare second element as bytes
          
          Example: If pairs (b'a', b'b') and (b'a', b'c') both have freq=100,
                   select (b'a', b'b') because b'b' < b'c'
          
          Implementation: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          This sorts by (frequency, pair) and takes the max.
                          Since we want highest freq but lowest pair for ties,
                          use: max(pair_counts, key=lambda p: (pair_counts[p], p))
                          
                          Note: Python compares bytes lexicographically by default.
       
       b. CREATE MERGED TOKEN:
          new_token = first + second  (bytes concatenation)
          Add to vocabulary with next available token_id
          Append (first, second) to merges list
       
       c. UPDATE WORD REPRESENTATIONS:
          For each word in word_freqs, apply the merge using merge_word()
          This replaces all occurrences of the pair with the merged token
       
       d. UPDATE PAIR COUNTS:
          Recompute pair frequencies for the updated words
          (Or incrementally update - subtract old pairs, add new pairs)
    
    5. RETURN
       Return (vocab, merges) where merges is the list of pairs in the order
       they were merged.
    
    Performance Note:
        A naive implementation recomputing all pair counts each iteration is O(n²).
        For efficiency, incrementally update pair counts by only processing words
        that contained the merged pair.
    """
    special_tokens = special_tokens or []
    
    # Read the corpus
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    
    # Build set of "forbidden" substrings from special tokens
    forbidden_substrings = set()
    for special in special_tokens:
        special_bytes = special.encode("utf-8")
        for i in range(2, len(special_bytes) + 1):
            forbidden_substrings.add(special_bytes[:i])
    
    # 1. Initialize vocabulary
    vocab = {}
    next_id = 0
    
    # Add special tokens first
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    # Add all 256 single-byte values
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1
    
    # 2. Word frequency counting
    word_freqs = Counter()
    for pre_token in pre_tokenize(text, special_tokens):
        word_bytes = pre_token.encode("utf-8")
        word_tuple = tuple(bytes([b]) for b in word_bytes)
        
        # Skip words that contain forbidden substrings (prefixes of special tokens)
        merged = b"".join(word_tuple)
        skip = False
        for fs in forbidden_substrings:
            if fs in merged:
                skip = True
                break
        if skip:
            continue
        
        word_freqs[word_tuple] += 1
    
    # 3. Compute initial pair frequencies (using set to deduplicate within each word)
    pair_counts = Counter()
    pair_to_words = {}
    
    for word, freq in word_freqs.items():
        seen_pairs = get_pairs(word)
        for p in seen_pairs:
            pair_counts[p] += freq
            if p not in pair_to_words:
                pair_to_words[p] = set()
            pair_to_words[p].add(word)
    
    merges = []
    num_merges = vocab_size - len(vocab)
    
    # 4. Merge loop
    for _ in range(num_merges):
        if not pair_counts:
            break
        
        # Select best pair: highest frequency, then lexicographically largest for tie-breaking
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        
        if pair_counts[best_pair] < 1:
            break
        
        first, second = best_pair
        new_token = first + second
        
        # Add to vocab
        vocab[next_id] = new_token
        next_id += 1
        merges.append(best_pair)
        
        # Update word representations and pair counts incrementally
        affected_words = pair_to_words.get(best_pair, set()).copy()
        
        # Collect all changes
        changes = []
        for word in affected_words:
            if word not in word_freqs:
                continue
            freq = word_freqs[word]
            if freq == 0:
                continue
            new_word = merge_word(word, best_pair)
            if new_word != word:
                changes.append((word, new_word, freq))
        
        for old_word, new_word, freq in changes:
            # Subtract old pairs (using set-based counting)
            old_pairs = get_pairs(old_word)
            for p in old_pairs:
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    if p in pair_counts:
                        del pair_counts[p]
                if p in pair_to_words:
                    pair_to_words[p].discard(old_word)
            
            # Handle case where new_word already exists
            if new_word in word_freqs and new_word != old_word:
                old_freq_of_new = word_freqs[new_word]
                new_pairs_existing = get_pairs(new_word)
                for p in new_pairs_existing:
                    pair_counts[p] = pair_counts.get(p, 0) - old_freq_of_new
                    if pair_counts.get(p, 0) <= 0:
                        if p in pair_counts:
                            del pair_counts[p]
                    if p in pair_to_words:
                        pair_to_words[p].discard(new_word)
                del word_freqs[old_word]
                word_freqs[new_word] = old_freq_of_new + freq
            else:
                del word_freqs[old_word]
                word_freqs[new_word] = freq
            
            # Add new pairs with full frequency (set-based)
            total_freq = word_freqs[new_word]
            new_pairs = get_pairs(new_word)
            for p in new_pairs:
                pair_counts[p] = pair_counts.get(p, 0) + total_freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(new_word)
    
    return vocab, merges
