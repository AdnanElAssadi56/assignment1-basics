from collections import Counter, defaultdict
import regex as re
from multiprocessing import Pool
from .pretokenization_example import find_chunk_boundaries
from typing import Optional

PRETOKENIZE_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_on_specials(text, special_tokens):
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, text)

# For naive implementation
def count_pairs(seqs):
    c = Counter()
    for seq in seqs:
        for i in range(len(seq) - 1):
            c[(seq[i], seq[i + 1])] += 1
    return c

def merge_sequence(seq, pair, new_id):
    a, b = pair
    out = []
    i = 0
    while i < len(seq):
        if i < len(seq)-1 and seq[i]==a and seq[i+1]==b:
            out.append(new_id) 
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out

def pretokenize(text, special_tokens):
    sequences = []
    for segment in split_on_specials(text, special_tokens):
        if not segment:
            continue
        for m in re.finditer(PRETOKENIZE_REGEX, segment):
            tok_bytes = m.group(0).encode("utf-8")
            sequences.append(list(tok_bytes))
    return sequences

def _process_chunk(input_path: str, start: int, end: int, special_tokens: list[str]):
    sequences = []
    pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")" if special_tokens else None
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pieces = re.split(pattern, chunk) if pattern is not None else [chunk]
    for piece in pieces:
        if special_tokens and piece in special_tokens:
            continue
        for m in re.finditer(PRETOKENIZE_REGEX, piece):
            tok_bytes = m.group(0).encode("utf-8")
            sequences.append(list(tok_bytes))
    return sequences


def _pretokenize_parallel(input_path: str, special_tokens: list[str], num_workers: int):
    with open(input_path, "rb") as f:
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
        boundaries = find_chunk_boundaries(f, num_workers, split_token)
    args = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    sequences: list[list[int]] = []
    with Pool(processes=num_workers) as pool:
        for chunk_sequences in pool.starmap(_process_chunk, args):
            sequences.extend(chunk_sequences)
    return sequences


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_workers: Optional[int] = None):
    """
    Train a byte-level BPE tokenizer.

    Returns:
        vocab (dict[int, bytes])
        merges (list[tuple[bytes, bytes]])
    """
    if num_workers is None or num_workers <= 1:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        sequences = pretokenize(text, special_tokens)
    else:
        sequences = _pretokenize_parallel(input_path, special_tokens, num_workers)

    
    merges = []
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    pre_token_freq = Counter(tuple(seq) for seq in sequences)

    # Initialize pair frequencies once
    pair_freq = Counter()
    # Index mapping pair -> set of pretokens that contain it
    pair_to_pretokens = defaultdict(set)
    for pretoken, freq in pre_token_freq.items():
        if len(pretoken) < 2:
            continue
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_freq[pair] += freq
            pair_to_pretokens[pair].add(pretoken)

    while len(vocab) < vocab_size:

        if not pair_freq:
            break
        most_common_pair = max(
            pair_freq.items(),
            key=lambda item: (item[1], (vocab[item[0][0]], vocab[item[0][1]])),
        )[0]

        a, b = most_common_pair
        vocab[next_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        affected_pretokens = pair_to_pretokens.get(most_common_pair, set())
        if not affected_pretokens:
            pair_freq.pop(most_common_pair, None)
            pair_to_pretokens.pop(most_common_pair, None)
            next_id += 1
            continue

        # Collect new memberships to add after processing all affected pretokens
        new_memberships = defaultdict(set)

        for old_pre in list(affected_pretokens):
            freq = pre_token_freq.get(old_pre, 0)
            if freq == 0:
                continue

            # Count pairs before
            before_pairs = Counter(zip(old_pre, old_pre[1:])) if len(old_pre) >= 2 else Counter()

            # Merge all occurrences within this pretoken
            pre_list = list(old_pre)
            i = 0
            while i < len(pre_list) - 1:
                if pre_list[i] == a and pre_list[i + 1] == b:
                    pre_list[i : i + 2] = [next_id]
                    # do not advance i to catch overlapping
                else:
                    i += 1

            new_pre = tuple(pre_list)

            # Count pairs after
            after_pairs = Counter(zip(new_pre, new_pre[1:])) if len(new_pre) >= 2 else Counter()

            # Update pair frequencies via delta, weighted by pretoken frequency
            for pair, cnt in before_pairs.items():
                pair_freq[pair] -= cnt * freq
            for pair, cnt in after_pairs.items():
                pair_freq[pair] += cnt * freq

            # Update index memberships for this pretoken
            for pair in before_pairs:
                s = pair_to_pretokens.get(pair)
                if s is not None:
                    s.discard(old_pre)
            for pair in after_pairs:
                new_memberships[pair].add(new_pre)

            # Move frequency mass in pretoken table
            pre_token_freq[old_pre] -= freq
            if pre_token_freq[old_pre] <= 0:
                del pre_token_freq[old_pre]
            pre_token_freq[new_pre] += freq

        # Apply new memberships
        for pair, pre_set in new_memberships.items():
            pair_to_pretokens[pair].update(pre_set)

        # Remove the merged pair entries
        pair_freq.pop(most_common_pair, None)
        pair_to_pretokens.pop(most_common_pair, None)

        next_id += 1

    return vocab, merges
