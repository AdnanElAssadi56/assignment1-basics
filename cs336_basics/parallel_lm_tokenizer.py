from multiprocessing import Pool
from tqdm import tqdm
from .tokenizer import Tokenizer
from .pretokenization_example import find_chunk_boundaries
import os
import numpy as np

# Global variable to hold tokenizer instance for multiprocessing workers
_tokenizer_instance = None

def _initialize_worker_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]):
    """Initializes a tokenizer instance in each worker process."""
    print(f"DEBUG (Worker {os.getpid()}): Starting tokenizer initialization...", flush=True)
    global _tokenizer_instance
    _tokenizer_instance = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    print(f"DEBUG (Worker {os.getpid()}): Tokenizer initialization completed.", flush=True)

def _tokenize_lm_chunk(input_path: str, start: int, end: int, chunk_id: int, temp_dir: str) -> str:
    """Worker function to tokenize a text chunk and save to temporary file."""
    print(f"DEBUG (Worker {os.getpid()}): Processing chunk {chunk_id} bytes {start}-{end}...", flush=True)
    global _tokenizer_instance
    if _tokenizer_instance is None:
        raise RuntimeError("Tokenizer not initialized in worker process.")

    # Read and decode the chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
    print(f"DEBUG (Worker {os.getpid()}): Chunk text loaded ({len(chunk_text)} chars). Starting encoding...", flush=True)
    
    # Tokenize the chunk sequentially
    token_ids_list = _tokenizer_instance.encode(chunk_text)
    
    # Save to temporary file
    temp_file = os.path.join(temp_dir, f"chunk_{chunk_id:03d}.npy")
    np.save(temp_file, np.array(token_ids_list, dtype=np.uint16))
    
    print(f"DEBUG (Worker {os.getpid()}): Chunk {chunk_id} completed, saved {len(token_ids_list)} tokens to {temp_file}", flush=True)
    return temp_file

def parallel_tokenize_and_save(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    num_workers: int,
    progress_bar: bool = True
) -> None:
    """
    Tokenizes a text file in parallel and saves directly to output file.
    Workers save chunks to temporary files to avoid memory issues.
    """
    print(f"Starting parallel tokenization for LM data from {input_path} with {num_workers} workers...")
    print(f"Will save results directly to {output_path}")

    # Create temporary directory for chunk files
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="tokenize_chunks_")
    
    try:
        # Determine chunk boundaries
        with open(input_path, "rb") as f:
            split_token = special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>"
            boundaries = find_chunk_boundaries(f, num_workers, split_token)
        
        # Prepare arguments for each worker (include chunk_id and temp_dir)
        worker_args = [(input_path, s, e, i, temp_dir) for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        
        # Parallel processing - workers save to temp files
        pbar = tqdm(total=len(worker_args), desc="Tokenizing chunks in parallel", disable=not progress_bar)
        
        chunk_files = []
        with Pool(processes=num_workers, initializer=_initialize_worker_tokenizer, initargs=(vocab_path, merges_path, special_tokens)) as pool:
            for chunk_file in pool.starmap(_tokenize_lm_chunk, worker_args):
                chunk_files.append(chunk_file)
                pbar.update(1)
        
        pbar.close()

        # Combine all chunk files in order
        print("Combining tokenized chunks...")
        all_token_ids = []
        
        pbar2 = tqdm(total=len(chunk_files), desc="Combining chunks", disable=not progress_bar)
        
        for chunk_file in sorted(chunk_files):  # Ensure correct order
            chunk_tokens = np.load(chunk_file)
            all_token_ids.extend(chunk_tokens.tolist())
            os.remove(chunk_file)  # Clean up temporary file
            pbar2.update(1)
        
        pbar2.close()
        
        # Save final result
        print(f"Saving {len(all_token_ids):,} tokens to {output_path}...")
        np.save(output_path, np.array(all_token_ids, dtype=np.uint16))
        print(f"Successfully saved tokenized data to {output_path}")
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
