import datasets
import tiktoken
from transformers import AutoTokenizer
# Load dataset in streaming mode
ds = datasets.load_dataset("roneneldan/TinyStories", split="train")
hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
hf_tokenizer.save_pretrained("gpt2_local")
def check_dataset_exists():
    try:
        # Attempt to load the dataset with reuse_cache_if_exists mode
        datasets.load_dataset("parquet", data_files="Tinystories_train.parquet", split="train")
        datasets.load_dataset("parquet", data_files="Tiny_stories_test.parquet", split="train")
        return True
    except FileNotFoundError:
        return False
    
if not check_dataset_exists():
    print("Tokenized dataset does not exist locally... Generating and saving to disk.")

    def tokenize_and_chunk(dataset, tokenizer, chunk_size=512, train_rows=100_000, test_rows=500):
        """
        Tokenizes and chunks the dataset into fixed-length 512-token segments.
        The 'target' sequence is shifted left by 1 token.
        Stops after generating `train_rows + test_rows` tokenized chunks.
        """
        buffer = []  # Rolling buffer for tokens
        row_count = 0

        for example in dataset:
            tokens = tokenizer(example["text"], truncation=False, padding=False)['input_ids']
            buffer.extend(tokens)

            # Yield full chunks until we reach train_rows + test_rows
            while len(buffer) >= chunk_size + 1:  # +1 to ensure we can shift target
                if row_count >= (train_rows + test_rows):
                    return  # Stop yielding once enough rows are reached

                # Create input-target pairs
                input_chunk = buffer[:chunk_size]         # First 512 tokens
                target_chunk = buffer[1:chunk_size + 1]  # Shifted by 1 token
                
                # Assign to train or test split
                split = "train" if row_count < train_rows else "test"

                yield {
                    "split": split,
                    "input": input_chunk, 
                    "target": target_chunk
                }
                
                buffer = buffer[chunk_size:]  # Remove used tokens
                row_count += 1

    # Set the max number of rows for training and testing
    TRAIN_ROWS = 500000  # Adjust as needed
    TEST_ROWS = 500   # Adjust as needed
    CHUNK_SIZE = 256

    # Convert generator to a Hugging Face Dataset
    tokenized_ds = datasets.Dataset.from_generator(lambda: tokenize_and_chunk(ds, hf_tokenizer,chunk_size=CHUNK_SIZE, train_rows=TRAIN_ROWS, test_rows=TEST_ROWS))

    # Split the dataset into `train` and `test`
    dataset_splits = tokenized_ds.train_test_split(test_size=TEST_ROWS / (TRAIN_ROWS + TEST_ROWS), seed=42)

    # Save to disk
    dataset_splits["train"].to_parquet("Tinystories_train.parquet")
    dataset_splits["test"].to_parquet("Tinystories_test.parquet")

    print(f"✅ Saved {TRAIN_ROWS} train rows and {TEST_ROWS} test rows.")
else:
    print("Tokenized dataset already exists locally.")
ds_valid = datasets.load_dataset("roneneldan/TinyStories", split="validation")
ds_valid.save_to_disk("Tinystories_valid")

