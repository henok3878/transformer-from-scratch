import os 
import argparse 
from typing import Iterator
from datasets import load_dataset, Dataset, IterableDataset, IterableDatasetDict 
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE 
from tokenizers.trainers import BpeTrainer 
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from src.config import TokenizationStrategy, load_config 

def get_all_sentences(ds: IterableDataset | IterableDatasetDict, lang: str):
    """Generator to yield sentences for a single language from the dataset"""
    for item in ds:
        yield item['translation'][lang] 

def get_all_sentences_combined(ds: IterableDataset | IterableDatasetDict, lang_src: str, lang_tgt: str) -> Iterator[str]:
    """Generator to yield sentences from both src and tgt languages."""
    for item in ds:
        yield item['translation'][lang_src]
        yield item['translation'][lang_tgt] 

def build_byte_level_bpe_tokenizer(texts: Iterator, vocab_size: int):
    """Builds and trains a Bytelevel BPE tokenizer."""
    tokenizer = Tokenizer(BPE(unk_token="[UNK]")) 
    tokenizer.pre_tokenizer = ByteLevel() # type: ignore
    # decoder to convert tokens back to text
    tokenizer.decoder = ByteLevelDecoder() # type: ignore

    # special tokens
    special_tokens = [
        AddedToken("[UNK]", single_word=True),
        AddedToken("[PAD]", single_word=True),
        AddedToken("[SOS]", single_word=True),
        AddedToken("[EOS]", single_word=True),
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size, special_tokens=special_tokens,min_frequency=2, ) # type: ignore  

    tokenizer.train_from_iterator(texts, trainer) 

    post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        pair="[SOS] $A [EOS] [SOS] $B [EOS]",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ])
    tokenizer.post_processor = post_processor # type: ignore 

    return tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    strategy = config.data.tokenization_strategy 

    os.makedirs("tokenizers", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    if strategy == TokenizationStrategy.JOINT:
        vocab_size = config.model.src_vocab_size
        tokenizer_path = f"tokenizers/tokenizer-joint-{config.data.subset}-vocab{vocab_size}.json" 
        processed_dir = f"data/processed-joint-{config.data.subset}-vocab{vocab_size}" 
        
        if not os.path.exists(tokenizer_path):
            raw_ds = load_dataset(config.data.dataset_name, config.data.subset, split="train", streaming=True)
            assert isinstance(raw_ds, IterableDataset)
            texts = get_all_sentences_combined(raw_ds, config.data.lang_src, config.data.lang_tgt) 
            tokenizer = build_byte_level_bpe_tokenizer(texts, vocab_size) 
            tokenizer.save(tokenizer_path)
            print(f"Saved joint tokenizer to {tokenizer_path}")
        else:
            print(f"Joint tokenizer already exists at {tokenizer_path}")
            tokenizer = Tokenizer.from_file(tokenizer_path)
        
        def tokenize_fn(examples):
            src_texts = [ex[config.data.lang_src] for ex in examples["translation"]]
            tgt_texts = [ex[config.data.lang_tgt] for ex in examples["translation"]]
            src_encodings = tokenizer.encode_batch(src_texts)
            tgt_encodings = tokenizer.encode_batch(tgt_texts)
            return {"src_ids": [e.ids for e in src_encodings], "tgt_ids": [e.ids for e in tgt_encodings]}

    elif strategy == TokenizationStrategy.SEPARATE:
        processed_dir = f"data/processed-separate-{config.data.subset}-src{config.model.src_vocab_size}-tgt{config.model.tgt_vocab_size}"

        # source tokenizer
        tokenizer_src_path = f"tokenizers/tokenizer-separate-{config.data.lang_src}-vocab{config.model.src_vocab_size}.json"
        if not os.path.exists(tokenizer_src_path):
            print(f"Building source tokenizer ({config.data.lang_src}, vocab size: {config.model.src_vocab_size})...")
            raw_ds_obj = load_dataset(config.data.dataset_name, config.data.subset, split='train', streaming=True)
            assert isinstance(raw_ds_obj, IterableDataset)
            texts = get_all_sentences(raw_ds_obj, config.data.lang_src)
            tokenizer_src = build_byte_level_bpe_tokenizer(texts, config.model.src_vocab_size)
            tokenizer_src.save(tokenizer_src_path)
            print(f"Saved source tokenizer to {tokenizer_src_path}")
        else:
            print(f"Source tokenizer already exists at {tokenizer_src_path}")
            tokenizer_src = Tokenizer.from_file(tokenizer_src_path)
        
        # target tokenizer
        tokenizer_tgt_path = f"tokenizers/tokenizer-separate-{config.data.lang_tgt}-vocab{config.model.tgt_vocab_size}.json"
        if not os.path.exists(tokenizer_tgt_path):
            print(f"Building tokenizer ({config.data.lang_tgt}, vocab size: {config.model.tgt_vocab_size})...")
            raw_ds_obj = load_dataset(config.data.dataset_name, config.data.subset, split='train', streaming=True)
            assert isinstance(raw_ds_obj, IterableDataset)
            texts = get_all_sentences(raw_ds_obj, config.data.lang_tgt)
            tokenizer_tgt = build_byte_level_bpe_tokenizer(texts, config.model.tgt_vocab_size)
            tokenizer_tgt.save(tokenizer_tgt_path)
            print(f"Saved target tokenizer to {tokenizer_tgt_path}")
        else:
            print(f"Target tokenizer already exists at {tokenizer_tgt_path}")
            tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_path)

        def tokenize_fn(examples):
            src_texts = [ex[config.data.lang_src] for ex in examples["translation"]]
            tgt_texts = [ex[config.data.lang_tgt] for ex in examples["translation"]]

            src_encodings = tokenizer_src.encode_batch(src_texts)
            tgt_encodings = tokenizer_tgt.encode_batch(tgt_texts)
            return {"src_ids": [e.ids for e in src_encodings], "tgt_ids": [e.ids for e in tgt_encodings]}
    else:
        raise ValueError(f"Expected joint or separate tokenization strategy but provided{strategy}")

    # apply tokenization and save 
    if os.path.exists(processed_dir):
        print(f"Processed data already exists. Skipping tokenization...")
    else:
        print("Loading dataset for tokenization...")
        raw_ds_full = load_dataset(config.data.dataset_name, config.data.subset, split='train')
        assert isinstance(raw_ds_full, Dataset)
        
        print("Tokenizing dataset...")
        tokenized_ds = raw_ds_full.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=os.cpu_count(),
            remove_columns=raw_ds_full.column_names
        )
        
        print(f"Saving tokenized dataset to {processed_dir}...")
        tokenized_ds.save_to_disk(processed_dir)

    print("Preprocessing complete!") 

if __name__ == '__main__':
    main() 
