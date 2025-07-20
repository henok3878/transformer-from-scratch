import argparse 
from datasets import load_from_disk 
from concurrent.futures import ProcessPoolExecutor 

def count_tokens(batch):
    src = sum(len(item) for item in batch["src_ids"])
    tgt = sum(len(item) for item in batch["tgt_ids"])
    return src, tgt 

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--processed_dir", type=str, default="data/processed-joint-de-en-vocab37000/train")
    parser.add_argument("--num_workers", type=int, default=8) 
    
    args = parser.parse_args()

    ds = load_from_disk(args.processed_dir) 
    batch_size = 1000 
    src_total = 0 
    tgt_total = 0 
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        batches = [ds[i:i+batch_size] for i in range(0, len(ds), batch_size)] 
        for src,tgt in executor.map(count_tokens, batches):
            src_total += src 
            tgt_total += tgt 
    
    print(f"Totol source tokens: {src_total}")
    print(f"Total target tokens: {tgt_total}")

if __name__ == "__main__":
    main() 

"""
wmt19: 
    Totol source tokens: 841370392
    Total target tokens: 870185119

wmt14: 
    Totol source tokens: 140867621
    Total target tokens: 144259471
"""