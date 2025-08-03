import torch

def greedy_search(model, src_ids, src_mask, tokenizer_tgt, max_len):
    if hasattr(model, "module"):
        model = model.module 
    batch_size = src_ids.size(0)
    device = src_ids.device

    sos = tokenizer_tgt.token_to_id("[SOS]")
    eos = tokenizer_tgt.token_to_id("[EOS]")
    if sos is None or eos is None:
        raise RuntimeError(f"Could not find SOS/EOS in vocab (SOS={sos}, EOS={eos})")

    src_emb = model.src_embedding(src_ids)
    kv = model.encoder(src_emb, src_mask)

    ys = torch.full((batch_size, 1), sos, device=device, dtype=torch.long)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_emb = model.tgt_embedding(ys)
        tgt_seq_len = ys.size(1)
        tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).bool()
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        dec_out = model.decoder(
            tgt_emb,
            kv,
            target_mask=tgt_mask,
            kv_mask=src_mask
        )
        logits = model.output_proj(dec_out)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ys = torch.cat([ys, next_tok], dim=1)
        finished |= (next_tok.squeeze(-1) == eos)
        if finished.all():
            break

    return ys.tolist()

def beam_search(model, src_ids, src_mask, tokenizer_tgt, max_len, beam_size=4):
    if hasattr(model, "module"):
        model = model.module 

    batch_size = src_ids.size(0)
    device = src_ids.device

    sos = tokenizer_tgt.token_to_id("[SOS]")
    eos = tokenizer_tgt.token_to_id("[EOS]")
    if sos is None or eos is None:
        raise RuntimeError(f"Could not find SOS/EOS in vocab (SOS={sos}, EOS={eos})")

    src_emb = model.src_embedding(src_ids)
    kv = model.encoder(src_emb, src_mask)

    # create beam for each seq in batch
    beams = [
        [(torch.tensor([sos], device=device, dtype=torch.long), 0.0, False)] * beam_size
        for _ in range(batch_size)
    ]

    for _ in range(max_len - 1):
        new_beams = []
        for i in range(batch_size):
            candidates = []
            for seq, score, done in beams[i]:
                if done:
                    candidates.append((seq, score, True))
                    continue
                tgt_emb = model.tgt_embedding(seq.unsqueeze(0))
                tgt_seq_len = seq.size(0)
                tgt_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len), device=device)).bool()
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)
                dec_out = model.decoder(
                    tgt_emb,
                    kv[i:i+1],
                    target_mask=tgt_mask,
                    kv_mask=src_mask[i:i+1]
                )
                logits = model.output_proj(dec_out)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)
                for k in range(beam_size):
                    next_tok = topk_indices[0, k].item()
                    next_score = score + topk_log_probs[0, k].item()
                    next_seq = torch.cat([seq, torch.tensor([next_tok], device=device, dtype=torch.long)])
                    done_flag = done or (next_tok == eos)
                    candidates.append((next_seq, next_score, done_flag))
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_beams.append(candidates[:beam_size])
        beams = new_beams

    # select best sequence from each beam
    results = []
    for beam in beams:
        best_seq = max(beam, key=lambda x: x[1])[0]
        results.append(best_seq.tolist())
    return results 