from esm import pretrained
import torch

model, alphabet = pretrained.load_model_and_alphabet('esm2_t30_150M_UR50D')
model.to(device='cuda')
model.eval()

def load_fasta(is_train=True):
    if is_train:
        fasta_file = '/data2/rymiao/propka/data/fine_tune/fine_tune_train_expand_fasta/train.fasta'
    else:
        fasta_file = '/data2/rymiao/propka/data/fine_tune/fine_tune_small_fasta/test.fasta'
    fasta_map = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                name = line.strip().lstrip('>PDB_')
                fasta_map[name] = ''
            else:
                fasta_map[name] += line.strip()
    return fasta_map

def load_fasta_info(datas, return_contacts, repr_layers=[-1]):
    global model, alphabet
    batch_size = len(datas)
    seq_str_list = datas
    seq_encoded_list = [alphabet.encode(seq_str) for seq_str in seq_str_list]
    max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
    tokens = torch.empty(
        (
            batch_size,
            max_len + int(alphabet.prepend_bos) + int(alphabet.append_eos),
        ),
        dtype=torch.int64,
    )
    tokens.fill_(alphabet.padding_idx)
    strs = []

    for i, (seq_str, seq_encoded) in enumerate(
        zip(seq_str_list, seq_encoded_list)
    ):
        strs.append(seq_str)
        if alphabet.prepend_bos:
            tokens[i, 0] = alphabet.cls_idx
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        tokens[
            i,
            int(alphabet.prepend_bos) : len(seq_encoded)
            + int(alphabet.prepend_bos),
        ] = seq
        if alphabet.append_eos:
            tokens[i, len(seq_encoded) + int(alphabet.prepend_bos)] = alphabet.eos_idx
    
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    if torch.cuda.is_available():
        toks = tokens.to(device="cuda", non_blocking=True)

    out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

    representations = torch.vstack(list(out["representations"].values()))
    del toks, out
    return representations.sum(dim=1).squeeze().view(batch_size,-1)
print(load_fasta_info(list(load_fasta(is_train=True).values())[:2], return_contacts=True))