import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "sigmoid": nn.Sigmoid,
               "elu": nn.ELU, "prelu": nn.PReLU}


class MLP(nn.Module):
    """Flatten the image, then `len(hidden)` blocks of Linear -> activation -> Dropout, then a
    linear output layer. With hidden=() this is multinomial logistic regression (the linear
    baseline of Q2.1).

    Args:
        in_dim: number of input features (3 * 64 * 64 = 12288)
        num_classes: number of output classes
        hidden: sequence of hidden-layer widths, e.g. (512, 256)
        activation: key of ACTIVATIONS
        dropout: dropout probability after each hidden activation
    """

    def __init__(self, in_dim=3 * 64 * 64, num_classes=10, hidden=(), activation="relu", dropout=0.0):
        super().__init__()
        layers, d = [nn.Flatten()], in_dim
        for width in hidden:
            layers += [nn.Linear(d, width), ACTIVATIONS[activation]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def features(self, x):
        """Activations of the last hidden layer (batch, hidden[-1]), e.g. to use the MLP as an
        image encoder in Part 3."""
        return self.net[:-1](x)

    @property
    def feature_dim(self):
        return self.net[-1].in_features

    @property
    def first_layer(self):
        """The first nn.Linear (its weight rows are the images visualized in Q2.4)."""
        return self.net[1]


class CNN(nn.Module):
    """A small VGG-style CNN: `len(channels)` blocks of [conv 3x3 (padding 1) -> (BatchNorm) ->
    activation] x `convs_per_block`, each block followed by 2x2 max pooling, then global average
    pooling over the remaining positions, dropout, and a linear output layer.

    Locality and weight sharing come from the 3x3 convolutions (Q1.1 (a)); padding 1 keeps the
    convolutions translation equivariant away from the border (Q1.1 (b)); global average pooling
    makes the feature vector (approximately) translation invariant. Max pooling with stride 2 only
    keeps exact equivariance for shifts that are multiples of 2 ** len(channels).

    Args:
        in_channels: 3 for RGB
        num_classes: number of output classes
        channels: output channels of each block, e.g. (32, 64, 128, 256)
        convs_per_block: number of 3x3 convolutions per block
        batchnorm: BatchNorm2d after each convolution
        activation: key of ACTIVATIONS
        dropout: dropout probability before the output layer

    `features(x)` returns the pooled feature vector (batch, channels[-1]), e.g. to use the CNN as
    an image encoder in Part 3.
    """

    def __init__(self, in_channels=3, num_classes=10, channels=(32, 64, 128, 256),
                 convs_per_block=1, batchnorm=True, activation="relu", dropout=0.0):
        super().__init__()
        layers, c_in = [], in_channels
        for c_out in channels:
            for _ in range(convs_per_block):
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=not batchnorm))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(c_out))
                layers.append(ACTIVATIONS[activation]())
                c_in = c_out
            layers.append(nn.MaxPool2d(2))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))

    def features(self, x):
        return self.pool(self.conv(x)).flatten(1)

    @property
    def feature_dim(self):
        return self.head[-1].in_features

    def forward(self, x):
        return self.head(self.pool(self.conv(x)))


class CaptionDecoding(nn.Module):
    """Decoding shared by the captioners (GRU and Transformer). A subclass provides

        encode(images) -> (state, context): `state` has the batch in dimension 1 (the GRU hidden
            state (1, B, H); for the Transformer the tokens generated so far (1, B, t)), `context`
            in dimension 0 (or None);
        step(tokens, state, context) -> (log-probabilities (B, V) of the next token, new state);

    and the attributes encoder, freeze_encoder, pad_id, bos_id, eos_id."""

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:   # a frozen encoder stays in eval mode (BatchNorm, dropout)
            self.encoder.eval()
        return self


    @torch.no_grad()
    def greedy(self, images, max_len=25):
        """Greedy decoding: token ids (B, <= max_len), each row ending with <eos> or cut at
        max_len; positions after <eos> are <pad>."""
        h, img = self.encode(images)
        B = images.shape[0]
        tokens = torch.full((B,), self.bos_id, dtype=torch.long, device=images.device)
        done = torch.zeros(B, dtype=torch.bool, device=images.device)
        result = []
        for _ in range(max_len):
            logp, h = self.step(tokens, h, img)
            tokens = logp.argmax(-1)
            tokens = torch.where(done, torch.full_like(tokens, self.pad_id), tokens)
            result.append(tokens)
            done |= tokens == self.eos_id
            if done.all():
                break
        return torch.stack(result, dim=1)

    @torch.no_grad()
    def beam_search(self, images, beam_size=3, length_norm=False, max_len=25):
        """Beam search. Every image keeps `beam_size` unfinished hypotheses, ranked by their
        log-probability. A hypothesis that ends with <eos> moves to the image's list of finished
        captions, and the next-best extension takes its place in the beam. An image is done once
        it has `beam_size` finished captions. Hypotheses that are still open after max_len tokens
        are also added as candidates. Among the finished captions the result is the one with the
        highest log p(y | x), or with length_norm, log p(y | x) / L (L = number of tokens incl.
        <eos>). beam_size=1 without length_norm is greedy decoding.

        Returns token ids (B, T) padded with <pad> (like `greedy`) and log p(y | x) (B,) of the
        returned captions."""
        h, img = self.encode(images)
        B, K, device = images.shape[0], beam_size, images.device
        h = h.repeat_interleave(K, dim=1)                          # (1, B*K, H)
        img = img.repeat_interleave(K, dim=0) if img is not None else None
        seqs = [[[] for _ in range(K)] for _ in range(B)]          # generated ids per beam
        last = torch.full((B * K,), self.bos_id, dtype=torch.long, device=device)
        scores = torch.full((B, K), float("-inf"), device=device)
        scores[:, 0] = 0.0                                         # all beams start identical
        finished = [[] for _ in range(B)]                          # (log p, ids incl. <eos>)
        done = [False] * B
        for t in range(max_len):
            logp, h = self.step(last, h, img)
            V = logp.shape[-1]
            cand = (scores.unsqueeze(-1) + logp.view(B, K, V)).view(B, K * V)
            # Among the 2K best extensions at least K do not end with <eos>.
            top_s, top_i = cand.topk(2 * K, dim=1)
            top_s, top_i = top_s.cpu().tolist(), top_i.cpu().tolist()
            src = torch.zeros(B, K, dtype=torch.long)
            words = torch.full((B, K), self.pad_id, dtype=torch.long)
            new_scores = torch.full((B, K), float("-inf"))
            for b in range(B):
                if done[b]:
                    continue
                k = 0
                for j, (s, i) in enumerate(zip(top_s[b], top_i[b])):
                    beam, word = divmod(i, V)
                    if word == self.eos_id:
                        if j < K:                    # only <eos> among the K best extensions
                            finished[b].append((s, seqs[b][beam] + [word]))
                    elif k < K:
                        src[b, k], words[b, k], new_scores[b, k] = beam, word, s
                        k += 1
                if len(finished[b]) >= K:
                    done[b] = True
                else:
                    seqs[b] = [seqs[b][src[b, k]] + [words[b, k].item()] for k in range(K)]
            if all(done):
                break
            flat = (torch.arange(B).unsqueeze(1) * K + src).view(-1).to(device)
            h = h[:, flat]
            last = words.view(-1).to(device)
            scores = new_scores.to(device)
        for b in range(B):                                         # open after max_len
            if not done[b]:
                finished[b] += [(scores[b, k].item(), seqs[b][k]) for k in range(K)
                                if scores[b, k] > float("-inf")]
        key = (lambda c: c[0] / len(c[1])) if length_norm else (lambda c: c[0])
        best = [max(f, key=key) for f in finished]
        T = max(len(ids) for _, ids in best)
        out = torch.full((B, T), self.pad_id, dtype=torch.long)
        for b, (_, ids) in enumerate(best):
            out[b, :len(ids)] = torch.tensor(ids)
        return out, torch.tensor([s for s, _ in best])



class Captioner(CaptionDecoding):
    """Encoder-decoder captioner in the style of Vinyals et al. (2015) (Part 3).

    encoder: an MLP or CNN from Part 2; `encoder.features(images)` gives one vector per image.
    The vector is projected to the initial hidden state of a GRU decoder (h0 = tanh(W f + b));
    with `concat_image=True` the projected vector is also concatenated to the word embedding at
    every time step. The GRU output goes through dropout and a linear layer to the vocabulary.

    Args:
        encoder: module with .features(x) and .feature_dim
        vocab_size: size of the output vocabulary (including the special tokens)
        pad_id, bos_id, eos_id: ids of <pad>, <bos>, <eos>
        embed_dim, hidden_dim: word embedding and GRU state sizes
        dropout: dropout on the embeddings and on the GRU outputs
        concat_image: also feed the image vector at every step
        freeze_encoder: keep the encoder fixed (no gradients, always in eval mode)
    """

    def __init__(self, encoder, vocab_size, pad_id, bos_id, eos_id, embed_dim=256, hidden_dim=512,
                 dropout=0.3, concat_image=False, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        self.concat_image = concat_image
        self.init_h = nn.Linear(encoder.feature_dim, hidden_dim)
        self.image_in = nn.Linear(encoder.feature_dim, embed_dim) if concat_image else None
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim * (2 if concat_image else 1), hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)


    def encode(self, images):
        """Image vectors and the decoder's context: (h0 (1, B, H), per-step image input or None)."""
        if self.freeze_encoder:
            with torch.no_grad():
                f = self.encoder.features(images)
        else:
            f = self.encoder.features(images)
        h0 = torch.tanh(self.init_h(f)).unsqueeze(0)
        img = self.image_in(f) if self.concat_image else None
        return h0, img

    def _inputs(self, tokens, img):
        x = self.drop(self.embed(tokens))                      # (B, T, E)
        if img is not None:
            x = torch.cat([x, img.unsqueeze(1).expand(-1, x.shape[1], -1)], dim=-1)
        return x

    def forward(self, images, tokens):
        """Teacher forcing: logits (B, T, V) for next-token prediction after each input token."""
        h0, img = self.encode(images)
        out, _ = self.gru(self._inputs(tokens, img), h0)
        return self.out(self.drop(out))

    def step(self, tokens, h, img):
        """One decoding step for a batch of last tokens (B,): log-probabilities (B, V) and the
        new hidden state. Used by greedy decoding and beam search."""
        out, h = self.gru(self._inputs(tokens.unsqueeze(1), img), h)
        return torch.log_softmax(self.out(out[:, 0]), dim=-1), h


class DecoderBlock(nn.Module):
    """One pre-LayerNorm Transformer decoder block (GPT-2 style, plus cross-attention):

        x = x + Dropout(SelfAttention(LN(x)))        causal: position t attends to positions <= t
        x = x + Dropout(CrossAttention(LN(x), image)) queries from the caption, keys/values = image
        x = x + Dropout(FFN(LN(x)))                  FFN = Linear(d, d_ff) -> GELU -> Linear(d_ff, d)

    The attention itself is nn.MultiheadAttention (multi-head scaled dot-product attention)."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln_self, self.ln_cross, self.ln_ff = (nn.LayerNorm(d_model) for _ in range(3))
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask, need_weights=False):
        """x (B, T, d), memory (B, R, d), causal_mask (T, T) bool (True = not allowed). Returns the
        new x and, with need_weights, the cross-attention weights (B, T, R) averaged over heads."""
        h = self.ln_self(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0])
        h = self.ln_cross(x)
        a, w = self.cross_attn(h, memory, memory, need_weights=need_weights)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln_ff(x)))
        return x, w


class TransformerCaptioner(CaptionDecoding):
    """Captioner with a small Transformer decoder (trained from scratch) that cross-attends to a
    grid of CNN features (Part 4.1).

    Image: the CNN encoder without its last max pooling (and without global average pooling)
    gives a grid of feature vectors, (64 / 2^(blocks - 1))^2 = 8 x 8 = 64 regions of 256 channels
    for our CNN (other encoders provide grid(images) -> (B, R, C), grid_size and grid_dim); each is projected to d_model and gets a learned position embedding of its region.
    Caption: token embedding + learned position embedding, `n_layers` DecoderBlocks, final
    LayerNorm, and an output layer that shares its weights with the token embedding (as in
    GPT-2). Padding needs no mask: with causal attention a real token never attends to the <pad>
    tokens after the caption, and the loss ignores the <pad> targets.

    Args:
        encoder: a CNN (models.CNN) or an encoder with grid(), grid_size, grid_dim
        vocab_size, pad_id, bos_id, eos_id: as for Captioner
        d_model, n_layers, n_heads, d_ff: decoder size
        dropout: dropout on embeddings, attention weights and residual branches
        max_len: maximum caption length incl. <bos> (position embeddings)
        image_size: input image size (to compute the number of regions)
        freeze_encoder: keep the encoder fixed (no gradients, always in eval mode)
    """

    def __init__(self, encoder, vocab_size, pad_id, bos_id, eos_id, d_model=256, n_layers=3,
                 n_heads=4, d_ff=1024, dropout=0.1, max_len=64, image_size=64, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.pad_id, self.bos_id, self.eos_id = pad_id, bos_id, eos_id
        if isinstance(encoder, CNN):
            n_pools = sum(isinstance(m, nn.MaxPool2d) for m in encoder.conv)
            self.grid_size, grid_dim = image_size // 2 ** (n_pools - 1), encoder.feature_dim
        elif hasattr(encoder, "grid"):          # e.g. a pretrained encoder (Part 4.2)
            self.grid_size, grid_dim = encoder.grid_size, encoder.grid_dim
        else:
            raise TypeError("TransformerCaptioner needs a CNN or an encoder with a grid() method")
        self.img_proj = nn.Linear(grid_dim, d_model)
        self.img_pos = nn.Parameter(0.02 * torch.randn(1, self.grid_size ** 2, d_model))
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embed.weight, std=0.02)
        with torch.no_grad():
            self.embed.weight[pad_id].zero_()
        self.pos = nn.Parameter(0.02 * torch.randn(1, max_len, d_model))
        self.blocks = nn.ModuleList(DecoderBlock(d_model, n_heads, d_ff, dropout)
                                    for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.out_bias = nn.Parameter(torch.zeros(vocab_size))
        self.drop = nn.Dropout(dropout)

    def grid(self, images):
        """Feature grid (B, R, C), R = grid_size^2 regions in row-major order."""
        if not isinstance(self.encoder, CNN):
            return self.encoder.grid(images)
        g = self.encoder.conv[:-1](images)                       # without the last max pooling
        return g.flatten(2).transpose(1, 2)

    def encode(self, images):
        """(tokens generated so far (1, B, 0), image memory (B, R, d_model))."""
        if self.freeze_encoder:
            with torch.no_grad():
                g = self.grid(images)
        else:
            g = self.grid(images)
        memory = self.drop(self.img_proj(g) + self.img_pos)
        state = torch.zeros(1, images.shape[0], 0, dtype=torch.long, device=images.device)
        return state, memory

    def decode(self, tokens, memory, need_weights=False):
        """Logits (B, T, V) for the next token after each input token, and (with need_weights)
        the cross-attention weights of every block, each (B, T, R)."""
        T = tokens.shape[1]
        x = self.drop(self.embed(tokens) + self.pos[:, :T])
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=1)
        weights = []
        for block in self.blocks:
            x, w = block(x, memory, mask, need_weights)
            weights.append(w)
        logits = self.ln_f(x) @ self.embed.weight.T + self.out_bias
        return logits, weights

    def forward(self, images, tokens):
        """Teacher forcing: logits (B, T, V)."""
        return self.decode(tokens, self.encode(images)[1])[0]

    def step(self, tokens, history, memory):
        """Append `tokens` (B,) to the history (1, B, t) and return the log-probabilities of the
        next token (B, V) and the new history. Recomputes the whole prefix (no key/value cache;
        captions are short)."""
        history = torch.cat([history, tokens.view(1, -1, 1)], dim=2)
        logits = self.decode(history[0], memory)[0][:, -1]
        return torch.log_softmax(logits, dim=-1), history

    @torch.no_grad()
    def attention(self, images, max_len=25):
        """Greedy captions (B, T) and the cross-attention weights over the image regions that were
        used to generate each token: list over blocks of (B, T, grid_size, grid_size)."""
        ids = self.greedy(images, max_len=max_len)
        _, memory = self.encode(images)
        inputs = torch.cat([torch.full_like(ids[:, :1], self.bos_id), ids[:, :-1]], dim=1)
        _, weights = self.decode(inputs, memory, need_weights=True)
        g = self.grid_size
        return ids, [w.view(*w.shape[:2], g, g) for w in weights]
