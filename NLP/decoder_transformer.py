import jax
import optax
import functools
import jax.numpy as jnp
import flax.linen as nn

from flax.training import common_utils
from flax.training.train_state import TrainState


# Code reference: https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
# All the code written here is taught by Andrej Karpathy in this video: https://www.youtube.com/watch?v=kCc8FmEb1nY
# Just to make more interesting written in JAX so that I can understand more in detail


# hyperparameters
seed = 1345
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = jnp.array(encode(text))
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(key, split):
    # generate a small batch of data of inputs x and targets y
    key, subkey = jax.random.split(key)
    data = train_data if split == "train" else val_data
    ix = jax.random.randint(
        subkey, minval=0, maxval=(len(data) - block_size), shape=(batch_size,)
    )
    x = jnp.stack([data[i : i + block_size] for i in ix])
    y = jnp.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y, key


class Head(nn.Module):
    head_size: int
    """A single head for self-attention in a Transformer block.

      This module performs the core computation of a single self-attention head,
      including key, query, and value projection, attention score calculation,
      masking for causal relationships, softmax normalization, weighted value
      aggregation, and dropout for regularization.

      Args:
          head_size: The dimension of each head's output.
    """

    def setup(self):
        self.key = nn.Dense(
            features=self.head_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )
        self.query = nn.Dense(
            features=self.head_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )
        self.value = nn.Dense(
            features=self.head_size,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )
        self.tril = jnp.tril(jnp.ones((block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    @nn.compact
    def __call__(self, x, training: bool = False):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose((0, 2, 1)) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        tril = self.tril[:T, :T]
        wei = jnp.where(tril == 0, -jnp.inf, wei)  # (B, T, T)
        wei = nn.softmax(wei, axis=-1)  # (B, T, T)
        wei = self.dropout(wei, deterministic=not training)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel for a Transformer block.

    This module combines multiple `Head` instances to create a parallel
    self-attention layer. It performs separate attention calculations for each
    head and concatenates the results.

    Args:
        num_heads: The number of parallel heads to use.
        head_size: The dimension of each head's output.
    """

    num_heads: int
    head_size: int

    def setup(self):
        self.heads = [Head(self.head_size) for _ in range(self.num_heads)]
        self.proj = nn.Dense(
            n_embed,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )
        self.dropout = nn.Dropout(dropout)

    @nn.compact
    def __call__(self, x, training=False):
        out = jnp.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out), deterministic=not training)
        return out


class FeedFoward(nn.Module):
    """A simple feed-forward network for a Transformer block.

    This module consists of two linear layers with a non-linearity (ReLU) in
    between, providing a non-linear transformation of the input.

    Args:
        n_embed: The embedding dimension of the input and output.
    """

    n_embed: int

    def setup(self):
        self.net = nn.Sequential(
            [
                nn.Dense(
                    4 * self.n_embed,
                    kernel_init=nn.initializers.normal(stddev=0.02),
                    bias_init=nn.initializers.zeros,
                ),
                nn.relu,
                nn.Dense(
                    self.n_embed,
                    kernel_init=nn.initializers.normal(stddev=0.02),
                    bias_init=nn.initializers.zeros,
                ),
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @nn.compact
    def __call__(self, x, training=False):
        return self.dropout(self.net(x), deterministic=not training)


class Block(nn.Module):
    """A single Transformer block with self-attention and feed-forward layers.

    This module performs self-attention using `MultiHeadAttention`, followed by
    a residual connection, Layer normalization, and a feed-forward network
    with residual connection and Layer normalization.

    Transformer block: communication followed by computation

    Args:
        n_embed: The embedding dimension of the input and output.
        n_head: The number of parallel heads to use in the `MultiHeadAttention` layer.
    """

    n_embed: int
    n_head: int

    def setup(self):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        head_size = self.n_embed // self.n_head
        # self.sa = MultiHeadAttention(self.n_head, head_size)
        self.sa = MultiHeadAttention(self.n_head, self.n_embed)
        self.ffwd = FeedFoward(self.n_embed)
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    @nn.compact
    def __call__(self, x, training=False):
        x = x + self.sa(self.ln1(x), training)
        x = x + self.ffwd(self.ln2(x), training)
        return x


class GPTLanguageModel(nn.Module):

    def setup(self):
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embed(
            vocab_size, n_embed, embedding_init=nn.initializers.normal(stddev=0.02)
        )
        self.position_embedding_table = nn.Embed(
            block_size, n_embed, embedding_init=nn.initializers.normal(stddev=0.02)
        )
        self.block = Block(n_embed, n_head)
        self.ln_f = nn.LayerNorm()  # final layer norm
        self.lm_head = nn.Dense(
            vocab_size,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )

    @nn.compact
    def __call__(self, idx, training=False):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(jnp.arange(T))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        for _ in range(n_layer):
            x = self.block(x, training)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits


@functools.partial(jax.jit, static_argnames=["training"])
def train_step(state, batch, training, dropout_rng=None):
    """Performs a single training step for the GPT language model.

    This function takes a `TrainState`, a batch of data (inputs and targets),
    a training flag, and an optional dropout random number generator key.
    It computes the loss using a softmax cross-entropy loss function, calculates
    gradients using `jax.value_and_grad`, updates the model parameters using
    the optimizer, and returns the updated loss and `TrainState`.

    Args:
        state: The `TrainState` object containing the current model parameters and
            optimizer state.
        batch: A tuple of two JAX arrays, where the first element is the batch
            of input token indices and the second element is the batch of target
            token indices.
        training: A boolean flag indicating whether to apply dropout during training.
        dropout_rng: An optional JAX random number generator key for dropout.

    Returns:
        A tuple containing the scalar loss value for the current batch and the
        updated `TrainState` object.
    """

    inputs, targets = batch
    dropout_train_key = jax.random.fold_in(key=dropout_rng, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            inputs,
            training=training,
            rngs={"dropout": dropout_train_key},
        )

        loss = (
            optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            .sum(axis=1)
            .mean()
        )
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def generate(key, jit_apply, params, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        key, sub_key = jax.random.split(key)
        idx_cond = idx[:, -block_size:]
        logits = jit_apply({"params": params}, idx_cond)
        # focus only on the last time step
        logits = logits[:, -1, :]  # becomes (B, C)
        idx_next = jax.random.categorical(sub_key, logits, shape=(1,))
        idx_next = jnp.expand_dims(idx_next, axis=1)
        # append sampled index to the running sequence
        idx = jnp.concatenate((idx, idx_next), axis=1)  # (B, T+1)

    return idx


init_rng, rng = jax.random.split(jax.random.PRNGKey(seed))
inputs, targets, _ = get_batch(rng, "train")

model = GPTLanguageModel()
initial_variables = jax.jit(model.init)(init_rng, inputs)
jit_apply = jax.jit(model.apply, static_argnames=["training"])
state = TrainState.create(
    apply_fn=model.apply,
    params=initial_variables["params"],
    tx=optax.adamw(learning_rate),
)


for step in range(max_iters):
    inputs, targets, rng = get_batch(rng, "train")
    loss, state = train_step(state, (inputs, targets), training=True, dropout_rng=rng)
    if step % eval_interval == 0 or step == max_iters - 1:
        print(f"Train Step: {step}, Train Loss: {loss}")


context = jnp.zeros((1, 1), dtype=jnp.int32)
tokens = generate(rng, jit_apply, state.params, context, max_new_tokens=500)
print(decode(tokens.tolist()[0]))
