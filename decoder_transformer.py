import jax
import optax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import functools

import flax

from flax import struct
from typing import Callable, Any
from flax.training import common_utils
from flax.training.train_state import TrainState


# hyperparameters
seed = 42
batch_size = 64
block_size = 256 # what is maximum context length for predictions
max_iter = 50000
learning_rate = 3e-4
n_embed = 384 # dimensional across transformer which is d_model as per paper
n_head = 6 # Number of attention heads
n_layer = 6 # Number attention blocks
dropout = 0.2
eval_interval = 500


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('/kaggle/working/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

    
@struct.dataclass
class TransformerConfig:
    """Global hyperparameters"""
    
    vocab_size: int
    # Pre-calculate the positional encoding
    pos_encode: Callable
    seed: int = 42
    d_model: int = 512
    dropout_rate: float = 0.1
    max_len: int = 2048
    num_heads: int = 8
    head_size: int = 128
    num_layers: int = 4
    block_size: int = 8
    mlp_dim: int = 2048
    deterministic: bool = False
    dtype: Any = jnp.float32
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    learning_rate: float = 3e-4
    max_iters: int = 5000
    momentum: float = 0.9
    per_device_batch_size: int = 64
    max_target_length: int = 256
    
def positional_encoding(length=2048, depth=512):
    depth = depth / 2

    positions = jnp.arange(length)[:, jnp.newaxis]  # (seq, 1)
    depths = jnp.arange(depth)[jnp.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = jnp.concatenate([jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1)

    return jnp.array(pos_encoding, dtype=jnp.float32)

class TextEmbed(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        """Applies Embedding layer to input batch of text data.

        Args:
          inputs: input data

        Returns:
          outputs a 3 dimension vector with added depth for each word (d_model)
          (batch_size, text, depth)
        """

        assert inputs.ndim == 2, (
            "Number of dimensions should be 2, but it is: %d" % inputs.ndim
        )
        input_embed = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.d_model,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )
        x = input_embed(inputs)

        return x


class PositionalEmbedding(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        length = inputs.shape[1]

        # Embed the input text batch
        x = TextEmbed(config, name="embed_input")(inputs)
        assert x.ndim == 3, "Number of dimensions should be 3, but it is: %d" % x.ndim
        x *= jnp.sqrt(config.d_model)
        x = x + config.pos_encode[jnp.newaxis, :length, :]
        return x

class SingleHead(nn.Module):
    config: TransformerConfig
    
    def setup(self):
        self.tril = jnp.tril(jnp.ones((config.block_size, config.block_size)))

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        key = nn.Dense(
            config.head_size,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init, 
            use_bias=False)(inputs)  # B, T, C
        query = nn.Dense(
            config.head_size,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False)(inputs)  # B, T, C
        value = nn.Dense(
            config.head_size, 
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False)(inputs)  # B, T, C
        
        length = inputs.shape[1]
        wei = query @ key.transpose((0, 2, 1))  # (B, T, 16) @ (B, 16,  T) -> (B, T, T)
        tril = self.tril[:length, :length]
        wei = jnp.where(tril == 0, -jnp.inf, wei)
        wei = nn.softmax(wei, axis=-1)
        wei = nn.Dropout(rate=config.dropout_rate)(
            wei, deterministic=config.deterministic)
        out = wei @ value
        return out


class MultiHeadAttention(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config
        x = jnp.concatenate(
            [SingleHead(config)(inputs) for _ in range(config.num_heads)],
            axis=-1,
        )
        x = nn.Dense(
            self.config.d_model,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,)(x)
        x = nn.Dropout(
            config.dropout_rate)(x, deterministic=config.deterministic)

        return x

class MlpBlock(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(
            inputs
        )  # d_model -> mlp_dim
        x = nn.relu(x)
        x = nn.Dense(
            config.d_model,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(
            x
        )  # mlp_dum - > d_model
        
        x = nn.Dropout(
            config.dropout_rate,
        )(x, deterministic=config.deterministic)

        return x
        
class DecoderBlock(nn.Module):
    """Transformer Decoder block: communication followed by computation"""

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        x = nn.LayerNorm()(inputs)
        sa_out = inputs + MultiHeadAttention(config)(x)

        mlp_in = nn.LayerNorm()(sa_out)
        logits = sa_out + MlpBlock(config)(mlp_in)

        return logits

class Decoder(nn.Module):
    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs):
        config = self.config

        # Input Embedding
        x = PositionalEmbedding(config, name="embedding_layer")(
            inputs
        )
        x = nn.Sequential(
            [DecoderBlock(config) for _ in range(config.num_layers)] + [nn.LayerNorm()]
        )(x)

        lm_head_out = nn.Dense(
            config.vocab_size,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init)(x)

        return lm_head_out

def compute_cross_entropy(logits, labels):
    vocab_size = logits.shape[-1]
    targets = common_utils.onehot(labels, vocab_size)
    loss = -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
    return loss.sum()

def compute_accuracy(logits, targets, weights=None):
    """Compute weighted accuracy for log probs and targets.

    Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical targets [batch, length] int array.
    weights: None or array of shape [batch, length]

    Returns:
    Tuple of scalar loss and batch normalizing factor.
    """
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    acc = jnp.equal(jnp.argmax(logits, axis=-1), targets)

    return acc.sum()

def compute_metrics(logits, labels):
    """Compute summary metrics."""
    loss = compute_cross_entropy(
        logits, labels
    )
    acc = compute_accuracy(logits, labels)
    metrics = {
        "loss": loss,
        "accuracy": acc,
    }
    metrics = jax.lax.psum(metrics, axis_name="batch")
    return metrics

def train_step(state, batch, config, dropout_rng=None):
    inputs, targets = batch
    dropout_train_key = jax.random.fold_in(key=dropout_rng, data=state.step)
    def loss_fn(params):
        logits = Decoder(config).apply(
            {"params": params},
            inputs,
            rngs={"dropout": dropout_train_key}
        )
        
        loss = compute_cross_entropy(logits, targets)
        return loss, logits
        
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.psum(grads, axis_name='batch')
    new_state = state.apply_gradients(grads=grads)
    print(logits.shape, targets.shape)
    metrics = compute_metrics(logits, targets)
    return metrics, new_state

def create_train_state(init_keys, config: TransformerConfig):
    decoder = Decoder(config)
    input_shape =  (config.per_device_batch_size, config.max_target_length)
    initial_variables = jax.jit(decoder.init)(
        init_keys['params'], 
        jnp.ones(input_shape, jnp.int32),)
    return TrainState.create(apply_fn=decoder.apply, 
                             params=initial_variables['params'], 
                             tx=optax.adam(config.learning_rate, config.momentum))

# data loading
def get_batch(key, split):
    # generate a small batch of data of inputs x and targets y
    key, subkey = jax.random.split(key)
    data = train_data if split == 'train' else val_data
    
    ix = jax.random.randint(subkey, minval=0, maxval=(len(data) - block_size), shape=(batch_size,))
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y, key

# declare config object
config = TransformerConfig(
    vocab_size=vocab_size,
    pos_encode=positional_encoding(depth=n_embed),
    block_size=block_size,
    d_model=n_embed,
    dropout_rate=0.2,
    num_heads=n_head,
    head_size=n_embed//n_head,
    num_layers=n_layer,
    mlp_dim=n_embed * 4,
    learning_rate=3e-4,
    max_iters=5000,
    deterministic=True, # Basically training is False
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=1e-6)
)

# Generate Initial Keys
rng = jax.random.key(config.seed)
rng, init_rng = jax.random.split(rng)

state = create_train_state(
    {'params': init_rng},
    config
)

config = config.replace(deterministic=False)

state = flax.jax_utils.replicate(state)
dropout_rngs = jax.random.split(rng, jax.local_device_count())

p_train_step = jax.pmap(
    functools.partial(
        train_step,
        config=config,
    ),
    axis_name="batch",
    donate_argnums=(0,),
)

print(config)

for step in range(config.max_iters):
    inputs, targets, rng = get_batch(rng, 'train') 
    # Shard data to devices and do a training step.
    with jax.profiler.StepTraceAnnotation("train", step_num=step):
        inputs = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, inputs)
        )
        targets = common_utils.shard(
            jax.tree_util.tree_map(np.asarray, targets)
        )
        metrics, state = p_train_step(state, (inputs, targets), dropout_rng=dropout_rngs)
    if step % 10 == 0:
        print(metrics)

def generate(key, config: TransformerConfig, params, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        key, sub_key = jax.random.split(key)
        idx_cond = idx[:, -config.block_size:]
        logits = Decoder(config).apply(
            {"params": params},
            idx_cond
        )
        # focus only on the last time step
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = nn.softmax(logits, axis=-1) # (B, C)
        # sample from the distribution
        idx_next = jax.random.categorical(sub_key, probs, axis=-1) # (B, 1)
        idx_next = jnp.expand_dims(idx_next, axis=1)
        # append sampled index to the running sequence
        idx = jnp.concatenate((idx, idx_next), axis=1) # (B, T+1)
    
    return idx

context = jnp.zeros((1, 1), dtype=jnp.int32)
config = config.replace(deterministic=True)
deprelicate_state = flax.jax_utils.unreplicate(state)
tokens = generate(rng, config, deprelicate_state.params, context, max_new_tokens=500)

decode(tokens.tolist()[0])
