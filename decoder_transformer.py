import jax
import optax
import jax.numpy as jnp
import numpy as np

from typing import Any, Callable
from flax import struct
from flax import linen as nn
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import common_utils


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
with open('input.txt', 'r', encoding='utf-8') as f:
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

class TrainState(train_state.TrainState):
  dynamic_scale: dynamic_scale_lib.DynamicScale



# data loading
def get_batch(key, split):
    # generate a small batch of data of inputs x and targets y
    key, subkey = jax.random.split(key)
    data = train_data if split == 'train' else val_data
    
    ix = jax.random.randint(subkey, minval=0, maxval=(len(data) - block_size), shape=(batch_size,))
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
    


def positional_encoding(length=2048, depth=512):
    depth = depth / 2

    positions = jnp.arange(length)[:, jnp.newaxis]  # (seq, 1)
    depths = jnp.arange(depth)[jnp.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = jnp.concatenate([jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1)

    return jnp.array(pos_encoding, dtype=jnp.float32)


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters"""

    vocab_size: int
    # Pre-calculate the positional encoding
    pos_encode: Callable
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
    momentum: int = 0.9


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

        wei = query @ key.transpose((0, 2, 1))  # (B, T, 16) @ (B, 16,  T) -> (B, T, T)
        tril = jnp.tril(jnp.ones((config.block_size, config.block_size)))
        wei = jnp.where(tril == 0, -jnp.inf, wei)
        wei = nn.softmax(wei, axis=-1)
        wei = nn.Dropout(rate=config.dropout_rate, 
                         deterministic=config.deterministic)(wei)
        out = wei @ value
        return out


class MultiHeadAttention(nn.Module):
    config: TransformerConfig
    num_layers: int = 4

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
            config.dropout_rate,
            deterministic=config.deterministic
        )(x)

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
            deterministic=config.deterministic
        )(x)

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
            bias_init=config.bias_init,)(x)

        return lm_head_out


def create_train_state(key, config: TransformerConfig):
    decoder = Decoder(config, name="decoder_transformer")
    initial_variables = jax.jit(decoder.init)(key, batched_data[0])
    adam_opt = optax.adam(config.learning_rate, config.momentum)
    return TrainState.create(apply_fn=decoder.apply, 
                             params=initial_variables['params'], 
                             dynamic_scale=None,
                             tx=adam_opt)


def compute_weighted_accuracy(logits, targets):
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)

    return loss.sum()

def compute_loss(logits, labels):
    vocab_size = logits.shape[-1]
    targets = common_utils.onehot(labels, vocab_size)
    loss = -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
    
    return loss.sum()

def compute_metrics(*, logits, labels):
  loss = compute_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


# @jax.jit
def train_step(
    config: TransformerConfig, 
    state: TrainState, 
    dropout_key,
    batch_data):
    
    
    inputs, labels = batch_data
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)
    
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params}, 
            inputs,
            rngs={'dropout': dropout_train_key})
        print(logits.shape)
        loss = compute_loss(logits, labels)
        return loss, logits

    (_, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=labels)
    return state, metrics

# @jax.jit
def eval_step(
    config: TransformerConfig, 
    state: TrainState , 
    inputs: jnp.ndarray, 
    labels: jnp.ndarray):
  logits = Decoder(config).apply({'params': state.params}, inputs)
  return compute_metrics(logits=logits, labels=labels)
    


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
    max_iters=5000
)

data_key, params_key, dropout_key, opt_key = jax.random.split(jax.random.PRNGKey(seed), 4)
batched_data = get_batch(data_key, 'train')

inti_keys = {'params': params_key, 'dropout': dropout_key}
state = create_train_state(inti_keys, config)


for iter in range(config.max_iters):
    data_key, data_subkey = jax.random.split(data_key)
    batch_data = get_batch(data_key, 'train')
    # print(inputs.dtype, labels.shape, train_state)
    state, metrics = train_step(config, state, dropout_key, batch_data)
    
    if iter % eval_interval == 0 or iter == config.max_iters - 1:
        print(f"Step: {iter}, training loss: {metrics['loss']}, training accuracy: {metrics['accuracy']}")
        eval_inputs, eval_labels = get_batch(data_subkey, 'val')
        val_metrics = eval_step(config, state, eval_inputs, eval_labels)
        print(f"Step: {iter}, validation loss: {val_metrics['loss']}, validation accuracy: {val_metrics['accuracy']}")
    
    
        
