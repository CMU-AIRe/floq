import jax
from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp




def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x



def to_probs(
    target: jax.Array,  # (batch_size,)
    support: jax.Array,  # (num_bins,)
    sigma: float,
  ) -> jax.Array:

    assert target.ndim == 1

    cdf_evals = jax.scipy.special.erf((support - target[:, None]) / (jnp.sqrt(2.0) * sigma))
    z = cdf_evals[..., -1:] - cdf_evals[..., :1]
    bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return bin_probs / (z)  # (batch_size, num_bins)

def compute_support(
    q_min,
    q_max,
    num_bins,
):
    return q_min + jnp.arange(num_bins)*(q_max - q_min)/(num_bins - 1)



class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feedforward layers."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: int = 2

    @nn.compact
    def __call__(self, x,):
        # Pre-LayerNorm architecture (more stable)
        # Multi-head self-attention with pre-norm
        attn_input = nn.LayerNorm()(x)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f'hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads}).'
            )
        head_dim = self.hidden_dim // self.num_heads

        query = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name='query'
        )(attn_input)
        key = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name='key'
        )(attn_input)
        value = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            name='value'
        )(attn_input)

        scale = jnp.sqrt(head_dim).astype(attn_input.dtype)
        logits = jnp.einsum('...qhd,...khd->...hqk', query, key) / scale
        attention_weights = nn.softmax(logits, axis=-1)

        attn_output = jnp.einsum('...hqk,...khd->...qhd', attention_weights, value)
        attn_output = attn_output.reshape(*attn_input.shape[:-1], self.hidden_dim)
        attn_output = nn.Dense(self.hidden_dim, name='out')(attn_output)

        x = x + attn_output

        # Feed-forward network with pre-norm
        mlp_input = nn.LayerNorm()(x)
        mlp_dim = self.hidden_dim * self.mlp_ratio
        mlp_out = nn.Dense(mlp_dim)(mlp_input)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.LayerNorm()(mlp_out)
        mlp_out = nn.Dense(self.hidden_dim)(mlp_out)


        x = x + mlp_out
        return x



class TransformerFeatures(nn.Module):
    """Transformer-based feature extractor with grouped obs/action tokens."""
    
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 2
    num_ensembles: int = 2

    
    n_obs_tokens: int = 1
    n_act_tokens: int = 1

    obs_mlp_layers: int = 1
    act_mlp_layers: int = 1

    postprocess_features: bool = False
    postprocess_mlp_dims: Optional[Sequence[int]] = None

    def setup(self):
        if self.postprocess_features:
            assert self.postprocess_mlp_dims is not None, 'postprocess_mlp_dims must be provided if postprocess_features is True'

            self.postprocess_mlp = MLP(
                hidden_dims=self.postprocess_mlp_dims,
                activate_final=True,
                layer_norm=True,
            )

        else:
            raise
            #self.postprocess_mlp = Identity()

    @nn.compact
    def __call__(self, observations, actions,):
        
        """
        Args:
            observations: (..., obs_dim)
            actions: (..., action_dim)
            returns: (..., 1)
            cls_output: (..., hidden_dim)
        """
        
        batch_shape = observations.shape[:-1]
        
        obs_feat = observations

        for i in range(self.obs_mlp_layers - 1):
            obs_feat = nn.Dense(self.hidden_dim, name=f'obs_mlp_{i}')(obs_feat)
            obs_feat = nn.gelu(obs_feat)
            obs_feat = nn.LayerNorm(name=f'obs_mlp_ln_{i}')(obs_feat)

        obs_feat = nn.Dense(
            self.n_obs_tokens * self.hidden_dim,
            name='obs_token_projection'
        )(obs_feat)

        obs_tokens = obs_feat.reshape(
            batch_shape + (self.n_obs_tokens, self.hidden_dim)
        )

        act_feat = actions
        
        for i in range(self.act_mlp_layers - 1):
            act_feat = nn.Dense(self.hidden_dim, name=f'action_mlp_{i}')(act_feat)
            act_feat = nn.gelu(act_feat)
            act_feat = nn.LayerNorm(name=f'action_mlp_ln_{i}')(act_feat)

        action_token = nn.Dense(
            self.n_act_tokens * self.hidden_dim,
            name='action_token_projection'
        )(act_feat)

        action_token = action_token.reshape(
            batch_shape + (self.n_act_tokens, self.hidden_dim)
        )

        tokens = jnp.concatenate([obs_tokens, action_token,], axis=-2)
        seq_len = self.n_obs_tokens + self.n_act_tokens 
        


        for i in range(self.num_layers):
        
            tokens = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f'transformer_block_{i}',
            )(tokens)

        tokens = nn.LayerNorm(name='final_ln')(tokens)

        cls_output = tokens[..., -1, :]

        if self.postprocess_features:
            cls_output = self.postprocess_mlp(cls_output)
            
        return cls_output

class TransformerFeatureEnsemble(nn.Module):
    """
    Ensemble wrapper for TransformerFeatures.

    Produces independent TransformerFeatures with separate parameters
    but shared inputs.

    Output shape:
        (num_ensembles, batch, hidden_dim)
    """

    num_ensembles: int = 2

    # pass-through TransformerFeatures config
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 8
    mlp_ratio: int = 2

    n_obs_tokens: int = 1
    n_act_tokens: int = 1

    obs_mlp_layers: int = 1
    act_mlp_layers: int = 1

    postprocess_features: bool = False
    postprocess_mlp_dims: Optional[Sequence[int]] = None

 


    def setup(self):
        feature_cls = TransformerFeatures

        # vmap TransformerFeatures across ensemble axis
        if self.num_ensembles > 1:
            
            feature_cls = ensemblize(
                feature_cls,
                num_qs=self.num_ensembles,
                in_axes=0,
                out_axes=0,
            )

        self.feature_net = feature_cls(
            
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            
            n_obs_tokens=self.n_obs_tokens,
            n_act_tokens=self.n_act_tokens,
            
            obs_mlp_layers=self.obs_mlp_layers,
            act_mlp_layers=self.act_mlp_layers,

            postprocess_features=self.postprocess_features,
            postprocess_mlp_dims=self.postprocess_mlp_dims,

        )

    @nn.compact
    def __call__(self, observations, actions,):

        assert observations.ndim == actions.ndim == 2 #(b, o), (b, a)

        observations = jnp.expand_dims(observations, axis = 0) #(1, b, o)
        actions = jnp.expand_dims(actions, axis = 0) #(1, b, a)

        observations = jnp.tile(observations, [self.num_ensembles, 1, 1]) #(e, b, o)
        actions = jnp.tile(actions, [self.num_ensembles, 1, 1]) #(e, b, a)

        return self.feature_net(observations, actions)


class ValueHead(nn.Module):
    
    """Value function head."""
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    output_dim: int = 1
    squeeze_output: bool = True

    use_prob_embed:bool = True
    q_min:float = -100.
    q_max:float = 0.
    sigma:float = 16.0 
    num_bins:int = 51

    def setup(self):
        mlp_class = MLP
        
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0,)

        value_net = mlp_class((*self.hidden_dims, self.output_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, features, returns=None, times=None):
        
        """Return values.

        Args:
            features: Input features.
        """
        

        assert features.ndim == 3 and features.shape[0] == self.num_ensembles, f'Expected features shape (num_ensembles, batch_size, hidden_dim), got {features.shape}'
        assert returns is None or returns.shape == features.shape[:2] + (1,), f'Expected returns shape (num_ensembles, batch_size, 1), got {returns.shape}'
        assert times is None or times.shape == (features.shape[1], 1), f'Expected times shape (batch_size, 1), got {times.shape}'
        
        if times is not None:
            times = jnp.expand_dims(times, axis=0) #(1, batch_size, 1)
            times = jnp.tile(times, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, 1)
            times = jnp.cos(times) #(num_ensembles, batch_size, 1)
        
        if self.use_prob_embed and returns is not None:
            support = compute_support(self.q_min, self.q_max, self.num_bins) #(num_bins,)
            bin_width = support[1] - support[0]
            num_ensembles, batch_size, _ = returns.shape
            returns = jnp.reshape(returns, (num_ensembles*batch_size,))
            returns = to_probs(returns, support, self.sigma*bin_width) #(num_ensembles*batch_size, num_bins - 1)
            returns = jnp.reshape(returns, (self.num_ensembles, batch_size, self.num_bins - 1)) #(num_ensembles, batch_size, num_bins - 1)

        if returns is not None and times is not None:
            inputs = jnp.concatenate([features, returns, times], axis = -1) #(num_ensembles, batch_size, hidden_dim + num_bins)
        else:
            inputs = features

        output = self.value_net(inputs)
        
        if self.squeeze_output:
            output = output.squeeze(-1) #(num_ensembles, batch_size,)
        
        return output


