from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


#Flow critics        
import jax
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


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    output_dim: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class((*self.hidden_dims, self.output_dim), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]

        if actions is not None:
            inputs.append(actions)
            
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.output_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        
        else:
            v = self.value_net(inputs)

        return v






class CriticVectorField(nn.Module):
    """Critic Vector Field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    encoder: nn.Module = None
    num_ensembles:int = 1
    output_dim: int = 1

    #embed t hparams
    embed_time:bool=True
    time_embed_dim:int=64
    
    #embed z hparams
    use_prob_embed:bool = True
    q_min:float = 0.
    q_max:float = 0.
    sigma:float = 16.0
    num_bins:int = 51


    def setup(self) -> None:
        
        mlp_class = MLP
        mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0)
        self.mlp = mlp_class((*self.hidden_dims, self.output_dim), activate_final=False, layer_norm=self.layer_norm)
        
        
    @nn.compact
    def __call__(self, observations, actions, returns=None, times=None, is_encoded=False):

        """Return the vectors at the given states, actions, returns, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            returns: Returns
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """

        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations) #(batch_size, encoding_dim)

        observations = jnp.concatenate([observations, actions], axis = -1) #(batch_size, encoding_dim + action_dim)
        observations = jnp.expand_dims(observations, axis = 0) #(1, batch_size, encoding_dim + action_dim)
        observations = jnp.tile(observations, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, encoding_dim + action_dim)

        if self.use_prob_embed:   
            support = compute_support(self.q_min, self.q_max, self.num_bins) #(n_bins,) 
            bin_width = support[1] - support[0]
            num_ensembles, batch_size, _ = returns.shape
            returns = jnp.reshape(returns, (num_ensembles*batch_size, ))
            returns = to_probs(returns, support, self.sigma*bin_width) #(num_ensembles*batch_size, num_bins - 1)
            returns = jnp.reshape(returns, (self.num_ensembles, batch_size, self.num_bins - 1)) #(self.num_ensembles, batch_size, num_bins - 1)
        else:
            returns = (returns - self.q_min)/(self.q_max - self.q_min)

        if self.embed_time:
            times_embed = jnp.tile(times, [1, self.time_embed_dim]) #(batch_size, time_embedding_dim)
            times_embed = (jnp.arange(1, self.time_embed_dim + 1, 1).astype(jnp.float32)* jnp.pi * times)
            times_embed = jnp.cos(times) #(batch_size, time embedding_dim)
            times_embed = jnp.expand_dims(times_embed, axis = 0) #(1, batch_size, time_embedding_dim)
            times_embed = jnp.tile(times_embed, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, time_embedding_dim)

        else:
            times_embed = jnp.expand_dims(times, axis = 0) #(1, batch_size, time_embedding_dim)
            times_embed = jnp.tile(times_embed, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, time_embedding_dim)

        inputs = jnp.concatenate([observations, returns, times_embed], axis=-1)
        v = self.mlp(inputs) #(num_ensembles, batch_size, 1) 
        return v # (num_ensembles, batch_size, 1)




#For plasticity experiments

class ValueFeatures(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class((*self.hidden_dims,), activate_final=True, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]

        if actions is not None:
            inputs.append(actions)
            
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs)

        return v






class CriticVectorFieldFeatures(nn.Module):
    """Critic Vector Field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    encoder: nn.Module = None
    num_ensembles:int = 1

    #embed t hparams
    embed_time:bool=True
    time_embed_dim:int=64
    
    #embed z hparams
    use_prob_embed:bool = True
    q_min:float = 0.
    q_max:float = 0.
    sigma:float = 16.0
    num_bins:int = 51


    def setup(self) -> None:
        
        mlp_class = MLP
        mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0)
        self.mlp = mlp_class((*self.hidden_dims,), activate_final=True, layer_norm=self.layer_norm)
        
        
    @nn.compact
    def __call__(self, observations, actions, returns=None, times=None, is_encoded=False):

        """Return the vectors at the given states, actions, returns, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            returns: Returns
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """

        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations) #(batch_size, encoding_dim)

        observations = jnp.concatenate([observations, actions], axis = -1) #(batch_size, encoding_dim + action_dim)
        observations = jnp.expand_dims(observations, axis = 0) #(1, batch_size, encoding_dim + action_dim)
        observations = jnp.tile(observations, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, encoding_dim + action_dim)

        if self.use_prob_embed:   
            support = compute_support(self.q_min, self.q_max, self.num_bins) #(n_bins,) 
            bin_width = support[1] - support[0]
            num_ensembles, batch_size, _ = returns.shape
            returns = jnp.reshape(returns, (num_ensembles*batch_size, ))
            returns = to_probs(returns, support, self.sigma*bin_width) #(num_ensembles*batch_size, num_bins - 1)
            returns = jnp.reshape(returns, (self.num_ensembles, batch_size, self.num_bins - 1)) #(self.num_ensembles, batch_size, num_bins - 1)
        else:
            returns = (returns - self.q_min)/(self.q_max - self.q_min)

        if self.embed_time:
            #times_embed = jnp.tile(times, [1, self.time_embed_dim]) #(batch_size, time_embedding_dim)
            #times_embed = (jnp.arange(1, self.time_embed_dim + 1, 1).astype(jnp.float32)* jnp.pi * times)
            times_embed = jnp.cos(times) #(batch_size, time embedding_dim)
            times_embed = jnp.expand_dims(times_embed, axis = 0) #(1, batch_size, time_embedding_dim)
            times_embed = jnp.tile(times_embed, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, time_embedding_dim)

        else:
            times_embed = jnp.expand_dims(times, axis = 0) #(1, batch_size, time_embedding_dim)
            times_embed = jnp.tile(times_embed, [self.num_ensembles, 1, 1]) #(num_ensembles, batch_size, time_embedding_dim)

        inputs = jnp.concatenate([observations, returns, times_embed], axis=-1)
        v = self.mlp(inputs) #(num_ensembles, batch_size, 1) 
        return v # (num_ensembles, batch_size, 1)




class ValueHead(nn.Module):
    
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    output_dim: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0)

        self.value_net = mlp_class(
            (*self.hidden_dims, self.output_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, features):

        assert features.shape[0] == self.num_ensembles

        inputs = features
        
        v = self.value_net(inputs)
        return v.squeeze(-1) if self.output_dim == 1 else v






class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


#gaussian and flow actors
class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution

class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v




