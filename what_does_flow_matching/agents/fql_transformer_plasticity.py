import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from functools import partial

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value

from utils.networks_plasticity_transformer import TransformerFeatureEnsemble, ValueHead



class FQLPlasticityTransformerAgent(flax.struct.PyTreeNode):
    """Flow-Critic (FloQ) agent."""
    
    rng: Any
    network: Any
    config: Any = nonpytree_field()


    def critic_loss_before_freeze(self, batch,  grad_params, rng,):
        """Compute the FQL critic loss."""


        batch_size, observation_dim = batch['observations'].shape #(b,o)

        #sample actions
        rng, sample_rng = jax.random.split(rng, 2)
        next_actions = jnp.clip(self.sample_actions(batch['next_observations'], seed=sample_rng,), -1, 1) #(b, a)

        #compute target returns (used for TD backup)
    
        next_features = self.network.select('target_critic_features')(batch['next_observations'], next_actions) #(e,b, feature_dim)
        next_returns = self.network.select('target_critic_head')(next_features) #(e,b)

        if self.config['q_agg'] == 'min':
            next_returns = jnp.min(next_returns, axis = 0) #(b,)
        else:
            next_returns = jnp.mean(next_returns, axis = 0) #(b,)

        target_returns = batch['rewards'] + self.config['discount']*batch['masks']*next_returns #(b,)
        
        assert target_returns.shape == (batch_size,)

        features = self.network.select('critic_features')(batch['observations'], batch['actions'], params=grad_params) #(e,b, feature_dim)
        q = self.network.select('critic_head')(features, params=grad_params) #(e,b,)

        critic_loss = (q - target_returns) ** 2

        #compute current returns (used for distillation)
        return critic_loss.mean(), {
            'critic_loss': critic_loss.mean(),
            'q' : q.mean(),
        }



    def critic_loss_after_freeze(self, batch,  grad_params, rng,):
        """Compute the FQL critic loss."""


        batch_size, observation_dim = batch['observations'].shape #(b,o)

        #sample actions
        rng, sample_rng = jax.random.split(rng, 2)
        next_actions = jnp.clip(self.sample_actions(batch['next_observations'], seed=sample_rng,), -1, 1) #(b, a)

        #compute target returns (used for TD backup)
    
        next_features = self.network.select('critic_features')(batch['next_observations'], next_actions) #(e,b, feature_dim)
        next_returns = self.network.select('target_critic_head')(next_features) #(e,b)

        if self.config['q_agg'] == 'min':
            next_returns = jnp.min(next_returns, axis = 0) #(b,)
        else:
            next_returns = jnp.mean(next_returns, axis = 0) #(b,)

        target_returns = batch['rewards'] + self.config['discount']*batch['masks']*next_returns #(b,)
        
        assert target_returns.shape == (batch_size,)

        features = self.network.select('critic_features')(batch['observations'], batch['actions']) #(e,b, feature_dim)
        q = self.network.select('critic_head')(features, params=grad_params) #(e,b,)

        critic_loss = (q - target_returns) ** 2

        #compute current returns (used for distillation)
        return critic_loss.mean(), {
            'critic_loss': critic_loss.mean(),
            'q' : q.mean(),
        }

        
    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = (pred - vel) ** 2

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        #qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        
        features = self.network.select('critic_features')(batch['observations'], actor_actions,) #(e, b, feature_dim)
        qs = self.network.select('critic_head')(features,) #(e,b)

        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss.mean() + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss.mean(),
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }



    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    

    @jax.jit
    def total_loss_before_freeze(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss_before_freeze(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info



    @jax.jit
    def update_before_freeze(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss_before_freeze(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic_features')
        self.target_update(new_network, 'critic_head')

        return self.replace(network=new_network, rng=new_rng), info


    @jax.jit
    def total_loss_after_freeze(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss_after_freeze(batch, grad_params, critic_rng)

        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info



    @jax.jit
    def update_after_freeze(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss_after_freeze(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic_head')

        return self.replace(network=new_network, rng=new_rng), info



    
    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions    
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['actor_flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['actor_flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['actor_flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions


    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            ex_retruns: Example batch of returns
            config: Configuration dictionary.
        """

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)


        actor_ex_times = ex_actions[..., :1] #(batch_size, 1)
        
        batch_size = ex_actions.shape[0]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_features'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()
            encoders['critic_head'] = encoder_module()
        
        
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )

        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

 
        critic_features_def = TransformerFeatureEnsemble(

            hidden_dim=config['transformer_hidden_dim'],
            num_layers=config['transformer_num_layers'],
            num_ensembles=config['transformer_num_ensembles'],
            num_heads=config['transformer_num_heads'],
        
            n_obs_tokens=config['transformer_n_obs_tokens'],
            n_act_tokens=config['transformer_n_act_tokens'],
        
            obs_mlp_layers=config['transformer_obs_mlp_layers'],
            act_mlp_layers=config['transformer_act_mlp_layers'],
            mlp_ratio=config['transformer_mlp_ratio'],  

            postprocess_features=True,
            postprocess_mlp_dims= (config['critic_head_block_depth'] - 1)*[config['critic_head_block_width'],]
        )



        critic_head_hidden_dims = [config['critic_head_block_width'],]

        critic_head_def = ValueHead(
            hidden_dims=critic_head_hidden_dims,
            num_ensembles=config['critic_num_ensembles'],
            squeeze_output=True,                
        )    

        ex_features = jnp.ones((config['transformer_num_ensembles'], 1, config['critic_head_block_width'])) 

        network_info = dict(
            
            critic_features=(critic_features_def, (ex_observations, ex_actions,)),
            critic_head = (critic_head_def, (ex_features,)),

            target_critic_features =  (copy.deepcopy(critic_features_def), (ex_observations, ex_actions,)),
            target_critic_head = (copy.deepcopy(critic_head_def), (ex_features,)),

            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, actor_ex_times,)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions, None,)),
        
        )

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        
        params['modules_target_critic_features'] = params['modules_critic_features']
        params['modules_target_critic_head'] = params['modules_critic_head']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config),)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fqlplasticitytransformer',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.

            layer_norm=True,  # Whether to use layer normalization.     
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient (need to be tuned for each environment).
            
            actor_flow_steps=10, #Number of actor flow steps.
            
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).

            transformer_hidden_dim=256,
            transformer_num_layers=2,
            transformer_num_ensembles=2,
            transformer_num_heads=8,
            transformer_n_obs_tokens=4,
            transformer_n_act_tokens=1,
            transformer_obs_mlp_layers=1,
            transformer_act_mlp_layers=1,
            transformer_mlp_ratio=2,

            critic_head_block_width=512,
            critic_head_block_depth=4,
            critic_num_ensembles=2,

        )
    )
    return config