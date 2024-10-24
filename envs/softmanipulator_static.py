from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax, random
from functools import partial


class ForwardLSTM(nn.Module):
    features: int 
    in_carry: Tuple = None

    def setup(self):
        self.lstm_layer = nn.scan(
        nn.OptimizedLSTMCell,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})(self.features)

        self.lstm1 = self.lstm_layer
        self.dense1 = nn.Dense(3)
        self.layer_norm_1 = nn.LayerNorm()
        self.layer_norm_2 = nn.LayerNorm()

    @nn.remat    
    def __call__(self, x_batch, carried=None):
        x = x_batch
        key1, key2 = random.split(random.PRNGKey(0))
        if carried is None:
            carry = self.lstm_layer.initialize_carry(key2,x.shape)  
        else:
            carry = carried
        x = jnp.expand_dims(x, 0)
        carry_next,x = self.lstm1(carry, x)
        x = self.layer_norm_1(x)
        x = nn.relu(x)
        x = self.dense1(x)
        x = jnp.squeeze(x, 0)   

        return x,carry_next,carry

@struct.dataclass
class EnvState:
    hidden_states: Tuple #this is the carry for the recurrent model
    current_time_step: int
    current_pos: jnp.ndarray
    # TODO: add goal here


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 512
    initial_x: jnp.ndarray = jnp.array([0.0336805,0.352837,0.2021779])
    initial_pressure: float = 2.0
    lstm_features: int = 1025
    # TODO: remove goal here
    final_target: jnp.ndarray = jnp.array([0.0336805,0.352837,0.2021779]) #this is the target for the final time step
    max_pressure: float = 13 #based on the data
    max_distance: float = 0.25 #based on the data

class SoftManipulator(environment.Environment):
    """
    JAX Compatible version SoftManipulator
    """

    def __init__(self,lstm_params,distance):
        super().__init__()
        self.obs_shape = (3,)
        self.lstm_params = lstm_params
        self.forward_model = ForwardLSTM(features=1024)
        self.distance = distance

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    @property
    def test_env_params(self) -> EnvParams:
        """non-default environment parameters for Pendulum-v0."""
        return EnvParams(max_steps_in_episode = 32,
                        initial_x = jnp.array([0.0336805,0.352837,0.2021779]),
                        initial_pressure = 2.0,
                        lstm_features = 512,
                        final_target = jnp.array([0.0336805+self.distance,0.352837,0.2021779]), #this is the target for the final time step
                        max_pressure = 13.0,#based on the data
                        max_distance = 0.25, #based on the data
                    )

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.ndarray,
        params: EnvParams,
    ) -> Tuple:
        """Use Learned Model"""
        action_forward = jnp.expand_dims(action,0)
        print("actions shape", action_forward.shape)

        x,new_hidden_state,_ = self.forward_model.apply(
                                                    self.lstm_params,
                                                    action_forward,
                                                    state.hidden_states)
        # TODO: use goal defined in state instead of env_params to compute reward
        reward = -jnp.linalg.norm(x - params.final_target)
        print("reward shape ",reward.shape)
        # reward = jnp.linalg.norm(x - params.final_target)

        state = EnvState(
            hidden_states=new_hidden_state, 
            current_time_step=state.current_time_step + 1,
            current_pos=x
        )        
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment."""

        initial_x = jnp.array([params.initial_x],dtype=jnp.float32) #where does x start from
        initial_pressure = jnp.array([[params.initial_pressure,params.initial_pressure,params.initial_pressure]],dtype=jnp.float32) #where does x start from
        print("initial pressure",initial_pressure.shape)
        x,next_carry,initialized_carry = self.forward_model.apply(self.lstm_params,initial_pressure) 
        state = EnvState(hidden_states=next_carry,
                         current_time_step=0,
                         current_pos=initial_x)
        

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return the states"""

        return state.current_pos

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        #this checks if we reached the end
        done = state.current_time_step >= params.max_steps_in_episode
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "SoftManipulator-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.test_env_params
        return spaces.Box(
            low=params.initial_pressure,
            high=params.max_pressure,
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-params.max_distance, params.max_distance, shape=(3,), dtype=jnp.float32)


    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""

        # Get the maximum finite value for the desired data type (e.g., float64)
        max_value = jnp.finfo(dtype = jnp.float32).max

        return spaces.Dict(
            {
                "hidden_states": spaces.Box(-max_value, max_value, dtype=jnp.float32),
                "current_time_step": spaces.Discrete(params.max_steps_in_episode),
            }
        )