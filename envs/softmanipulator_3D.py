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
            carry = self.lstm_layer.initialize_carry(key2,x[:, 0].shape)  
        else:
            carry = carried
        carry_next,x = self.lstm1(carry, x)
        x = self.layer_norm_1(x)
        x = nn.relu(x)
        x = self.dense1(x)
        return x,carry_next

@struct.dataclass
class EnvState:
    hidden_states: Tuple #this is the carry for the recurrent model
    current_time_step: int
    current_pos: jnp.ndarray
    goal: jnp.ndarray


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 512
    initial_pose: jnp.ndarray = jnp.array([0.0336805,0.352837,0.2021779])
    initial_pressure: jnp.ndarray = jnp.array([2.0,2.0,2.0])
    lstm_features: int = 512
    final_target: jnp.ndarray = jnp.array([0.03099,0.353485,0.201845]) #this is the target for the final time step
    max_pressure: float = 13 #based on the data
    max_distance: float = 0.25 #based on the data


class SoftManipulator(environment.Environment):
    """
    JAX Compatible version SoftManipulator
    """

    def __init__(self,lstm_params):
        super().__init__()
        self.obs_shape = (1,1,3)
        self.lstm_params = lstm_params
        self.forward_model = ForwardLSTM(features=512)

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for Pendulum-v0."""
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: jnp.array,
        params: EnvParams,
    ) -> Tuple:
        """Use Learned Model"""

        x,new_hidden_state = self.forward_model.apply(
                                                    self.lstm_params,
                                                    action,
                                                    state.hidden_states)
        print(x.shape,new_hidden_state[0].shape,new_hidden_state[1].shape)
        reward = -jnp.linalg.norm(x - state.goal)
        print("reward",reward.shape)

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

        initial_pose = jnp.array([params.initial_pose],dtype=jnp.float32) #where does x start from
        initial_action = jnp.array([[[2,2,2]]])
        initial_pose, initialized_carry = self.forward_model.apply(self.lstm_params,initial_action)

        state = EnvState(hidden_states=initialized_carry,
                         current_time_step=0,
                         current_pos=initial_pose)

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return the states"""

        hidden_state_array = jnp.array(state.hidden_states)
        integer_array = jnp.tile(jnp.array([state.current_time_step]),hidden_state_array.shape)

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
            params = self.default_params
        return spaces.Box(
            low=-params.max_pressure,
            high=params.max_pressure,
            shape=(1,1,3),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.array([params.max_distance, params.max_distance, params.max_distance], dtype=jnp.float32)
        return spaces.Box(-high, high, shape=(1,1,3), dtype=jnp.float32)

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