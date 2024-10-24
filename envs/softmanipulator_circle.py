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
    episodic_goal: jnp.array
    current_pos: jnp.array


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int = 9
    initial_xx: float = 0.03581521029788145 #mean from data (all data)
    initial_xy: float = 0.352837
    initial_xz: float = 0.20612798350944453
    initial_pressure: float = 0.0
    max_pressure: float = 13.0 #based on the data
    max_min_x: float = 0.027449 
    max_min_y: float = 0.352999
    max_min_z: float = 0.195184
    min_max_x: float = 0.038193
    min_max_y: float = 0.352999
    min_max_z: float = 0.209992
    lstm_features: int = 1024
    goal_perturbation_noise: float = 0.0001
    x_goal: float = 0.03581521029788145
    z_goal: float = 0.20612798350944453

class SoftManipulator(environment.Environment):
    """
    JAX Compatible version SoftManipulator
    """

    def __init__(self,lstm_params):
        super().__init__()
        self.internal_goal_count = 0
        self.obs_shape = (3,)
        self.lstm_params = lstm_params
        self.forward_model = ForwardLSTM(features=1024)

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
        action_forward = jnp.expand_dims(action,0)
        print("actions shape", action_forward.shape)

        x,new_hidden_state,_ = self.forward_model.apply(
                                                    self.lstm_params,
                                                    action_forward,
                                                    state.hidden_states)
        reward = -jnp.linalg.norm(x - state.episodic_goal)
        print("reward shape ",reward.shape)

        state = EnvState(
            hidden_states=new_hidden_state, 
            current_time_step=state.current_time_step + 1,
            current_pos=x,
            episodic_goal = state.episodic_goal
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
        initial_x_params = jnp.array([params.initial_xx,params.initial_xy,params.initial_xz])
        initial_x = jnp.array([initial_x_params],dtype=jnp.float32) #where does x start from
        initial_pressure = jnp.array([[params.initial_pressure,params.initial_pressure,params.initial_pressure]],dtype=jnp.float32) #where does x start from
        print("initial pressure",initial_pressure.shape)
        x,next_carry,initialized_carry = self.forward_model.apply(self.lstm_params,initial_pressure) 
        y_goal = params.initial_xy

        state = EnvState(hidden_states=next_carry,
                         current_time_step=0,
                         episodic_goal = jnp.array([params.x_goal,y_goal,params.z_goal]),
                         current_pos=initial_x)
        

        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return the states"""
        current_pos = jnp.squeeze(state.current_pos,0)
        print("77777777777777777777777777777777777777777777777777777777 ",state.current_pos,state.episodic_goal)
        return jnp.expand_dims(jnp.concatenate((current_pos,state.episodic_goal)),0)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        #this checks if we reached the end
        end_time = state.current_time_step >= params.max_steps_in_episode
        done = jnp.logical_or(end_time, jnp.linalg.norm(state.current_pos - state.episodic_goal) < 5*params.goal_perturbation_noise)
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
            low=0,
            high=params.max_pressure,
            shape=(3,),
            dtype=jnp.float32,
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(-params.max_min_x, params.min_max_x, shape=(6,), dtype=jnp.float32)


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