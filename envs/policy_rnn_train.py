import functools
import itertools
import os
import pickle
import pprint
from datetime import date, datetime
from typing import Any, Dict, NamedTuple, Sequence
import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.serialization import from_bytes, msgpack_serialize, to_state_dict
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jax.lib import xla_bridge
from matplotlib import pyplot as plt
from softmanipulator import *
import pprint

print(xla_bridge.get_backend().platform)


def save_checkpoint(ckpt_path, state, epoch):
    with open(ckpt_path, "wb") as outfile:
        outfile.write(msgpack_serialize(to_state_dict(state)))
    artifact = wandb.Artifact(
        f'{wandb.run.name}-checkpoint', type='model'
    )
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])

def normalize_array_jax(data, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    Normalize an array of shape (16, 6) with JAX, where each row is two triplets of x, y, z coordinates.

    Parameters:
    data (jnp.ndarray): The input array of shape (16, 6).
    x_min, x_max, y_min, y_max, z_min, z_max (float): The min and max values for x, y, z coordinates.

    Returns:
    jnp.ndarray: The normalized array.
    """
    data = data.at[:, [0, 3]].set((data[:, [0, 3]] - x_min) / (x_max - x_min))  # Normalize x coordinates
    data = data.at[:, [1, 4]].set((data[:, [1, 4]] - y_min) / (y_max - y_min))  # Normalize y coordinates
    data = data.at[:, [2, 5]].set((data[:, [2, 5]] - z_min) / (z_max - z_min))  # Normalize z coordinates
    
    return data

wandb.login()


checkpoint_name = "checkpoints/Oct-31-2023 08:51.pkl"

file = open(checkpoint_name, "rb")
forward_params = pickle.load(file)

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )

    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=x[0].shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.

        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
        jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(256, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = SoftManipulator(forward_params)
    # env_params = env.test_env_params(config)
    env_params_init = env.default_params
    env_params = env_params_init.replace(
        max_steps_in_episode = config["max_steps_in_episode"],
        goal_perturbation_noise = config["goal_perturbation_noise"],
        initial_xx = config["initial_xx"],
        initial_xz = config["initial_xz"],
        observation_noise = config["observation_noise"],
        action_noise = config["action_noise"],
    )
    
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).shape[0], config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )

        # instance_RNN = ScannedRNN()
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 256)
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        # init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["NUM_ENVS"])

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                last_obs = normalize_array_jax(last_obs, env_params.max_min_x, env_params.min_max_x, 
                                               env_params.max_min_y, env_params.min_max_y, 
                                               env_params.max_min_z, env_params.min_max_z)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, done, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward 
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

def log_trajectory(info):
    trj = info
    print(trj.shape)
    env = 0
    start = trj[0,0, env, 0], trj[0,0, env, 2]
    goal = trj[0,0, env, -3], trj[0,0, env, -1]
    finish = trj[0,-1, env, 0], trj[0,-1, env, 2]
    plt.plot(trj[0,:, env, 0], trj[0,:, env, 2], "-o", label="Trajectory")
    plt.plot(start[0], start[1], "go", label="Start")
    plt.plot(goal[0], goal[1], "kx", label="Goal")
    plt.plot(finish[0], finish[1], "ro", label="Finish")
    plt.legend()
    wandb.log({"trajectory example": wandb.Image(plt)})
    plt.clf()
    plt.show()

def main():
    wandb.init(project="Soft_RL_RNN_seed")
    rng = jax.random.PRNGKey(wandb.config["seed"])
    train_jit = jax.jit(make_train(wandb.config))
    out = train_jit(rng)
    info = out["metrics"]
    return_values = info["returned_episode_returns"][info["returned_episode"]]
    for val in return_values:
        wandb.log({'return_curve': val})
    save_checkpoint("checkpoint.msgpack", out["runner_state"], wandb.config["TOTAL_TIMESTEPS"])

sweep_config = {
    'method': 'random',
    }

metric = {
    'name': 'return_curve',
    'goal': 'maximize'
    }

sweep_config['metric'] = metric
parameters_dict = {
    'seed': {
        'values': [2]
        },
    'LR': {
        'values': [1e-4, 2.5e-4, 5e-4]
        },
    'NUM_ENVS': {
        'values': [16, 32, 64]
        },
    'NUM_STEPS': {
          'values': [16,32,64]
        },
    'UPDATE_EPOCHS': {
          'values': [8,16,32]
        },
    'NUM_MINIBATCHES': {
          'values': [4,8,16]
        },
    'max_steps_in_episode': {
          'min': 60,
          'max': 100
        },
    'observation_noise': {
        'values':[0.0001,0.0005,0.001,0.005,0.01]
    },
    'action_noise': {
        'values':[0.0001,0.0005,0.001,0.005,0.01]
    },
}


parameters_dict.update({
    'TOTAL_TIMESTEPS': {
        'value': 1e6
    },
    'GAMMA': {
        'value': 0.99
    },
    'GAE_LAMBDA': {
        'value': 0.95
    },
    'CLIP_EPS': {
        'value': 0.2
    },
    'ENT_COEF': {
        'value': 0.01
    },
    'VF_COEF': {
        'value': 0.5
    },
    'MAX_GRAD_NORM': {
        'value': 0.5
    },
    'ACTIVATION': {
        'value': 'relu'
    },
    'ANNEAL_LR': {
        'value': True
    },
    'initial_xx': {
        'value': 0.03508081968179406
    },
    'initial_xz': {
        'value': 0.2099031186065612
    },
    'goal_perturbation_noise': {
        'value': 0.0
    },
})


sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep=sweep_config, project="Goal_conditioning_Paper_RNN")

wandb.agent(sweep_id, function=main, count=1000)