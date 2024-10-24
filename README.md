<h1 align="center"> <em>SoRoLEX</em>: Soft Robotics using Learned Environment in JAX </h1>

<p align="center">
    <a href="https://ieeexplore.ieee.org/abstract/document/10522003">
        <img src="https://img.shields.io/badge/IEEE-10522003-00629B.svg" /></a>
    <a href= "https://github.com/uljad/SoRoLEX/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>

</p>

These are some of the tools used on *Towards Reinforcement Learning Controllers for Soft Robots Using Learned Environments*, IEEE Robosoft 2024.

<em>SoRoLEX</em> is a framework for learning soft robot controllers using reinforcement learning with learned environments. Our implementation is fully built in <a href="https://github.com/google/jax">JAX</a>.

[**Setup**](#setup) | [**Usage**](#usage) | [**Citation**](#citation)


<h4 align="center"> <strong>⚠️...Under Construction...⚠️</strong> 😢</h4>

# 🛠️ Setup

## 📋 Requirements

Requirements can be found in the `lstm/setup/` directory. You can view them [here](lstm/setup/).

<details>
<summary><strong>Important Dependencies</strong></summary>

```
Base Requirements:
- distrax==0.1.3
- flax==0.7.2
- gymnax==0.0.6
- pre-commit==3.3.3
- wandb==0.15.8

CPU-specific:
- jax==0.4.13
- jaxlib==0.4.13

GPU-specific:
- jax[cuda12_pip]==0.4.13
```

Please ensure you have the correct versions installed for your system (CPU or GPU).
</details>

## 🐳 Running Via Docker

1. **Build the Docker container** with the provided script:
```
cd setup/docker && ./build.sh
```
2. **Add your [WandB key](https://wandb.ai/authorize)** to the `lstm/setup/docker` folder:

```
echo <wandb_key> > setup/docker/wandb_key

```

👼 just add a `wandb_key` file without any extensions containing the key from the link above. the `.gitignore` is set up to ignore it and ensure the privacy of your key and your data. 


### 🎮 Usage

Place the running script in the relevant directory

```
./run_docker.sh <gpu_id> python3 train.py <arguments>
```
For example, to train the agent on the learned environment using GPU 3, run:
```
cd envs
./run_docker.sh 3 python3 train.py
```
To train the lstm on GPU 5

```
cd lstm
./run_docker.sh 5 python3 train.py
```
**This repo follows the [jax-rl template](https://github.com/EmptyJackson/jax-rl-template/blob/main/README.md?plain=1)**. You can refer to that for more details

## Acknowledgement

This work would have been possible without the following:

🚀 **[Jax Ecosystem](https://github.com/jax-ml/jax_)** ⚡ 

💪 **[Gymnax](https://github.com/RobertTLange/gymnax)** 🏋️‍♂️

🌟  **[PureJaxRL](https://github.com/luchris429/purejaxrl/tree/main)** 🌟


## 📚 Citation
If you use any of these tools, it would be really nice if you could please cite 😍 :

```
@INPROCEEDINGS{10522003,
  author={Berdica, Uljad and Jackson, Matthew and Veronese, Niccolò Enrico and Foerster, Jakob and Maiolino Perla},
  booktitle={2024 IEEE 7th International Conference on Soft Robotics (RoboSoft)}, 
  title={Reinforcement Learning Controllers for Soft Robots Using Learned Environments}, 
  year={2024},
  pages={933-939},
  doi={10.1109/RoboSoft60065.2024.10522003},
}
```