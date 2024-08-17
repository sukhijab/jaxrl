# Run

OpenAI Gym MuJoCo tasks

```bash
python train.py --env_name=HalfCheetah-v2 --save_dir=./tmp/
```

Experiment tracking with Weights and Biases

```bash
python train.py --env_name=HalfCheetah-v2 --save_dir=./tmp/ --track
```


DeepMind Control suite (--env-name=domain-task)

```bash
python train.py --env_name=cheetah-run --save_dir=./tmp/
```

For continuous control from pixels

```bash
MUJOCO_GL=egl python train_pixels.py --env_name=cheetah-run --save_dir=./tmp/
```

For offline RL


For sample efficient RL

```bash
python train.py --env_name=Hopper-v2 --start_training=5000 --max_steps=300000 --updates_per_step=20 --config=configs/redq_default.py --save_dir=./tmp/
```