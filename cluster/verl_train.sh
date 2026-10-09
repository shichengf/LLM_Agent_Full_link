#!/usr/bin/env bash
set -euo pipefail
source configs/cluster.env
export PYTHONPATH="$COURSE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COURSE_DATA_DIR="$COURSE_ROOT/data"
export RAY_ADDRESS="$COURSE_MASTER_ADDR:$COURSE_RAY_PORT"
cd "$COURSE_ROOT"
[[ "$(git -C "$COURSE_VERL_ROOT" rev-parse HEAD)" == "$(cat configs/verl.commit)" ]] || { echo 'verl source changed'; exit 2; }
python -c 'from verl.tools.function_tool import function_tool; from src.verl_tools import lookup, calculate'
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$COURSE_ROOT/data/train.parquet" \
  data.val_files="$COURSE_ROOT/data/dev.parquet" \
  data.train_batch_size=128 data.max_prompt_length=2048 data.max_response_length=2048 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$COURSE_TRAIN_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
  actor_rollout_ref.rollout.multi_turn.function_tool_path="$COURSE_ROOT/src/verl_tools.py" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  custom_reward_function.path="$COURSE_ROOT/src/verl_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node="$COURSE_GPUS_PER_NODE" trainer.nnodes="$COURSE_NNODES" \
  trainer.project_name=agent-course trainer.experiment_name="$COURSE_RUN_ID" \
  trainer.rollout_data_dir="$COURSE_ROOT/runs/verl-$COURSE_RUN_ID/rollouts" \
  trainer.validation_data_dir="$COURSE_ROOT/runs/verl-$COURSE_RUN_ID/validation" \
  'trainer.logger=[console]' trainer.save_freq=10 trainer.test_freq=10 \
  trainer.total_training_steps=100 trainer.total_epochs=100 \
  trainer.default_local_dir="$COURSE_ROOT/runs/verl-$COURSE_RUN_ID" "$@"
