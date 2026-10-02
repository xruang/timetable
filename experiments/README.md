# Experiment configurations

These JSON files are the single source of truth for new runs.

- `lower_sac_action_consistent_dt_0_2.json`: recommended fresh regression experiment
  after the 2026-09-17 implementation fixes. Replay, actor and target critics all
  use raw proposals; terminal protection is stateless and event-triggered. Same
  reward weights/gamma/alpha settings as the preceding conditioned experiment.
  Device is auto (CUDA when available). Old continuous checkpoints are rejected.
  `within5s` counts safe arrivals within 5 seconds; `strict` retains 0.5 seconds.
  Neither a completed late run nor a forced-zero endpoint counts as success.

- `lower_sac_smoke.json`: two-episode pipeline check only.
- `lower_dqn_baseline.json`: lower-only DQN baseline.
- `lower_ddpg_baseline.json`: lower-only DDPG baseline.
- `lower_sac_baseline.json`: legacy lower-only SAC baseline retaining the old
  85% speed-limit ratio for reproducibility.
- `lower_sac_speed_limit_1_0.json`: lower-only SAC using 100% of the
  configured line speed limit; the safety action shield remains enabled.
- `lower_sac_schedule_state_gamma_0_995.json`: SAC diagnostic run using the
  true speed limit, explicit signed schedule deviation and gamma 0.995.
- `lower_sac_shared_context_replay_1m.json`: conditioned shared SAC using
  direction/section context, 5x schedule-deviation scaling and a one-million-
  transition replay buffer.
- `lower_sac_shared_context_dt_0_1.json`: the same conditioned shared SAC with
  down gradients reversed from the up profile, a balanced grid of 2 directions
  x 19 sections x 9 mass anchors, discount-consistent schedule shaping, gamma
  0.9998, event-triggered minimal safety projection and a 0.2-second dynamics
  step. Every 342 episodes visits every grid combination exactly once, in a
  newly shuffled order; the configured 5130 episodes are 15 complete cycles.
  The config name retains `dt_0_1` for launcher compatibility; the actual step is 0.2 s.
  SAC alpha is fixed at 0.02 for the first 1026 episodes (three complete grid
  cycles), then auto-tuned with learning rate 3e-5 and a 0.01 lower bound.

The three baselines use the same seed, train count, target times, arrival
tolerance and episode budget. Behavior cloning and random-mass training are
disabled in the baseline comparison so they do not become confounding factors.
Add them later as separate ablation configs.

Do not put generated output in this directory. The launcher always writes to
`results/<experiment_name>/<timestamp>_seed<seed>/`.
