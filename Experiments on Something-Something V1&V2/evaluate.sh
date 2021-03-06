CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  dataset=somethingv1 \
  data_dir=PATH_TO_DATASET \
  train_stage=2 \
  batch_size=32 \
  num_segments_glancer=8 \
  num_segments_focuser=12 \
  video_div=1 \
  workers=4 \
  policy_lr=0.0003 \
  epochs=50 \
  eval_freq=1 \
  random_patch=False \
  glance_size=224 \
  patch_size=144 \
  gamma=0.7 \
  with_glancer=True \
  reward=2 \
  ppo_continuous=True \
  action_std=0.25 \
  actorcritic_with_bn=True \
  evaluate=True \
  resume=PATH_TO_PRETRAINED_MODEL # load the pretrained model