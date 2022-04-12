CUDA_VISIBLE_DEVICES=0 python stage2.py \
  dataset=somethingv1 \
  data_dir=PATH_TO_DATASET \
  train_stage=2 \
  batch_size=64 \
  num_segments_glancer=8 \
  num_segments_focuser=12 \
  glance_size=224 \
  patch_size=144 \
  random_patch=False \
  epochs=50 \
  policy_lr=0.0003 \
  gamma=0.7 \
  with_glancer=True \
  ppo_continuous=True \
  action_std=0.25 \
  actorcritic_with_bn=True \
  workers=8 \
  eval_freq=1 \
  pretrained=PATH_TO_STAGE1_PRETRAINED_MODEL # load the stage1 pretrained model



