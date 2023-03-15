export JAX_ENABLE_X64=True
export CUDA_VISIBLE_DEVICES=1
jaxreaxff --init_FF Datasets/cobalt/ffield_lit             \
          --params Datasets/cobalt/params                  \
          --geo Datasets/cobalt/geo                        \
          --train_file Datasets/cobalt/trainset.in         \
          --num_e_minim_steps 200                          \
          --e_minim_LR 1e-3                                \
          --out_folder ffields                             \
          --save_opt all                                   \
          --num_trials 5                                   \
          --num_steps 20                                   \
          --init_FF_type fixed                             \
          --backend gpu
