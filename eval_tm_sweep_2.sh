# python eval_tm_sweep.py --config ./configs/eval_tm_sweep_config.yaml
for i in {0..9}
do
    python eval_tm_sweep.py --config ./configs/eval_tm_sweep_t2v_config_$i.yaml
done