gpu=3

pilot_lstm:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --gpu_id=5 --exp_name='pilot_lstm' --pred_before=6 --model_type='lstm' --modality='emr' --pilot_activate --is_imputed
pilot_eval_lstm:
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --exp_name='pilot_lstm' --pred_before=6 --model_type='lstm' --modality='emr' --pilot_activate --is_imputed
pilot_bilstm:
	CUDA_VISIBLE_DEVICES=3 python train.py --exp_name='pilot_lstm' --pred_before=6 --model_type='bilstm_att' --modality='emr' --pilot_activate --is_imputed

1_soobin_txt:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=1 \
	--exp_name='1_soobin_txt' --use_embedding_file --modality='txt'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=1 \
	--exp_name='1_soobin_txt' --use_embedding_file --modality='txt'
6_soobin_txt:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=6 \
	--exp_name='6_soobin_txt' --use_embedding_file --modality='txt'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=6 \
	--exp_name='6_soobin_txt' --use_embedding_file --modality='txt'
12_soobin_txt:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=12 \
	--exp_name='12_soobin_txt' --use_embedding_file --modality='txt'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=12 \
	--exp_name='12_soobin_txt' --use_embedding_file --modality='txt'
24_soobin_txt:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=24 \
	--exp_name='24_soobin_txt' --use_embedding_file --modality='txt'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=24 \
	--exp_name='24_soobin_txt' --use_embedding_file --modality='txt'

1_soobin_both_setnp:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=1 \
	--model_type='setnp' --exp_name='1_soobin_both' --use_embedding_file --modality='both'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=1 \
	--model_type='setnp' --exp_name='1_soobin_both' --use_embedding_file --modality='both'
6_soobin_both_setnp:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=6 \
	--model_type='setnp' --exp_name='6_soobin_both' --use_embedding_file --modality='both'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=6 \
	--model_type='setnp' --exp_name='6_soobin_both' --use_embedding_file --modality='both'
12_soobin_both_setnp:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=12 \
	--model_type='setnp' --exp_name='12_soobin_both' --use_embedding_file --modality='both'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=12 \
	--model_type='setnp' --exp_name='12_soobin_both' --use_embedding_file --modality='both'
24_soobin_both_setnp:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=24 \
	--model_type='setnp' --exp_name='24_soobin_both' --use_embedding_file --modality='both'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=24 \
	--model_type='setnp' --exp_name='24_soobin_both' --use_embedding_file --modality='both'

1_bilstm:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=1 \
	--exp_name='1_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=1 \
	--exp_name='1_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
6_bilstm:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=6 \
	--exp_name='6_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=6 \
	--exp_name='6_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
12_bilstm:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=12 \
	--exp_name='12_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=12 \
	--exp_name='12_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
24_bilstm:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=24 \
	--exp_name='24_bilstm' --is_imputed --modality='emr' --model_type='bilstm'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=24 \
	--exp_name='24_bilstm' --is_imputed --modality='emr' --model_type='bilstm'

1_bilstm_att:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=1 \
	--exp_name='1_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=1 \
	--exp_name='1_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
6_bilstm_att:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=6 \
	--exp_name='6_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=6 \
	--exp_name='6_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
12_bilstm_att:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=12 \
	--exp_name='12_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=12 \
	--exp_name='12_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
24_bilstm_att:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --pred_before=24 \
	--exp_name='24_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --pred_before=24 \
	--exp_name='24_bilstm_att' --is_imputed --modality='emr' --model_type='bilstm_att'

pilot_soobin:
	CUDA_VISIBLE_DEVICES=$(gpu) python train.py --gpu_id=$(gpu) --task_idx=0 --exp_name='pilot_soobin' --pred_before=6
	# python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='pilot_soobin' --pred_before=6
pilot_eval_soobin:
	CUDA_VISIBLE_DEVICES=$(gpu) python evaluate.py --gpu_id=$(gpu) --task_idx=0 --exp_name='pilot_soobin' --pred_before=6 --pilot_activate

pilot_dupont:
	python train.py --gpu_id=5 --task_idx=0 --exp_name='pilot_dupont' --pred_before=6 --pilot_activate --model_type='dupont'
pilot_eval_dupont:
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='pilot_dupont' --pred_before=6 --pilot_activate --model_type='dupont'

## train soobin
anp_mimic_sep_pred1: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred1_exactbf2' --pred_before=1
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred1_exactbf2' --pred_before=1
anp_mimic_sep_pred6: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred6_exactbf2' --pred_before=6
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred6_exactbf2' --pred_before=6
anp_mimic_sep_pred12: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred12_exactbf2' --pred_before=12
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred12_exactbf2' --pred_before=12
anp_mimic_sep_pred24: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred24_exactbf2' --pred_before=24
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred24_exactbf2' --pred_before=24


anp_mimic_sep_pred1_within: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred1_within' --pred_before=1
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred1_within' --pred_before=1
anp_mimic_sep_pred6_within: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred6_within' --pred_before=6
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred6_within' --pred_before=6
anp_mimic_sep_pred12_within: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred12_within' --pred_before=12
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred12_within' --pred_before=12
anp_mimic_sep_pred24_within: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred24_within' --pred_before=24
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred24_within' --pred_before=24


# dupont
anp_mimic_dupont_sep_pred1: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred1' --pred_before=1 --model_type='dupont'
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred1' --pred_before=1 --model_type='dupont'
anp_mimic_dupont_sep_pred6: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred6' --pred_before=6 --model_type='dupont'
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred6' --pred_before=6 --model_type='dupont'
anp_mimic_dupont_sep_pred12: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred12' --pred_before=12 --model_type='dupont'
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred12' --pred_before=12 --model_type='dupont'
anp_mimic_dupont_sep_pred24: # task_idx of sepsis for MIMIC: 0
	python train.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred24' --pred_before=24 --model_type='dupont'
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_dupont_sep_pred24' --pred_before=24 --model_type='dupont'


## evaluate
eval_anp_mimic_sep_pred1: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred1_exactbf2' --pred_before=1
eval_anp_mimic_sep_pred6: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred6_exactbf2' --pred_before=6
eval_anp_mimic_sep_pred12: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred12_exactbf2' --pred_before=12
eval_anp_mimic_sep_pred24: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred24_exactbf2' --pred_before=24

eval_anp_mimic_sep_pred1_within: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred1_within' --pred_before=1
eval_anp_mimic_sep_pred6_within: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred6_within' --pred_before=6
eval_anp_mimic_sep_pred12_within: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=6 --task_idx=0 --exp_name='anp_mimic_sep_pred12_within' --pred_before=12
eval_anp_mimic_sep_pred24_within: # task_idx of sepsis for MIMIC: 0
	python evaluate.py --gpu_id=5 --task_idx=0 --exp_name='anp_mimic_sep_pred24_within' --pred_before=24
