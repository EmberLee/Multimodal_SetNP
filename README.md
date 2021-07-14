# Multi-Modal Neural Process

\<Paper title> Lucas, Eric



## 파일 및 폴더 설명
1. `config.py`  
   학습 코드 또는 eval 코드 실행 시 필요한 argument 옵션의 전체 목록.
   train.py 나 evaluate.py 실행 시 `--exp_name` 과 같이 지정 가능.
3. `data_loader.py`  
   pkl 파일로 구성된 데이터셋을 torch.utils.data.Dataset 으로 wrapping 해주는 코드. get_EMR_loader() 는 NP를 위한 데이터셋을 생성한다.  
   나중에 `utils/collate.py` 에서 배치를 구성한 데이터를 가공해준다.
5. `train.py`  
   학습 코드. 사용법은 아래 Getting started 참조.
   모델 체크포인트 파일이 `exp/exp_name/` 위치에 저장된다.
   학습 기록은 `logs/` 위치에서 확인할 수 있다.
   1 epoch 이 완료될 때마다 validation set 에 대해 성능 평가를 진행한다. `evaluate.py` 내의 `evaluate()` 함수를 사용.

7. `evaluate.py`  
   추론 코드. 사용법은 아래 Getting started 참조.
   성능 평가 지표로 **AUROC**, AUPRC 를 사용. F1 score 도 도입 예정.

9. `Makefile`  
   학습에 필요한 여러 argument 를 매번 타이핑해주는 불편을 해소하기 위해, 전체 command 를 미리 작성해놓기 위한 파일.

10. `utils/`  
11. `exp/`  
12. `logs/`  

## 시작하기(Getting started)

### 전처리 (Preprocessing steps)

1-1. Preprocess EMR data   
    1-1-1. Run reform-rawdata    
    - Generate environment files at each step. Make sure you have env_files that you want to preprocess.
    ```
    cd 01_VitalCare-Model-v2/script/reform-rawdata/
    ./generate_envs.sh
    ```   
    - Add your env_file name into the env_file_list in `run_all.sh` and run it.
    - This script will run `00_split_by_hash.sh` and `01_split_by_chid.sh` to generate splitted raw files in `01_VitalCare-Model-v2/data/reform-rawdata/`.

    ```
    ./run_all.sh
    ```   
    - (Optional) Run `python mimic_patients_info.py --dataset='MIMIC'` when you do with MIMIC or MIMIC-IV dataset.

    1-1-2. Run extract-feature   
    - Generate environment files at each step. Make sure you have env_files that you want to preprocess.
    ```
    cd 01_VitalCare-Model-v2/script/extract-feature/
    ./generate_envs.sh
    ```
    -
    1-1-3. Run diagnose-sepsis   
    - Generate environment files at each step. Make sure you have env_files that you want to preprocess.
    ```
    cd 01_VitalCare-Model-v2/script/diagnose-sepsis/
    ./generate_envs.sh
    ```
    1-1-4. Run define-profile   
    - Generate environment files at each step. Make sure you have env_files that you want to preprocess.
    ```
    cd 01_VitalCare-Model-v2/script/define-profile/
    ./generate_envs.sh
    ```
    -
    1-1-5. Run trajectory_data_w_time.py   
    ```
    cd 01_VitalCare-Model-v2/pysrc/
    python 01_trajectory_data_w_time_mimic_iv.py --dataset=MIMIC-IV --out_dir=mmmt_data_step1
    ```
    - (Optional) Run `python 01_trajectory_data_w_time.py` when the above lines do not work.   
    - It creates `MIMIC_ICU_SEPSIS_trajectory_time.pkl` and `MIMIC_ICU_SEPSIS_trajectory_time_raw.pkl` in the `../data/mmmt_data_step1/`.   



2. Preprocess for Multimodal & Multi-task setting   
    2-1. Gather and align multimodal data   
    Modify path in 771 line in `01_VitalCare-Model-v2/pysrc/02_get_multimodal_multilabel_MIMIC~.py`:
    ```python
        emr_path = os.path.join('../data/mmmt_data_step1/MIMIC/MIMIC_ICU_SEPSIS_trajectory_time.pkl')
        emr_path_raw = os.path.join('../data/mmmt_data_step1/MIMIC/MIMIC_ICU_SEPSIS_trajectory_time_raw.pkl')

    ```
    - Run `python 02_get_multimodal_multilabel_MIMIC_mmmtge.py`
    - It creates `multimodal_dataset_MIMIC_data_step2.pkl`, `multimodal_dataset_MIMIC_data_step2_raw.pkl` in `../data/mmmt_data_step2`.   

    2-2. Split into train/val/test

    ```
    cd 01_VitalCare-Model-v2/pysrc/
    python 02-1_get_split_mmmtge.py
    ```
    - It creates three part of splits (train/val/test) set such as `multimodal_MIMIC_train_raw_mmmt_ge.pkl` for normalized version and `multimodal_MIMIC_train_raw_mmmt_ge.pkl` for raw version, and its statistics `multimodal_MIMIC_onset_cases_mmmt_ge.pkl` in the `../data/mmmt_data_step2_1/`.   



## 학습 및 추론 (Train and Inference)

1. Train
   - `python train.py --exp_name='pilot' --task_idx=0`
   - or Makefile 확인 및 실행
2. Inference
   `python evaluate.py --exp_name='pilot' --task_idx=0`
   - or Makefile 확인 및 실행


## Dataset Format

데이터셋은 환자 별 전체 재실기간에 대해 imputation 된 형태로 제공될 예정. 아래 포맷은 재구성 예정인 데이터셋 형태.
The example of the dataset is as follows:
```python
{
'CHID': '100003',
'SEQ_LENGTH': 21,
'ICU_IN': datetime(2105, 5, 10, 22, 10, 55),
'ICU_OUT': datetime(2105, 5, 22, 05, 29, 32),
'AGE': 0.3,
'GENDER': 1.0,
'LABEL': ndarray(21, 2),
'EMR': ndarray(21, 29), # non-valid: [] -> [(29,), [], (29,), (29,), [], [], ... ]
'TXT': ndarray(21, (len_tokens)),
'IMG': ndarray(21, (img_path)), # [[], [], [], ..., [], (img_path), [], ..., (img_path), ... ]
'EMR_VALID_STEPS': [0, 2, 3, 10],
'TXT_VALID_STEPS': [1, 5],
'IMG_VALID_STEPS': [8, 16, 17],
'POS': ndarray (21, 32),
}
```
The format of the dataset is as follows:
- 'SEQ_LENGTH' (integer): 전체 재실 시간 (단위: 1시간)
- 'ICU_IN' (datetime object - "YYYY-mm-dd HH:MM:SS"): 입원 시간
- 'ICU_OUT' (datetime object - "YYYY-mm-dd HH:MM:SS"): 퇴원 시간
- 'AGE':
- 'GENDER':
- 'LABEL': (total timesteps, # of events)
- 'TXT': (total timesteps, # of tokens)
- 'EMR': (total timesteps, 29)
- 'TXT_VALID_STEPS':
- 'EMR_VALID_STEPS':
- 'CHID':



Dataset is encoded as python dictionary and saved as .pkl file

```python
import pickle as pkl

# NOTE: Use 'wb' mode
with open('data.pkl', 'wb') as f:
   pkl.dump(data, f)
```

## Experiments
성능 기록:
https://mechkbd.wiki/ (타 링크 임시 게재)

|AUROC|내용|설명|
|------|---|---|

|제목|내용|설명|
|:---|---:|:---:|
|내용|우측정렬|중앙정렬|
|왼쪽정렬|*기울이기*|**강조**|
|왼쪽정렬|<span style="color:red">강조3</span>|중앙정렬|

## Contacts

- Inggeol Lee: ingulbull@naver.com
