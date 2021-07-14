# coding: utf-8

from baselines.BiLSTM_Attention import FEATURE_CODE


DYNAMIC_FEATURES = dict()
DYNAMIC_FEATURES['VITAL'] = [
    FEATURE_CODE.PULSE,
    FEATURE_CODE.RESP,
    FEATURE_CODE.SBP,
    FEATURE_CODE.DBP,
    FEATURE_CODE.TEMP,
    FEATURE_CODE.SpO2,
    FEATURE_CODE.GCS,
]

DYNAMIC_FEATURES['LAB'] = [
    FEATURE_CODE.PULSE,
    FEATURE_CODE.RESP,
    FEATURE_CODE.SBP,
    FEATURE_CODE.DBP,
    FEATURE_CODE.TEMP,
    FEATURE_CODE.SpO2,
    FEATURE_CODE.GCS,

    FEATURE_CODE.BILIRUBIN,
    FEATURE_CODE.LACTATE,
    FEATURE_CODE.CREATININE,
    FEATURE_CODE.PLATELET,
    FEATURE_CODE.pH,
    FEATURE_CODE.SODIUM,
    FEATURE_CODE.POTASSIUM,
    FEATURE_CODE.HEMATOCRIT,
    FEATURE_CODE.WBC,
    FEATURE_CODE.HCO3,
    FEATURE_CODE.CRP,
    FEATURE_CODE.DDIMER,
]
DYNAMIC_FEATURES['DNI'] = DYNAMIC_FEATURES['LAB'] + [FEATURE_CODE.DNI]
