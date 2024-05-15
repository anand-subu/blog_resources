import argparse
import warnings
import numpy as np
import pandas as pd
import simple_icd_10_cm as cm
import os
import json

"""
@authors: antonio, Joseph Boyle

The original version of this evaluation script is sourced from the CodiEsp Shared Task evaluation scripts, provided by the authors.

The current version is a modified version of the original, provided by the authors of the "Automated clinical coding using off-the-shelf large language models" paper.

This script contains some modifications I made for parsing the predictions made by the LLMs before calculating the metrics
"""

# Load valid codes lists
def read_gs(gs_path: str, valid_codes: set) -> pd.DataFrame:
    """Read the gold-standard labels and filter to select the set of valid codes."""
    gs_data = pd.read_csv(gs_path, sep="\t", names=['clinical_case', 'code'], dtype={'clinical_case': object, 'code': object})
    gs_data = gs_data[gs_data['code'].str.lower().isin(valid_codes)]
    gs_data.code = gs_data.code.str.lower()
    return gs_data

def read_run(pred_path: str, valid_codes: set) -> pd.DataFrame:
    run_data = pd.read_csv(pred_path, sep="\t", names=['clinical_case', 'code'], dtype={'clinical_case': object, 'code': object})
    run_data.code = run_data.code.str.lower()
    run_data = run_data[run_data['code'].isin(valid_codes)]
    if run_data.shape[0] == 0:
        warnings.warn('None of the predicted codes are considered valid codes')
    return run_data

def calculate_metrics(df_gs: pd.DataFrame, df_pred: pd.DataFrame) -> tuple[float]:
    pred_per_cc = df_pred.drop_duplicates(subset=['clinical_case', "code"]).groupby("clinical_case")["code"].count()
    Pred_Pos = df_pred.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    true_per_cc = df_gs.drop_duplicates(subset=['clinical_case', "code"]).groupby("clinical_case")["code"].count()
    GS_Pos = df_gs.drop_duplicates(subset=['clinical_case', "code"]).shape[0]
    cc = set(df_gs.clinical_case.tolist())
    TP_per_cc = pd.Series(dtype=float)
    for c in cc:
        pred = set(df_pred.loc[df_pred['clinical_case'] == c, 'code'].values)
        gs = set(df_gs.loc[df_gs['clinical_case'] == c, 'code'].values)
        TP_per_cc[c] = len(pred.intersection(gs))
    TP = sum(TP_per_cc.values)
    precision_per_cc = TP_per_cc / pred_per_cc
    recall_per_cc = TP_per_cc / true_per_cc
    f1_score_per_cc = (2 * precision_per_cc * recall_per_cc) / (precision_per_cc + recall_per_cc + 1e-10)
    precision = TP / Pred_Pos
    recall = TP / GS_Pos
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    return precision_per_cc, precision, recall_per_cc, recall, f1_score_per_cc, f1_score

def calculate_metrics_simple(true: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Compute the macro-precision, macro-recall and macro-f1 scores."""
    true_positives = 0
    for case_id in set(true.clinical_case):
        true_labels = set(true.loc[true.clinical_case == case_id].code)
        pred_labels = set(pred.loc[pred.clinical_case == case_id].code)
        true_positives += len(pred_labels.intersection(true_labels))
    macro_precision = true_positives / len(pred)
    macro_recall = true_positives / len(true)
    macro_f1 = 2 / (macro_recall**-1 + macro_precision**-1)
    return dict(precision=macro_precision, recall=macro_recall, f1_score=macro_f1)

def compute_macro_averaged_scores(df_gs: pd.DataFrame, df_run: pd.DataFrame) -> tuple[float]:
    codes = set(df_gs.code)
    precisions, recalls, f1_scores = [], [], []
    for code in codes:
        true_cases = df_gs[df_gs.code == code]
        pred_cases = df_run[df_run.code == code]
        true_positive_count = len(set(pred_cases.clinical_case).intersection(set(true_cases.clinical_case)))
        precision = true_positive_count / len(pred_cases) if true_positive_count > 0 else 0
        recall = true_positive_count / len(true_cases)
        f1_score = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

def analyse_errors(true: pd.DataFrame, pred: pd.DataFrame) -> None:
    """Print some randomly sampled Errors."""
    import simple_icd_10_cm as cm
    cm.get_all_codes()
    total_errors, related_preds = 0, 0
    for case_id in ("S0004-06142006000100010-1", "S2254-28842014000300010-1", "S0004-06142006000100010-1"):
        print("##################\nCASE ID        : ", case_id)
        true_labels = set(true.loc[true.clinical_case == case_id].code)
        pred_labels = set(pred.loc[pred.clinical_case == case_id].code)
        false_positives = pred_labels.difference(true_labels)
        false_negatives = true_labels.difference(pred_labels)
        print("ASSIGNED DESC'S: ", "\n\t".join([cm.get_description(x.upper()) for x in list(pred_labels)]))
        true_positives = pred_labels.intersection(true_labels)
        print(f"True positives: {len(true_positives)} / {len(pred_labels)} \n\t", "\n\t".join([cm.get_description(x.upper()) for x in list(true_positives)]))
        print("False positives:\n\t", "\n\t".join([cm.get_description(x.upper()) for x in list(false_positives)]))
        print("False negatives:\n\t", "\n\t".join([cm.get_description(x.upper()) for x in list(false_negatives)]))
        total_errors += len(false_positives) + len(false_negatives)
        subcategory = {c[:3] for c in pred_labels}
        related_preds += len({c for c in false_positives if c[:3] in subcategory})
    print("TOTAL ERRORS ", total_errors)
    print("RELATED PREDS ", related_preds)

def parse_arguments() -> tuple[str]:
    parser = argparse.ArgumentParser(description='Process user-given parameters')
    parser.add_argument("-g", "--gs_path", required=True, dest="gs_path", help="Path to GS file")
    parser.add_argument("-p", "--pred_path", required=True, dest="pred_path", help="Path to predictions file")
    parser.add_argument("-c", "--valid_codes_path", default='codiesp/codiesp-D_codes.tsv', dest="codes_path", help="Path to valid codes TSV")
    parser.add_argument("-n", "--n_first_documents", default=None, dest="n_first_documents", type=int)
    args = parser.parse_args()
    return args.gs_path, args.pred_path, args.codes_path, args.n_first_documents

def read_valid_codes(codes_path: str) -> set:
    valid_codes = set(pd.read_csv(codes_path, sep='\t', header=None, usecols=[0])[0].tolist())
    valid_codes = set(x.lower() for x in valid_codes)
    return valid_codes

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process and evaluate medical text predictions.")
    parser.add_argument("--input_json", help="JSON file with predictions")
    parser.add_argument("--gold_standard_tsv", help="Gold standard TSV file")
    args = parser.parse_args()    
    
    code_map = json.loads(open(args.input_json).read())
    file_tsv_predictions = []

    for key, value in code_map.items():

        if value == []:
            file_tsv_predictions.append([key.replace(".txt", ""), ""])

        else:
            for code in value:
                file_tsv_predictions.append([key.replace(".txt", ""),code.lower()])

    df_pred = pd.DataFrame(file_tsv_predictions)
    df_pred.columns = ["clinical_case", "code"]
    df_pred.to_csv(args.input_json.replace(".json", ".tsv"), sep="\t", index=False)


    df_gs = pd.read_csv(args.gold_standard_tsv, sep="\t")
    df_gs.columns = ["clinical_case", "code"]
    df_gs.to_csv("gt_test.tsv", sep="\t", index=False)

    valid_codes = set([x.lower() for x in cm.get_all_codes() if cm.is_leaf(x)])
    
    gs_path = "gt_test.tsv"
    pred_path = args.input_json.replace(".json", ".tsv")
    df_gs = read_gs(gs_path, valid_codes)
    df_run = read_run(pred_path, valid_codes)
    
    precision_per_cc, precision, recall_per_cc, recall, f1_per_cc, f1_score = calculate_metrics(df_gs, df_run)
    print('\n-----------------------------------------------------')
    print('Clinical case name\t\t\tPrecision')
    print('-----------------------------------------------------')

    for index, val in precision_per_cc.items():
        print(f"{index}\t\t{round(val, 3)}")
    if any(precision_per_cc.isna()):
        warnings.warn('Some documents do not have predicted codes, document-wise Precision not computed for them.')
    print('\nMicro-average precision = {}\n'.format(round(precision, 3)))
    print('\n-----------------------------------------------------')
    print('Clinical case name\t\t\tRecall')
    print('-----------------------------------------------------')

    for index, val in recall_per_cc.items():
        print(f"{index}\t\t{round(val, 3)}")
    if any(recall_per_cc.isna()):
        warnings.warn('Some documents do not have Gold Standard codes, document-wise Recall not computed for them.')
    print('\nMicro-average recall = {}\n'.format(round(recall, 3)))
    print('\n-----------------------------------------------------')
    print('Clinical case name\t\t\tF-score')
    print('-----------------------------------------------------')

    print('\nMicro-average F-score = {}\n'.format(round(f1_score, 3)))

    macro_precision, macro_recall, macro_f1 = compute_macro_averaged_scores(df_gs, df_run)
    print('MACRO-AVERAGE STATISTICS:')
    print(f"Macro-average precision = {round(macro_precision, 3)}")
    print(f"Macro-average recall = {round(macro_recall, 3)}")
    print(f"Macro-average F-score = {round(macro_f1, 3)}")
    print(f"Number of documents with predictions {len(set(df_run.clinical_case))}")
    
    # clear the temporary gt tsv file
    os.remove("gt_test.tsv")
