import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from step2.step2_agent_openai_multi import openai_main
from step2.step2_agent_qwq_multi import qwq_main
from step2.step2_agent_deepseek_multi import deepseek_main


def extract_pmids_from_folder(folder_path: str) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    filename_pattern = re.compile(r'(\d+)_')

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.xls', '.xlsx')):
            continue

        m = filename_pattern.search(fname)
        if not m:
            continue

        key = int(m.group(1))
        fullpath = os.path.join(folder_path, fname)

        try:
            df = pd.read_excel(fullpath, sheet_name='Exception')
        except ValueError:
            continue

        if 'PMID' not in df.columns:
            raise Exception(f"[WARN] 文件 {fname} 的 'Exception' 表中未找到 'PMID' 列")

        pmid_list = df['PMID'].dropna().astype(int).unique().tolist()
        result[key] = pmid_list

    return result


def run_selected_agent(func, key: int, pmids: list[int]):
    func({key: pmids})
    print(f"Finished {func.__name__} for key {key} \n")


def return_dic(LLM):
    if LLM == "openai":
        folder = 'step2/res/openai_results_multi'
        main_func = openai_main
    elif LLM == "deepseek":
        folder = 'step2/res/deepseek_results_multi'
        main_func = deepseek_main
    elif LLM == "qwq":
        folder = "step2/res/qwq_results_multi"
        main_func = qwq_main
    else:
        raise ValueError("Unknown LLM type")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(base_dir, folder)
    pmid_dict = extract_pmids_from_folder(folder)

    total_num = sum(len(v) for v in pmid_dict.values())
    for k, v in pmid_dict.items():
        print(f"{k}: {v}")

    return pmid_dict, total_num, main_func

if __name__ == '__main__':

    print("len(sys.argv)", len(sys.argv))
    if len(sys.argv) < 2 :
        LLM = "openai"   
        # LLM = "deepseek"  
        # LLM = "qwq"  

    else:
        LLM = sys.argv[1]

    print(sys.argv)

    pmid_dict, total_num, main_func = return_dic(LLM)
    
    print(f"\n\nTotal Exception PMIDs: {total_num}")

    with ThreadPoolExecutor(max_workers=5) as executor:
        for key, pmids in pmid_dict.items():
            executor.submit(run_selected_agent, main_func, key, pmids)
    
    print("All finished ! \n ")

    pmid_dict, total_num, main_func = return_dic(LLM)
    print(f"\n\nRemain Exception PMIDs: {total_num}")