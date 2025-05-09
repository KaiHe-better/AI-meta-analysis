import os
import io
import time
import base64
import argparse
from PIL import Image
from mimetypes import guess_type
from pdf2image import convert_from_path
import openai
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)



def extract_with_gpt4vision(ID, out_txt) -> str:
    prompt_text = (
        f"""
        # The following content is from an academic paper processed by OCR. Your task is to extract the complete Method or Methodology section (including sections with other titles that semantically belong to the Method section).
        - Please be aware that due to double-column formatting and OCR errors, the text sequence may be disrupted. Headers, footers, or unrelated content might appear inside or between sections.
        - Carefully reconstruct the correct logical order based on semantic coherence. Ignore and remove any inserted or misplaced content that does not belong to the Method section.
        - Extract and return only the full, continuous text of the Method section as plain text, with no additional explanations, formatting, or commentary.
        - If the paper is of the "Comments", "Editorial" or other similar types that does not contain a Method section, please return "None".

        The content need be extracted:
        {out_txt}
        """
    )
    content = [
        {"type": "text", "text": prompt_text},
    ]
    messages = [{"role": "user", "content": content}]

    max_retries = 5
    retry_delay = 5
    response_text = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                timeout=60,
                temperature=0
            )
            response_text = response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"[Retry] {ID} attempt {attempt+1}/{max_retries}, error: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2

    return response_text





def process_folder(root_dir: str, allowed_subdirs: list, output_dir: str):
    warning_list = []
    for first in os.listdir(root_dir):
        first_path = os.path.join(root_dir, first)
        if not os.path.isdir(first_path):
            continue
        if allowed_subdirs and first not in allowed_subdirs:
            continue


        res_dic = {}
        for fname in os.listdir(first_path):
            if not fname.lower().endswith(".txt"):
                continue

            parts = fname.split("_page_")
            if len(parts) != 2:
                raise Exception("error") 

            fname_ID = parts[0]
            page_num = int(parts[1].split(".")[0])  

            txt_path = os.path.join(first_path, fname)
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()

            res_dic.setdefault(fname_ID, []).append((page_num, text))  

        for fname_ID in res_dic:
            res_dic[fname_ID].sort(key=lambda x: x[0])  
            combined_text = '\n'.join(text for _, text in res_dic[fname_ID]) 
            res_dic[fname_ID] = combined_text
        
        out_dir = os.path.join(output_dir, first)
        os.makedirs(out_dir, exist_ok=True)

        for ID, content in res_dic.items():
            out_file_path = os.path.join(out_dir, ID+".txt")
            text = extract_with_gpt4vision(ID, content)
            with open(out_file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            print(f"Saved extraction to {out_file_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR: one image per request using OpenAI's multimodal model"    )
    parser.add_argument('--root_dir', nargs='?', default="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/OCR_results", help="Root folder containing PDFs")
    parser.add_argument('--output_dir', nargs='?', default="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/OCR_extracted", help="Folder to save OCR txt files")
    parser.add_argument('--need_list', nargs='+', default=["SR_344"], help="List of first-level subdirs to process")
    args = parser.parse_args()

    process_folder(args.root_dir, args.need_list, args.output_dir)

if __name__ == "__main__":
    main()
