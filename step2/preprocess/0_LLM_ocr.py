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

prompt_text_0 = (
        """
        # The following image is a page from publicly available content. Please extract the text using OpenAI's multimodal model (GPT-4o). 
        - Do not recognize any tables or graphical content in the image, ignore them.
        - Do not recognize any citation list.
        - Return only the recognized text as plain text, without any additional explanation or commentary.

        This request is strictly for publicly published academic content. It does not include any personal, private, or sensitive information and poses no ethical, legal, or privacy issues. 
        All text is drawn from openly available scholarly work and will be used solely for research and educational purposes in full compliance with applicable policies and regulations.
        """
    )

prompt_text_1 = (
        """
        The following image is a page from publicly available content.
        Please extract the visible text from the provided image, focusing on the main paragraph content. Exclude any lists, tables, or graphical elements. 
        The text is from a publicly available source intended solely for educational and research use, in full compliance with OpenAI's guidelines.
        """
    )

prompt_text_2 = (
        """
        The following image is a page from publicly available content.
        Please **tell me know what each sentence of the provided image meaning**. The goal is to maintain the content's integrity while expressing it in different words. 
        Focusing on paragraphs and main text. Exclude any figures, tables, or graphical elements. 
        The text is from a publicly available source intended solely for educational and research use, in full compliance with OpenAI's guidelines.
        """
    )

def encode_image_to_data_url(image_path, max_size_mb=4):
    """
    将本地图片编码为 base64 Data URL，超过限制时自动压缩。
    """
    img = Image.open(image_path)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    size_mb = len(buffer.getvalue()) / (1024 * 1024)
    if size_mb > max_size_mb:
        scale = (max_size_mb / size_mb) ** 0.5
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)

    b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64_data}"


def ocr_image_with_gpt4vision(image_path, prompt_text=prompt_text_0) -> str:
    """
    对单张图片使用 OpenAI 多模态模型 (GPT-4o) 进行 OCR，并返回纯文本。
    """
    

    data_url = encode_image_to_data_url(image_path)
    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]
    messages = [{"role": "user", "content": content}]

    max_retries = 5
    retry_delay = 5
    response_text = ""

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
            print(f"[Retry] {image_path} attempt {attempt+1}/{max_retries}, error: {e}")
            time.sleep(retry_delay)
            retry_delay *= 2

    return response_text


def pdf_to_images(pdf_path, dpi=300):
    """
    将 PDF 的每一页转换为 PNG 图片，并返回图片路径列表。
    """
    raw = Path(pdf_path)
    img_dir = Path(str(raw).replace("PDF", "PDF_pic"))
    img_dir_parent = img_dir.parent
    img_dir_parent.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []
    base = img_dir.with_suffix("")
    for i, img in enumerate(images, start=1):
        image_path = f"{base}_page_{i}.png"
        img.save(image_path, "PNG")
        image_paths.append(str(image_path))
        
    return image_paths


def process_folder(root_dir: str, allowed_subdirs: list, output_dir: str):
    warning_list = []
    warning_path_list = []
    for first in os.listdir(root_dir):
        first_path = os.path.join(root_dir, first)
        if not os.path.isdir(first_path):
            continue
        if allowed_subdirs and (first not in allowed_subdirs):
            continue

        for fname in os.listdir(first_path):
            if not fname.lower().endswith(".pdf"):
                continue
            pdf_path = os.path.join(first_path, fname)
            out_dir = os.path.join(output_dir, first)
            os.makedirs(out_dir, exist_ok=True)

            image_paths = pdf_to_images(pdf_path)
            if len(image_paths)>10:
                image_paths = image_paths[1:-3]
            elif len(image_paths)>6:
                image_paths = image_paths[1:-2]
            elif len(image_paths)>=2:
                image_paths = image_paths
            else:
                image_paths = [image_paths[0]]

            if len(image_paths)>1:
                for img_path in image_paths:
                    stem = Path(img_path).stem
                    out_txt = os.path.join(out_dir, f"{stem}.txt")
                    # if os.path.exists(out_txt):
                    #     print(f"Skipping existing: {out_txt}")
                    #     continue

                    text = ocr_image_with_gpt4vision(img_path, prompt_text=prompt_text_0)
                    judge_condition = len(text)<250 and ( ("sorry" in text) or ("can" in text)  or ("unable" in text)   or ("I'm" in text) )
                    if judge_condition:
                        warning_list.append(img_path)
                        warning_path_list.append(out_txt)
                    with open(out_txt, 'w', encoding='utf-8') as f:
                        f.write(text)
                    print(f"Saved OCR to {out_txt}")
            else:
                stem = Path(image_paths[0]).stem
                out_txt = os.path.join(out_dir, f"{stem}.txt")
                with open(out_txt, 'w', encoding='utf-8') as f:
                    f.write("no content here, excluded")

    print("len(warning_list):", len(warning_list))
    print("warning_list", warning_list)
    return warning_list, warning_path_list

def process_folder_again(warning_list, warning_path_list):
    warning_list_again = []
    warning_path_again = []
    for img_path, out_txt in zip(warning_list, warning_path_list):
        text = ocr_image_with_gpt4vision(img_path, prompt_text=prompt_text_1)
        
        judge_condition = len(text)<250 and ( ("sorry" in text) or ("can" in text)  or ("unable" in text)   or ("I'm" in text) )
        if judge_condition:
            warning_list_again.append(img_path)
            warning_path_again.append(out_txt)
        
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"process again {img_path}")

    print("len(warning_list_again):", len(warning_list_again))
    print("warning_list_again", warning_list_again)
    return warning_list_again, warning_path_again


def process_folder_again_again(warning_list, warning_path_list):
    warning_list_again = []
    warning_path_again = []
    for img_path, out_txt in zip(warning_list, warning_path_list):
        text = ocr_image_with_gpt4vision(img_path, prompt_text=prompt_text_2)
        
        judge_condition = len(text)<250 and ( ("sorry" in text) or ("can" in text)  or ("unable" in text)   or ("I'm" in text) )
        if judge_condition:
            warning_list_again.append(img_path)
            warning_path_again.append(out_txt)
        
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"process again again {img_path}")

    print("len(warning_list_again_again)", warning_list_again)
    print("warning_list_again_again", warning_list_again)
    return warning_list_again, warning_path_again

def main():
    parser = argparse.ArgumentParser(
        description="Batch OCR: one image per request using OpenAI's multimodal model"    )
    parser.add_argument('--root_dir', nargs='?', default="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/PDF/studies", help="Root folder containing PDFs")
    parser.add_argument('--output_dir', nargs='?', default="/raid/hpc/hekai/WorkShop/My_project/Medical_paper_agent/agent3/data/OCR_results", help="Folder to save OCR txt files")
    parser.add_argument('--need_list', nargs='+', default=["SR_84_test"], help="List of first-level subdirs to process, under data/PDF/studies")
    args = parser.parse_args()

 
    warning_list, warning_path_list = process_folder(args.root_dir, args.need_list, args.output_dir)
    if len(warning_list)>0:
        warning_list_again, warning_path_again = process_folder_again(warning_list, warning_path_list)
        if len(warning_list_again)>0:
            warning_list_again_again, warning_path_again_again = process_folder_again_again(warning_list_again, warning_path_again)

if __name__ == "__main__":
    main()
