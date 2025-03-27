import csv
import requests
import json
import os
from tqdm import tqdm

def get_model_response(prompt):
    url = 'http://localhost:11434/api/chat'
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": "deepseek-r1:1.5b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "未生成内容")
        return f"错误: HTTP {response.status_code}"
    except Exception as e:
        return f"错误: 连接失败 {e}"

def read_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as csvfile:  # 去除 BOM
            reader = csv.DictReader(csvfile)
            # print("CSV 列名:", reader.fieldnames)
            rows = []
            for row in reader:
                # 跳过包含 "goal,think,prompt" 的行
                if row.get("\ufeffColumn1", row.get("Column1")) == "goal" and \
                   row["Column2"] == "think" and \
                   row["Column3"] == "prompt":
                    # print("跳过列名定义行:", row)
                    continue
                rows.append(row)
            # if rows:
            #     print("第一行有效数据:", rows[0])
            # else:
            #     print("没有有效数据行")
            return rows
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    except Exception as e:
        print(f"读取 CSV 文件时出错: {e}")
        return []

# 主逻辑
input_dir = "processed/deepseek"
input_file_path = os.path.join(input_dir, "1.5b-jailbreakprompt-test.csv")  # 你的文件名
output_dir = "processed/deepseek/responses"
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, "1.5b-jailbreakprompt-responses.csv")

# print("检查文件路径:", input_file_path)
# print("文件是否存在:", os.path.exists(input_file_path))

input_data = read_csv(input_file_path)

if not input_data:
    print("没有数据可处理，程序退出")
    exit()

csv_headers = ["goal", "think", "prompt", "response"]

with open(output_file_path, "w", newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(csv_headers)

    with tqdm(total=len(input_data), desc="处理提示词", unit="prompt") as pbar:
        for row in input_data:
            goal = row["\ufeffColumn1"] if "\ufeffColumn1" in row else row["Column1"]
            think = row["Column2"]
            prompt = row["Column3"]
            response = get_model_response(prompt)
            writer.writerow([goal, think, prompt, response])
            pbar.update(1)

print(f"模型响应已保存至 {output_file_path}")
