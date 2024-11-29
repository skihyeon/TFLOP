import jsonlines
from tqdm import tqdm
import os

def split_jsonl_by_split(input_file, output_dir):
    """
    JSONL 파일을 split 값에 따라 분리하여 저장합니다.
    
    Args:
        input_file: 입력 JSONL 파일 경로
        output_dir: 출력 파일들이 저장될 디렉토리
    """
    # 출력 디렉토리가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 각 split 별로 writer 생성
    writers = {
        'train': jsonlines.open(os.path.join(output_dir, 'train.jsonl'), mode='w'),
        'val': jsonlines.open(os.path.join(output_dir, 'val.jsonl'), mode='w'),
        'test': jsonlines.open(os.path.join(output_dir, 'test.jsonl'), mode='w')
    }
    
    # 각 split 별 카운터 초기화
    counters = {'train': 0, 'val': 0, 'test': 0}
    
    try:
        # 전체 라인 수를 먼저 계산
        total_lines = sum(1 for _ in jsonlines.open(input_file))
        
        # 파일 읽기 및 분리
        with jsonlines.open(input_file) as reader:
            for line in tqdm(reader, total=total_lines, desc="Splitting file"):
                split = line['split']
                if split in writers:
                    writers[split].write(line)
                    counters[split] += 1
    
    finally:
        # 모든 writer 닫기
        for writer in writers.values():
            writer.close()
    
    # 결과 출력
    print("\nSplit complete!")
    for split, count in counters.items():
        print(f"{split}: {count:,} samples")

# 실행
input_path = "data/pubtabnet/PubTabNet_2.0.0.jsonl"
output_dir = "data/pubtabnet/"
split_jsonl_by_split(input_path, output_dir)