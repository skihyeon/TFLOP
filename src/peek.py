import jsonlines
import json
def peek_jsonl(file_path, n_lines=1):
    """
    JSONL 파일의 처음 n줄을 읽어서 row_spans 키를 찾아 출력합니다.
    
    Args:
        file_path: JSONL 파일 경로
        n_lines: 읽을 줄 수 (기본값: 1)
    """
    try:
        with jsonlines.open(file_path) as reader:
            for i, line in enumerate(reader):
                # if i >= n_lines:
                    # break
                # print(f"\n=== Line {i+1} ===")
                
                # row_spans 키 찾기
                if  line['split'] == 'test':
                    print(f"yes")
                        
    except Exception as e:
        print(f"파일 읽기 오류: {e}")

# 실행
jsonl_path = "data/pubtabnet/PubTabNet_2.0.0.jsonl"
peek_jsonl(jsonl_path)