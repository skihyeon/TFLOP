import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import jsonlines
from typing import Dict, List, Tuple
from utils.util import construct_table_html_gt
from pathlib import Path

def validate_table_html(html_data: Dict) -> Tuple[bool, str]:
    """테이블 HTML의 유효성 검사
    construct_table_html_gt 결과에서 빈 행이 있는지 검사
    """
    try:
        # GT HTML 생성
        table_html = construct_table_html_gt(html_data)
        
        # 행 단위로 분리하여 검사
        rows = table_html.split('<tr>')
        for i, row in enumerate(rows[1:], 1):  # 첫 번째는 <table> 태그이므로 제외
            # </tr> 이전까지의 실제 내용만 검사
            row_content = row.split('</tr>')[0].strip()
            
            # 행에서 모든 셀의 내용 추출
            cells = row_content.split('</td>')
            has_text = False
            
            for cell in cells:
                if not cell.strip():
                    continue
                # <td>나 <td rowspan=...> 등을 제외한 실제 텍스트 확인
                cell_text = cell.split('>')[-1].strip()
                if cell_text and cell_text != '&nbsp;':
                    has_text = True
                    break
            
            if not has_text:
                return False, f"Empty row found at index {i}"
        
        return True, ""
        
    except Exception as e:
        return False, str(e)

def filter_dataset(data_dir: str):
    """데이터셋 필터링 및 저장"""
    splits = ['train', 'val']
    
    for split in splits:
        input_path = os.path.join(data_dir, f"{split}.jsonl")
        output_path = os.path.join(data_dir, f"{split}_filtered.jsonl")
        error_log_path = os.path.join(data_dir, f"{split}_filtered_errors.log")
        
        if os.path.exists(output_path):
            print(f"Filtered file already exists: {output_path}")
            continue
            
        print(f"\nProcessing {split} split...")
        filtered_data = []
        error_logs = []
        total_count = 0
        valid_count = 0
        error_counts = {}
        
        with jsonlines.open(input_path) as reader:
            for item in reader:
                total_count += 1
                
                try:
                    # HTML 구조 검증
                    is_valid, error_msg = validate_table_html(item['html'])
                    
                    if is_valid:
                        filtered_data.append(item)
                        valid_count += 1
                    else:
                        error_logs.append(f"{item['filename']}: {error_msg}")
                        error_type = error_msg.split(':')[0]
                        error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    
                    # 진행상황 출력
                    if total_count % 1000 == 0:
                        print(f"Processed {total_count} items, {valid_count} valid")
                        
                except Exception as e:
                    error_msg = f"Unexpected error - {str(e)}"
                    error_logs.append(f"{item['filename']}: {error_msg}")
                    error_counts['unexpected_error'] = error_counts.get('unexpected_error', 0) + 1
        
        # 필터링된 데이터 저장
        with jsonlines.open(output_path, 'w') as writer:
            writer.write_all(filtered_data)
        
        # 에러 로그 저장
        with open(error_log_path, 'w') as f:
            f.write('\n'.join(error_logs))
        
        # 통계 출력
        print(f"\n{split} split filtering complete:")
        print(f"Total samples: {total_count}")
        print(f"Valid samples: {valid_count}")
        print(f"Filtered out: {total_count - valid_count}")
        print("\nError statistics:")
        for error_type, count in error_counts.items():
            print(f"{error_type}: {count} ({count/total_count*100:.2f}%)")
        print(f"\nFiltered data saved to: {output_path}")
        print(f"Error logs saved to: {error_log_path}")

if __name__ == "__main__":
    data_dir = "./data/pubtabnet"
    filter_dataset(data_dir)