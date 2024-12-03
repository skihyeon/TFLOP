from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer
import re

class OTSLTokenizer:
    """
    OTSL (Optimized Table Structure Language) Tokenizer
    
    논문 4.1 Language Definition 참조:
    - 5개의 기본 토큰으로 테이블 구조 표현
    - HTML과 1:1 매핑 보장
    - 문법 규칙으로 유효성 검증 가능
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 1376,  # 논문 4.2
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]"
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.special_tokens = {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token
        }
        
        # OTSL의 5가지 기본 토큰 (논문 4.1 Language Definition 참조)
        self.otsl_tags = {
            "C": "cell",           # 새로운 테이블 셀
            "L": "left-looking",   # 왼쪽 이웃 셀과 병합
            "U": "up-looking",     # 위쪽 이웃 셀과 병합
            "X": "cross",          # 왼쪽과 위쪽 이웃 셀 모두와 병합
            "NL": "new-line"       # 다음 행으로 이동
        }
        
        # HTML과의 매핑 정보
        self.html_mapping = {
            "C": "<td>",           # 기본 셀
            "L": '<td colspan="2">', # 가로 병합
            "U": '<td rowspan="2">', # 세로 병합
            "X": '<td colspan="2" rowspan="2">', # 2D 병합
            "NL": "</tr><tr>"      # 새 행
        }
        
        self._build_vocab()
    
    def _build_vocab(self):
        """Vocabulary 구축"""
        # 1. Special tokens
        self.token2id = {
            token: idx 
            for idx, token in enumerate(self.special_tokens.values())
        }
        
        # 2. OTSL tags
        for tag in self.otsl_tags.keys():
            if tag not in self.token2id:
                self.token2id[tag] = len(self.token2id)
        
        # 3. ID to token mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        # Special token IDs
        self.pad_token_id = self.token2id[self.special_tokens["pad_token"]]
        self.unk_token_id = self.token2id[self.special_tokens["unk_token"]]
        self.bos_token_id = self.token2id[self.special_tokens["bos_token"]]
        self.eos_token_id = self.token2id[self.special_tokens["eos_token"]]
    
    def validate_syntax(self, tokens: List[str]) -> bool:
        """OTSL 문법 규칙 검증 (논문 4.2 Language Syntax 참조)"""
        rows = []
        current_row = []
        
        for token in tokens:
            if token == "NL":
                if not current_row:
                    print("Validation failed: Empty row found")  # 디버깅
                    return False  # 빈 행은 허용되지 않음
                rows.append(current_row)
                current_row = []
                continue
            current_row.append(token)
        
        if current_row:  # 마지막 행 처리
            rows.append(current_row)
        
        # 규칙 6: 직사각형 규칙 - 모든 행의 길이가 동일해야 함
        if not rows:
            # print("Validation failed: No rows found")  # 디버깅
            return False
        if not all(len(row) == len(rows[0]) for row in rows):
            # print(f"Validation failed: Inconsistent row lengths - {[len(row) for row in rows]}")  # 디버깅
            return False
                
        for i, row in enumerate(rows):
            for j, token in enumerate(row):
                # 규칙 4: 첫 번째 행은 L과 C만 허용
                if i == 0 and token not in ["C", "L"]:
                    # print(f"Validation failed: Invalid token '{token}' in first row")  # 디버깅
                    # print("Grid:")
                    # for r in rows:
                    #     print(" ".join(r))
                    return False
                        
                # 규칙 5: 첫 번째 열은 U와 C만 허용 
                if j == 0 and token not in ["C", "U"]:
                    # print(f"Validation failed: Invalid token '{token}' in first column")  # 디버깅
                    # print("Grid:")
                    # for r in rows:
                    #     print(" ".join(r))
                    return False
                        
                # 규칙 1: L 셀의 왼쪽 이웃은 L 또는 C여야 함
                if token == "L" and j > 0 and row[j-1] not in ["C", "L"]:
                    # print(f"Validation failed: Invalid left neighbor '{row[j-1]}' for L token")  # 디버깅
                    # print("Grid:")
                    # for r in rows:
                    #     print(" ".join(r))
                    return False
                        
                # 규칙 2: U 셀의 위쪽 이웃은 U 또는 C여야 함
                if token == "U" and i > 0 and rows[i-1][j] not in ["C", "U"]:
                    # print(f"Validation failed: Invalid upper neighbor '{rows[i-1][j]}' for U token")  # 디버깅
                    # print("Grid:")
                    # for r in rows:
                    #     print(" ".join(r))
                    return False
                        
                # 규칙 3: X 셀 규칙
                if token == "X":
                    if j == 0 or i == 0:  # X는 첫 행/열에 올 수 없음
                        # print("Validation failed: X token in first row/column")  # 디버깅
                        # print("Grid:")
                        # for r in rows:
                        #     print(" ".join(r))
                        return False
                    # 왼쪽 이웃은 X 또는 U여야 함
                    if row[j-1] not in ["X", "U"]:
                        # print(f"Validation failed: Invalid left neighbor '{row[j-1]}' for X token")  # 디버깅
                        # print("Grid:")
                        # for r in rows:
                        #     print(" ".join(r))
                        return False
                    # 위쪽 이웃은 X 또는 L이어야 함
                    if rows[i-1][j] not in ["X", "L"]:
                        # print(f"Validation failed: Invalid upper neighbor '{rows[i-1][j]}' for X token")  # 디버깅
                        # print("Grid:")
                        # for r in rows:
                        #     print(" ".join(r))
                        return False
        return True

    def validate_otsl_sequence(self, sequence: str) -> bool:
        """OTSL 시퀀스 유효성 검사"""
        tokens = sequence.split()
        if not tokens:
            return False
            
        # 각 행이 최소 하나의 셀을 포함하는지 확인
        current_row = []
        for token in tokens:
            if token == 'NL':
                if not current_row:  # 빈 행 체크
                    return False
                current_row = []
            else:
                current_row.append(token)
                
        # 마지막 행 체크
        if not current_row:
            return False
            
        return True

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        html_structure: Optional[str] = None
    ) -> List[int]:
        """OTSL 시퀀스를 토큰 ID로 변환"""
        if max_length is None:
            max_length = self.max_length
        
        # 1. Tokenize
        tokens = text.split()
        if len(tokens) == 0 or len(tokens) > max_length:
            raise ValueError("Invalid OTSL sequence length")
        
        # 2. Validate syntax
        if not self.validate_syntax(tokens):
            raise ValueError("Invalid OTSL syntax")
        
        # 3. Convert to IDs (special token 제외)
        token_ids = []
        for token in tokens:
            if token in self.otsl_tags:
                token_ids.append(self.token2id[token])
            else:
                token_ids.append(self.unk_token_id)
        
        # 4. Truncation (special token 고려)
        if truncation and len(token_ids) > max_length - 2:  # BOS/EOS를 위한 공간 확보
            token_ids = token_ids[:max_length-2]
        
        # 5. Special tokens 추가
        final_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        # 6. Padding
        if padding == "max_length":
            pad_length = max_length - len(final_ids)
            if pad_length > 0:
                final_ids.extend([self.pad_token_id] * pad_length)
        
        return final_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """토큰 ID를 OTSL 시퀀스로 변환"""
        tokens = []
        special_token_ids = {
            self.bos_token_id,
            self.eos_token_id,
            self.pad_token_id,
            self.unk_token_id
        }
        
        for token_id in token_ids:
            # special token은 항상 스킵
            if token_id in special_token_ids:
                continue
            
            if token_id in self.id2token:
                token = self.id2token[token_id]
                # 일반 special token 처리 (skip_special_tokens가 True인 경우)
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                tokens.append(token)
        
        # OTSL 문법 규칙에 맞게 토큰 시퀀스 정리
        otsl_tokens = []
        current_row = []
        
        for token in tokens:
            if token == 'NL':
                if current_row:  # 빈 행이 아닌 경우에만 추가
                    otsl_tokens.extend(current_row)
                    otsl_tokens.append('NL')
                    current_row = []
            elif token in self.otsl_tags:
                current_row.append(token)
        
        # 마지막 행 처리
        if current_row:
            otsl_tokens.extend(current_row)
        
        return ' '.join(otsl_tokens)
