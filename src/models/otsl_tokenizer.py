from typing import List, Dict, Optional, Union
import re
import torch

class OTSLTokenizer:
    """
    OTSL (Optimised Table-Structure Language) Tokenizer
    
    OTSL은 5개의 기본 토큰으로 테이블 구조를 표현:
    - C: 새로운 테이블 셀 (기본 셀)
    - L: 왼쪽 이웃 셀과 병합하는 셀 (수평 병합)
    - U: 위쪽 이웃 셀과 병합하는 셀 (수직 병합)
    - X: 왼쪽과 위쪽 이웃 셀 모두와 병합하는 셀 (교차 병합)
    - NL: 새로운 행 시작
    
    문법 규칙:
    1. Left-looking cell rule: L 셀의 왼쪽 이웃은 L 또는 C여야 함
    2. Up-looking cell rule: U 셀의 위쪽 이웃은 U 또는 C여야 함
    3. Cross cell rule: X 셀의 왼쪽 이웃은 X 또는 U, 위쪽 이웃은 X 또는 L이어야 함
    4. First row rule: 첫 번째 행은 L과 C만 허용
    5. First column rule: 첫 번째 열은 U와 C만 허용
    6. Rectangular rule: 모든 행은 동일한 토큰 수를 가져야 함
    
    Note:
    - max_length는 OTSL 시퀀스만을 위한 길이입니다 (layout prompt 제외)
    - layout_prompt_length는 데이터셋의 대 bounding box 개수에 따라 결정됩니다
    - total_sequence_length는 논문의 실험 설정값 1376입니다
    """
    # 클래스 레벨에서 상수 정의
    SPECIAL_TOKENS = {
        "bos_token": "[BOS]",
        "pad_token": "[PAD]",
        "eos_token": "[EOS]"
    }
    OTSL_TAGS = ["C", "L", "U", "X", "NL"]

    def __init__(
        self,
        otsl_sequence_length: int,  # 논문 4.2 Experimental Settings
        bos_token: str = SPECIAL_TOKENS["bos_token"],
        pad_token: str = SPECIAL_TOKENS["pad_token"],
        eos_token: str = SPECIAL_TOKENS["eos_token"]
    ):
        self.otsl_sequence_length = otsl_sequence_length
        
        # 한 번에 vocabulary 생성
        all_tokens = list(self.SPECIAL_TOKENS.values()) + self.OTSL_TAGS
        self.token2id = {token: idx for idx, token in enumerate(all_tokens)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        
        # Special tokens와 their IDs
        self.bos_token = bos_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        
        self.bos_token_id = self.token2id[bos_token]
        self.pad_token_id = self.token2id[pad_token]
        self.eos_token_id = self.token2id[eos_token]
        
        # OTSL token IDs
        self.c_token_id = self.token2id["C"]
        self.l_token_id = self.token2id["L"]
        self.u_token_id = self.token2id["U"]
        self.x_token_id = self.token2id["X"]
        self.nl_token_id = self.token2id["NL"]
        
        # 모든 토큰 ID 리스트
        self.otsl_token_ids = list(self.token2id.values())

    
    def encode(
        self,
        tokens: List[str],
        padding: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        if not self.validate_syntax(tokens):
            raise ValueError("Invalid OTSL syntax")
        
        max_length = self.otsl_sequence_length
        
        token_ids = [self.pad_token_id] * max_length if padding else []
        
        current_pos = 0
        token_ids[current_pos] = self.bos_token_id
        current_pos += 1
        
        for token in tokens:
            if current_pos >= max_length - 1:
                break
            token_ids[current_pos] = self.token2id[token]
            current_pos += 1
        
        if current_pos < max_length:
            token_ids[current_pos] = self.eos_token_id
        
        if not padding:
            token_ids = token_ids[:current_pos + 1]
        
        return torch.tensor(token_ids) if return_tensors == "pt" else token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor]
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        
        tokens = []
        tokens.extend(
            self.id2token[token_id]
            for token_id in token_ids
            if token_id not in special_ids and token_id in self.id2token
        )
        
        return " ".join(tokens)
    
        
    def validate_syntax(self, tokens: List[str]) -> bool:
        """OTSL 문법 규칙 검증"""
        if not tokens:
            return False
            
        # 토큰을 2D 그리드로 변환
        rows = []
        current_row = []
        
        for token in tokens:
            if token == "NL":
                if not current_row:  # 빈 행은 허용되지 않음
                    return False
                rows.append(current_row)
                current_row = []
            else:
                current_row.append(token)
                
        if current_row:  # 마지막 행 처리
            rows.append(current_row)
            
        # 규칙 6: 직사각형 규칙 - 모든 행의 길이가 동일해야 함
        if not all(len(row) == len(rows[0]) for row in rows):
            return False
            
        # 나머지 규칙 검증
        for i, row in enumerate(rows):
            for j, token in enumerate(row):
                # 규칙 4: 첫 번째 행은 L과 C만 허용
                if i == 0 and token not in ["C", "L"]:
                    return False
                    
                # 규칙 5: 첫 번째 열은 U와 C만 허용
                if j == 0 and token not in ["C", "U"]:
                    return False
                    
                # 규칙 1: L 셀의 왼쪽 이웃은 L 또는 C여야 함
                if token == "L" and j > 0 and row[j-1] not in ["C", "L"]:
                    return False
                    
                # 규칙 2: U 셀의 위쪽 이웃은 U 또는 C여야 함
                if token == "U" and i > 0 and rows[i-1][j] not in ["C", "U"]:
                    return False
                    
                # 규칙 3: X 셀 규칙
                if token == "X":
                    if j == 0 or i == 0:  # X는 첫 행/열에 올 수 없음
                        return False
                    # 왼쪽 이웃은 X 또는 U여야 함
                    if row[j-1] not in ["X", "U"]:
                        return False
                    # 위쪽 이웃은 X 또는 L이어야 함
                    if rows[i-1][j] not in ["X", "L"]:
                        return False
                        
        return True
    
    
