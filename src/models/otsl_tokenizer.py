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
    def __init__(
        self,
        otsl_sequence_length: int = 1376 // 2,  # 논문 4.2 Experimental Settings
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
    ):
        # 기본 설정
        self.otsl_sequence_length = otsl_sequence_length
        
        # Special tokens
        self.special_tokens = {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
        }
        
        # OTSL의 5가지 기본 토큰
        self.otsl_tags = ["C", "L", "U", "X", "NL"]
        
        # Build vocabulary
        self.token2id = {}
        idx = 0
        
        # Add special tokens
        for token in self.special_tokens.values():
            self.token2id[token] = idx
            idx += 1
            
        # Add OTSL tokens
        for token in self.otsl_tags:
            self.token2id[token] = idx
            idx += 1
            
        # Create reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        
        # Save attributes
        self.vocab_size = len(self.token2id)
        
        # Special token IDs
        self.pad_token_id = self.token2id[pad_token]
        self.unk_token_id = self.token2id[unk_token]
        self.bos_token_id = self.token2id[bos_token]
        self.eos_token_id = self.token2id[eos_token]
        
        # Special token attributes
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # OTSL token IDs
        self.c_token_id = self.token2id["C"]
        self.l_token_id = self.token2id["L"]
        self.u_token_id = self.token2id["U"]
        self.x_token_id = self.token2id["X"]
        self.nl_token_id = self.token2id["NL"]
    
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
    
    def encode(
        self,
        tokens: List[str],
        add_special_tokens: bool = True,
        padding: bool = True,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], torch.Tensor]:
        """OTSL 시퀀스를 토큰 ID로 변환"""
        max_length = self.otsl_sequence_length
            
        # Validate syntax
        if not self.validate_syntax(tokens):
            raise ValueError("Invalid OTSL syntax")
            
        # Convert to IDs
        token_ids = []
        
        # Add BOS token
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
            
        # Add OTSL tokens
        for token in tokens:
            if token in self.token2id:
                token_ids.append(self.token2id[token])
            else:
                token_ids.append(self.unk_token_id)
                
        # Add EOS token
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
            
        # Pad if needed
        if padding and len(token_ids) < max_length:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
            
        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor(token_ids)
            
        return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """토큰 ID를 OTSL 시퀀스로 변환"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        tokens = []
        
        for token_id in token_ids:
            # Skip special tokens if requested
            if skip_special_tokens:
                if token_id in [self.pad_token_id, self.unk_token_id,
                                self.bos_token_id, self.eos_token_id]:
                    continue
                    
            # Convert token ID to text
            if token_id in self.id2token:
                tokens.append(self.id2token[token_id])
            else:
                tokens.append(self.unk_token)
                
        # Join tokens
        text = " ".join(tokens)
        
        # Clean up spaces if requested
        if clean_up_tokenization_spaces:
            text = self._clean_up_tokenization(text)
            
        return text
    
    def _clean_up_tokenization(self, text: str) -> str:
        """토크나이제이션 후처리"""
        # OTSL 토큰 사이의 공백 처리
        text = re.sub(r'\s+(NL)\s+', r' \1 ', text)
        text = re.sub(r'\s+([CLUX])\s+', r' \1 ', text)
        return text.strip()
    
    def convert_html_to_otsl(self, html: str) -> str:
        """HTML을 OTSL로 변환 (구현 필요)"""
        raise NotImplementedError("HTML to OTSL conversion not implemented yet")
    
    def convert_otsl_to_html(self, otsl: str) -> str:
        """OTSL을 HTML로 변환 (구현 필요)"""
        raise NotImplementedError("OTSL to HTML conversion not implemented yet")
