from typing import Dict, List, Tuple
import re
from bs4 import BeautifulSoup

class HTMLtoOTSLConverter:
    """
    HTML을 OTSL로 변환하는 클래스
    
    논문 Section 3.4 참조:
    - HTML 태그를 더 짧은 OTSL 태그로 변환
    - Span 정보 보존 (논문 3.6 Span-aware Contrastive Learning)
    """
    def __init__(self) -> None:
        # OTSL -> HTML 태그 매핑 (논문 3.4)
        self.otsl_to_html = {
            'C': '<td>',           # 기본 셀
            'L': '<td colspan="2">', # 가로 병합
            'U': '<td rowspan="2">', # 세로 병합
            'X': '<td colspan="2" rowspan="2">', # 2D 병합
            'NL': '</tr><tr>'      # 새 행
        }
        
        # HTML -> OTSL 태그 매핑
        self.html_to_otsl = {v: k for k, v in self.otsl_to_html.items()}
    
    def convert_otsl_to_html(self, otsl_sequence: str) -> str:
        """OTSL 시퀀스를 HTML로 변환"""
        tokens = otsl_sequence.strip().split()
        html_parts = ['<table><tr>']  # 시작 태그
        
        for token in tokens:
            if token == 'NL':
                # 현재 행을 닫고 새 행 시작
                if html_parts[-1] != '<tr>':  # 빈 행이 아닌 경우에만
                    html_parts.append('</tr><tr>')
            else:
                # 셀 태그 추가
                html_tag = self.otsl_to_html.get(token, '<td>')
                html_parts.append(html_tag)
                html_parts.append('</td>')  # 셀 닫기
        
        # 마지막 행과 테이블 닫기
        if html_parts[-1] == '<tr>':
            html_parts.pop()  # 빈 행 제거
        else:
            html_parts.append('</tr>')
        html_parts.append('</table>')
        
        return ' '.join(html_parts)
    
    def convert_html_to_otsl(self, html_structure: str) -> str:
        """HTML을 OTSL로 변환"""
        # HTML 태그 파싱
        tokens = []
        current_row = []
        in_cell = False
        
        # HTML 문자열을 토큰으로 분리
        html_tokens = html_structure.strip().split()
        
        for token in html_tokens:
            if token == '<table>' or token == '</table>':
                continue
            elif token == '<tr>':
                if current_row:  # 이전 행이 있으면 NL 추가
                    tokens.append('NL')
            elif token == '</tr>':
                if current_row:
                    tokens.extend(current_row)
                    current_row = []
            elif token.startswith('<td') or token.startswith('<th'):
                # Span 정보 추출
                if 'colspan="2"' in token and 'rowspan="2"' in token:
                    current_row.append('X')
                elif 'colspan="2"' in token:
                    current_row.append('L')
                elif 'rowspan="2"' in token:
                    current_row.append('U')
                else:
                    current_row.append('C')
                in_cell = True
            elif token == '</td>' or token == '</th>':
                in_cell = False
        
        # 마지막 행 처리
        if current_row:
            tokens.extend(current_row)
        
        return ' '.join(tokens)