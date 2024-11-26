from typing import List, Tuple, Dict
import numpy as np
from zss import simple_distance
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Node:
    """HTML 트리의 노드"""
    tag: str
    text: str = ""
    children: List['Node'] = None
    
    def __init__(self, tag: str, text: str = "", children: List['Node'] = None):
        self.tag = tag
        self.text = text
        self.children = children if children is not None else []

def html_to_tree(html: str) -> Node:
    """HTML 문자열을 트리 구조로 변환"""
    # 기본 태그들
    TABLE_START = "<table>"
    TABLE_END = "</table>"
    TR_START = "<tr>"
    TR_END = "</tr>"
    TD_START = "<td"
    TD_END = "</td>"
    
    def extract_td_info(td_str: str) -> Tuple[str, Dict[str, str]]:
        """TD 태그에서 속성과 텍스트 추출"""
        # 태그와 텍스트 분리
        start_idx = td_str.find(">") + 1
        end_idx = td_str.rfind(TD_END)
        if start_idx == 0 or end_idx == -1:
            return "", {}
            
        text = td_str[start_idx:end_idx].strip()
        
        # 속성 추출 (colspan, rowspan)
        attrs = {}
        if 'colspan="' in td_str:
            colspan = td_str[td_str.find('colspan="')+9:]
            attrs['colspan'] = colspan[:colspan.find('"')]
        if 'rowspan="' in td_str:
            rowspan = td_str[td_str.find('rowspan="')+9:]
            attrs['rowspan'] = rowspan[:rowspan.find('"')]
            
        return text, attrs

    try:
        # HTML 전처리
        html = html.replace("\n", "").strip()
        if not html.startswith(TABLE_START):
            html = TABLE_START + html
        if not html.endswith(TABLE_END):
            html = html + TABLE_END
        
        # 테이블 루트 노드 생성
        root = Node("table")
        
        # TR 태그 파싱
        tr_parts = html[len(TABLE_START):-len(TABLE_END)].split(TR_START)
        for tr_part in tr_parts:
            if not tr_part.strip():
                continue
                
            if TR_END not in tr_part:
                tr_part += TR_END
                
            tr_node = Node("tr")
            
            # TD 태그 파싱
            td_parts = tr_part[:-len(TR_END)].split(TD_START)[1:]  # 첫 빈 문자열 제거
            for td_part in td_parts:
                if TD_END not in td_part:
                    td_part += TD_END
                    
                text, attrs = extract_td_info(td_part)
                td_node = Node("td", text=text)
                tr_node.children.append(td_node)
                
            if td_parts:  # TD가 하나라도 있는 경우에만 TR 추가
                root.children.append(tr_node)
            
        return root
        
    except Exception as e:
        print(f"HTML parsing error: {e}")
        print(f"Problematic HTML: {html}")
        raise

def compute_tree_edit_distance(tree1: Node, tree2: Node) -> int:
    """Tree Edit Distance 계산"""
    def get_children(node: Node) -> List[Node]:
        return node.children
        
    def get_label(node: Node) -> str:
        return f"{node.tag}|{node.text}" if node.text else node.tag
    
    return simple_distance(
        tree1, tree2,
        get_children=get_children,
        get_label=get_label
    )

def compute_teds(pred_html: str, true_html: str) -> float:
    """
    TEDS (Tree-Edit-Distance-based Similarity) 계산
    
    Args:
        pred_html: 예측된 HTML 문자열
        true_html: 정답 HTML 문자열
    
    Returns:
        float: TEDS 점수 (0~1)
    """
    try:
        # HTML을 트리로 변환
        pred_tree = html_to_tree(pred_html)
        true_tree = html_to_tree(true_html)
        
        # Tree Edit Distance 계산
        edit_distance = compute_tree_edit_distance(pred_tree, true_tree)
        
        # 트리 크기 계산 (노드 수)
        def count_nodes(node: Node) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)
        
        pred_size = count_nodes(pred_tree)
        true_size = count_nodes(true_tree)
        
        # TEDS 계산 (논문 Equation 8)
        teds = 1 - (edit_distance / max(pred_size, true_size))
        
        return teds
        
    except Exception as e:
        print(f"Error computing TEDS: {e}")
        return 0.0

def compute_teds_struct(pred_html: str, true_html: str) -> float:
    """
    TEDS-Struct (구조만 고려한 TEDS) 계산
    
    Args:
        pred_html: 예측된 HTML 문자열
        true_html: 정답 HTML 문자열
    
    Returns:
        float: TEDS-Struct 점수 (0~1)
    """
    # 텍스트 제거하고 구조만 비교
    def remove_text(html: str) -> str:
        return html.replace("></td>", "/>").replace("</td>", "/>")
    
    return compute_teds(remove_text(pred_html), remove_text(true_html))