from typing import List, Tuple, Dict, Optional
import numpy as np
from zss import simple_distance
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
import re

@dataclass(frozen=True)
class Node:
    """HTML 트리의 노드"""
    tag: str
    text: str = ""
    children: Tuple['Node', ...] = ()
    
    def __init__(self, tag: str, text: str = "", children: Optional[List['Node']] = None):
        object.__setattr__(self, 'tag', tag)
        object.__setattr__(self, 'text', text)
        object.__setattr__(self, 'children', tuple(children) if children else ())

def _html_to_tree_impl(html: str) -> Optional[Node]:
    """실제 HTML 파싱 구현"""
    try:
        # HTML 전처리
        html = re.sub(r'\s+', ' ', html).strip()
        if not html:
            return None
            
        # 기본 태그들
        TABLE_START = "<table>"
        TABLE_END = "</table>"
        TR_START = "<tr>"
        TR_END = "</tr>"
        TD_START = "<td"
        TD_END = "</td>"
        
        def extract_td_info(td_str: str) -> Tuple[str, Dict[str, str]]:
            if not td_str:
                return "", {}
                
            text_match = re.search(r'>(.*?)</td>', td_str)
            if not text_match:
                return "", {}
            
            text = text_match.group(1).strip()
            attrs = {}
            
            colspan_match = re.search(r'colspan="(\d+)"', td_str)
            if colspan_match:
                attrs['colspan'] = colspan_match.group(1)
                
            rowspan_match = re.search(r'rowspan="(\d+)"', td_str)
            if rowspan_match:
                attrs['rowspan'] = rowspan_match.group(1)
                
            return text, attrs

        # HTML 정규화
        if not html.startswith(TABLE_START):
            html = TABLE_START + html
        if not html.endswith(TABLE_END):
            html = html + TABLE_END
        
        # 테이블 루트 노드 생성
        root_children = []
        
        # TR 태그 파싱
        tr_parts = re.split(f'{TR_START}|{TR_END}', html[len(TABLE_START):-len(TABLE_END)])
        
        for tr_part in tr_parts:
            if not tr_part.strip():
                continue
                
            tr_children = []
            
            # TD 태그 파싱
            td_parts = re.split(f'({TD_START}.*?{TD_END})', tr_part)
            for td_part in td_parts:
                if TD_START not in td_part:
                    continue
                    
                text, _ = extract_td_info(td_part)
                td_node = Node("td", text=text)
                tr_children.append(td_node)
                
            if tr_children:  # TD가 하나라도 있는 경우에만 TR 추가
                tr_node = Node("tr", children=tr_children)
                root_children.append(tr_node)
            
        return Node("table", children=root_children)
        
    except Exception as e:
        print(f"HTML parsing error: {e}")
        print(f"Problematic HTML: {html}")
        return None

@lru_cache(maxsize=1024)
def html_to_tree(html: str) -> Optional[Node]:
    """캐시를 사용하는 HTML 트리 변환 함수"""
    return _html_to_tree_impl(html)

def compute_tree_edit_distance(tree1: Node, tree2: Node) -> int:
    """Tree Edit Distance 계산"""
    if tree1 is None or tree2 is None:
        return 0
        
    def get_children(node: Node) -> Tuple[Node, ...]:
        return node.children
        
    def get_label(node: Node) -> str:
        return f"{node.tag}|{node.text}" if node.text else node.tag
    
    return simple_distance(
        tree1, tree2,
        get_children=get_children,
        get_label=get_label
    )

def compute_teds(pred_html: str, true_html: str) -> float:
    """TEDS 계산"""
    try:
        pred_tree = html_to_tree(pred_html)
        true_tree = html_to_tree(true_html)
        
        if pred_tree is None or true_tree is None:
            return 0.0
        
        edit_distance = compute_tree_edit_distance(pred_tree, true_tree)
        
        def count_nodes(node: Node) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)
        
        pred_size = count_nodes(pred_tree)
        true_size = count_nodes(true_tree)
        
        teds = 1 - (edit_distance / max(pred_size, true_size))
        return max(0.0, min(1.0, teds))
        
    except Exception as e:
        print(f"Error computing TEDS: {e}")
        return 0.0

def compute_teds_struct(pred_html: str, true_html: str) -> float:
    """TEDS-Struct 계산"""
    def remove_text(html: str) -> str:
        return re.sub(r'>(.*?)</td>', '/>', html)
    
    return compute_teds(remove_text(pred_html), remove_text(true_html))