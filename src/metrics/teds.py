from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from zss import simple_distance
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
import re
from concurrent.futures import ThreadPoolExecutor
import torch

@dataclass(frozen=True)
class Node:
    """최적화된 HTML 트리 노드"""
    tag: str
    text: str = ""
    children: Tuple['Node', ...] = ()
    _hash: int = None  # 해시 캐싱
    _size: int = None  # 노드 수 캐싱
    
    def __post_init__(self):
        # frozen=True이므로 object.__setattr__ 사용
        if self._hash is None:
            object.__setattr__(self, '_hash', hash((self.tag, self.text, self.children)))
        if self._size is None:
            object.__setattr__(self, '_size', 1 + sum(child._size for child in self.children))
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other: 'Node') -> bool:
        if not isinstance(other, Node):
            return False
        return self._hash == other._hash  # 해시 비교로 최적화

# 정규식 패턴을 전역으로 한 번만 컴파일
TD_PATTERN = re.compile(r'>(.*?)</td>')
COLSPAN_PATTERN = re.compile(r'colspan="(\d+)"')
ROWSPAN_PATTERN = re.compile(r'rowspan="(\d+)"')
TEXT_REMOVE_PATTERN = re.compile(r'>([^<]*)</td>')

# Tree Edit Distance 계산을 위한 최적화된 함수들
@lru_cache(maxsize=16384)  # 캐시 크기 증가
def get_node_label(node: Node) -> str:
    return f"{node.tag}|{node.text}" if node.text else node.tag

@lru_cache(maxsize=16384)
def compute_tree_edit_distance(tree1: Node, tree2: Node) -> int:
    """최적화된 Tree Edit Distance 계산"""
    if tree1 is None or tree2 is None:
        return 0
    
    return simple_distance(
        tree1, tree2,
        get_children=lambda x: x.children,
        get_label=get_node_label
    )

class TEDSCalculator:
    """TEDS 계산을 위한 최적화된 클래스"""
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache = {}
        self._tree_cache = {}
        self._struct_pattern = re.compile(r'>([^<]*)</td>')
    
    def _get_cached_tree(self, html: str) -> Optional[Node]:
        """HTML to Tree 변환 결과 캐싱"""
        if html not in self._tree_cache:
            self._tree_cache[html] = html_to_tree(html)
        return self._tree_cache[html]
    
    def compute_batch_teds_struct(self, pred_htmls: List[str], true_htmls: List[str]) -> List[float]:
        """구조적 TEDS 배치 계산"""
        pred_htmls_struct = [self._struct_pattern.sub('></td>', html) for html in pred_htmls]
        true_htmls_struct = [self._struct_pattern.sub('></td>', html) for html in true_htmls]
        return self.compute_batch_teds(pred_htmls_struct, true_htmls_struct)
    
    def compute_batch_teds(self, pred_htmls: List[str], true_htmls: List[str]) -> List[float]:
        """배치 단위 TEDS 계산 (병렬 처리)"""
        futures = []
        results = [0.0] * len(pred_htmls)  # 미리 결과 리스트 할당
        
        # 캐시 확인 및 미스된 항목만 계산
        for i, (pred, true) in enumerate(zip(pred_htmls, true_htmls)):
            key = hash((pred, true))
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                futures.append((i, self._executor.submit(self._compute_single_teds, pred, true)))
        
        # 병렬 계산 결과 수집
        for i, future in futures:
            results[i] = future.result()
            
        return results
    
    def _compute_single_teds(self, pred_html: str, true_html: str) -> Tuple[float, float]:
        """단일 HTML 쌍에 대해 TEDS와 TEDS-struct를 동시에 계산"""
        return compute_teds_both(pred_html, true_html)
    
    @staticmethod
    @lru_cache(maxsize=16384)
    def _count_nodes(node: Node) -> int:
        """노드 수 계산 (캐시 사용)"""
        return 1 + sum(TEDSCalculator._count_nodes(child) for child in node.children)

def html_to_tree(html: str) -> Optional[Node]:
    """최적화된 HTML 파싱"""
    if not html or html.isspace():
        return None
    
    try:
        # HTML 파싱 최적화
        parts = html.split('<')
        tokens = []
        for part in parts[1:]:  # 첫 번째 빈 문자열 건너뛰기
            if '>' in part:
                tag, content = part.split('>', 1)
                if tag.startswith('/'):  # 닫는 태그
                    tokens.append(f'</{tag[1:]}>')
                else:  # 여는 태그
                    tokens.append(f'<{tag}>')
                    if content and not content.isspace():
                        tokens.append(content)
        
        def build_tree(tokens: List[str], start: int = 0) -> Tuple[Optional[Node], int]:
            if start >= len(tokens):
                return None, start
            
            token = tokens[start]
            
            # 닫는 태그 처리
            if token.startswith('</'):
                return None, start
            
            # 여는 태그 처리
            if token.startswith('<'):
                tag = token[1:].rstrip('>').split()[0]
                children = []
                current_pos = start + 1
                
                while current_pos < len(tokens):
                    child, new_pos = build_tree(tokens, current_pos)
                    if child is None:
                        break
                    children.append(child)
                    current_pos = new_pos + 1
                
                return Node(tag=tag, children=tuple(children)), current_pos
            
            # 텍스트 노드 처리
            return Node(tag="text", text=token), start
        
        root, _ = build_tree(tokens)
        return root
        
    except Exception as e:
        print(f"Error in html_to_tree: {str(e)}")
        return None

# TEDS 계산기 인스턴스 생성 (전역으로 한 번만)
teds_calculator = TEDSCalculator(max_workers=4)

def compute_teds(pred_html: str, true_html: str) -> float:
    """TEDS 계산 인터페이스"""
    return teds_calculator._compute_single_teds(pred_html, true_html)

def compute_teds_struct(pred_html: str, true_html: str) -> float:
    """TEDS-Struct 계산 인터페이스
    셀의 텍스트 내용은 무시하고 테이블 구조만 비교
    
    예시:
    <td colspan="2">Hello</td> -> <td colspan="2"></td>
    <td>World</td> -> <td></td>
    """
    def remove_cell_content(html):
        # colspan, rowspan 등의 구조 정보는 유지하면서 셀 내용만 제거
        return re.sub(r'(<td[^>]*>).*?(</td>)', r'\1\2', html)
    
    pred_html_struct = remove_cell_content(pred_html)
    true_html_struct = remove_cell_content(true_html)
    return compute_teds(pred_html_struct, true_html_struct)

def compute_teds_both(pred_html: str, true_html: str) -> Tuple[float, float]:
    """TEDS와 TEDS-struct를 동시에 계산
    한 번의 파싱으로 두 메트릭을 모두 계산하여 효율성 향상
    
    Returns:
        Tuple[float, float]: (teds, teds_struct) 점수
    """
    try:
        # 1. HTML 파싱 (한 번만 수행)
        pred_tree_full = html_to_tree(pred_html)
        true_tree_full = html_to_tree(true_html)
        
        if pred_tree_full is None or true_tree_full is None:
            return 0.0, 0.0
            
        # 2. 구조적 트리 생성 (텍스트 제거)
        def create_struct_tree(node: Node) -> Node:
            return Node(
                tag=node.tag,
                text="",  # 텍스트 제거
                children=tuple(create_struct_tree(child) for child in node.children)
            )
            
        pred_tree_struct = create_struct_tree(pred_tree_full)
        true_tree_struct = create_struct_tree(true_tree_full)
        
        # 3. Tree Edit Distance 계산 (두 버전 모두)
        edit_distance_full = compute_tree_edit_distance(pred_tree_full, true_tree_full)
        edit_distance_struct = compute_tree_edit_distance(pred_tree_struct, true_tree_struct)
        
        # 4. 노드 수 계산 (한 번만 수행)
        pred_size = pred_tree_full._size  # 캐시된 크기 사용
        true_size = true_tree_full._size  # 캐시된 크기 사용
        max_size = max(pred_size, true_size)
        
        # 5. TEDS 점수 계산
        teds = 1 - (edit_distance_full / max_size)
        teds_struct = 1 - (edit_distance_struct / max_size)
        
        return max(0.0, min(1.0, teds)), max(0.0, min(1.0, teds_struct))
        
    except Exception as e:
        print(f"Error computing TEDS: {e}")
        return 0.0, 0.0