import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from typing import Iterator, List, Optional


class OTSLLengthBatchSampler(Sampler):
    """OTSL 시퀀스 길이 기반 BatchSampler
    버킷 수 = 배치 사이즈로 설정하여 각 버킷에서 하나씩 샘플링
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_buckets = batch_size  # 버킷 수 = 배치 사이즈
        
        # 길이 계산
        self.lengths = np.array([
            len(dataset.annotations[dataset.image_names[idx]]['otsl_tokens_list'])
            for idx in range(len(dataset))
        ])
        
        # 버킷 경계 계산 (균등 분할)
        self.boundaries = np.percentile(
            self.lengths, 
            np.linspace(0, 100, self.num_buckets + 1)
        )
        
        # 인덱스를 버킷에 할당
        self.buckets = [[] for _ in range(self.num_buckets)]
        for idx, length in enumerate(self.lengths):
            bucket_idx = np.searchsorted(self.boundaries[1:], length)
            self.buckets[min(bucket_idx, self.num_buckets-1)].append(idx)
        
        # 버킷 통계 출력
        print("\nBucket Statistics:")
        for i, bucket in enumerate(self.buckets):
            if len(bucket) > 0:
                min_len = min(self.lengths[bucket])
                max_len = max(self.lengths[bucket])
                print(f"Bucket {i}: {len(bucket)} samples, "
                      f"length range: {min_len}-{max_len}")
    
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # 각 버킷 내부를 섞음
            shuffled_buckets = [np.random.permutation(bucket).tolist() 
                              for bucket in self.buckets]
        else:
            shuffled_buckets = [bucket.copy() for bucket in self.buckets]
        
        batches = []
        while True:
            # 비어있지 않은 버킷 확인
            valid_buckets = [b for b in shuffled_buckets if len(b) > 0]
            
            if not valid_buckets:
                break
                
            # 현재 배치 생성
            current_batch = []
            for bucket in shuffled_buckets:
                if bucket:  # 버킷에 샘플이 있으면
                    current_batch.append(bucket.pop())  # 마지막 요소 추출
            
            if len(current_batch) == self.batch_size or (not self.drop_last and current_batch):
                batches.append(current_batch)
        
        if self.shuffle:
            np.random.shuffle(batches)
        
        return iter(batches)
    
    def __len__(self) -> int:
        min_bucket_size = min(len(bucket) for bucket in self.buckets)
        num_full_batches = min_bucket_size
        
        if self.drop_last:
            return num_full_batches
        
        remaining_samples = sum(
            len(bucket) - num_full_batches 
            for bucket in self.buckets
        )
        return num_full_batches + (remaining_samples + self.batch_size - 1) // self.batch_size