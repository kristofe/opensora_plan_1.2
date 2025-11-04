"""
Decord API compatibility wrapper using PyAV.
This provides a drop-in replacement for decord.VideoReader using the av library.
"""

import av
import numpy as np
import torch
from typing import Union, List


class NDArray:
    """
    Wrapper around numpy array that provides decord's .asnumpy() method.
    This allows the wrapper to match decord's API exactly.
    """
    
    def __init__(self, array: np.ndarray):
        self._array = array
    
    def asnumpy(self):
        """Return the underlying numpy array."""
        return self._array
    
    def __array__(self):
        """Allow numpy operations on this object."""
        return self._array
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying numpy array."""
        return getattr(self._array, name)
    
    def __getitem__(self, key):
        """Delegate indexing to underlying numpy array."""
        return self._array[key]
    
    def __repr__(self):
        return f"NDArray({self._array.__repr__()})"
    
    def __str__(self):
        return self._array.__str__()


class _Context:
    """Mock context object to match decord's cpu() function."""
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.device_type = "cpu"


def cpu(device_id=0):
    """Mock decord.cpu() function."""
    return _Context(device_id)


class VideoReader:
    """
    Drop-in replacement for decord.VideoReader using PyAV.
    Provides the same interface as decord for minimal code disruption.
    """
    
    def __init__(self, uri, ctx=None, num_threads=1, **kwargs):
        """
        Initialize VideoReader.
        
        Args:
            uri: Path to video file or URL
            ctx: Context (ignored, kept for API compatibility)
            num_threads: Number of threads for decoding
            **kwargs: Additional arguments (ignored for compatibility)
        """
        self.uri = uri
        self.ctx = ctx
        self.num_threads = num_threads
        
        # Open video with PyAV
        self.container = av.open(uri, metadata_errors='ignore')
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self.stream.thread_count = num_threads
        
        # Cache video properties
        self._duration = self.stream.duration
        self._fps = float(self.stream.average_rate)
        self._num_frames = self.stream.frames
        
        # If frame count is not available, estimate from duration and fps
        if self._num_frames == 0:
            if self._duration and self.stream.time_base:
                duration_sec = float(self._duration * self.stream.time_base)
                self._num_frames = int(duration_sec * self._fps)
        
        # Frame cache for efficient random access
        self._frame_cache = {}
        
    def __len__(self):
        """Return total number of frames."""
        return self._num_frames
    
    def __getitem__(self, indices: Union[int, List[int], slice, np.ndarray, torch.Tensor]):
        """
        Get frame(s) by index.
        
        Args:
            indices: Frame index, list of indices, slice, or tensor of indices
            
        Returns:
            numpy array of shape (H, W, C) for single frame or (N, H, W, C) for multiple frames
        """
        # Convert various index types to list
        if isinstance(indices, int):
            result = self._get_frames([indices])
            # For single frame, return the unwrapped numpy array
            if len(result.asnumpy()) > 0:
                return result.asnumpy()[0]
            return None
        elif isinstance(indices, slice):
            start = indices.start or 0
            stop = indices.stop or self._num_frames
            step = indices.step or 1
            indices_list = list(range(start, stop, step))
            return self._get_frames(indices_list)
        elif isinstance(indices, (list, tuple)):
            return self._get_frames(list(indices))
        elif isinstance(indices, np.ndarray):
            return self._get_frames(indices.tolist())
        elif isinstance(indices, torch.Tensor):
            return self._get_frames(indices.tolist())
        else:
            raise TypeError(f"Unsupported index type: {type(indices)}")
    
    def _get_frames(self, indices: List[int]):
        """
        Get multiple frames by indices.
        
        Args:
            indices: List of frame indices
            
        Returns:
            NDArray object (numpy array with .asnumpy() method)
        """
        if len(indices) == 0:
            return NDArray(np.array([]))
        
        frames = []
        indices_sorted = sorted(set(indices))
        
        # Determine if we should use seeking or sequential reading
        # Sequential is faster if indices are mostly consecutive
        use_sequential = len(indices_sorted) > 10 and self._is_mostly_sequential(indices_sorted)
        
        if use_sequential:
            frames_dict = self._read_sequential(indices_sorted)
        else:
            frames_dict = self._read_with_seeking(indices_sorted)
        
        # Return frames in the original order (including duplicates)
        frames = [frames_dict[idx] for idx in indices if idx in frames_dict]
        
        if not frames:
            return NDArray(np.array([]))
        
        return NDArray(np.stack(frames, axis=0))
    
    def _is_mostly_sequential(self, indices: List[int], threshold=0.7):
        """Check if indices are mostly sequential."""
        if len(indices) < 2:
            return True
        consecutive = sum(1 for i in range(len(indices)-1) if indices[i+1] - indices[i] == 1)
        return consecutive / (len(indices) - 1) >= threshold
    
    def _read_sequential(self, indices: List[int]):
        """Read frames sequentially (efficient for consecutive frames)."""
        frames_dict = {}
        target_indices = set(indices)
        
        # Reopen container for clean sequential read
        self.container.close()
        self.container = av.open(self.uri, metadata_errors='ignore')
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self.stream.thread_count = self.num_threads
        
        frame_idx = 0
        for frame in self.container.decode(video=0):
            if frame_idx in target_indices:
                frames_dict[frame_idx] = frame.to_ndarray(format='rgb24')
                if len(frames_dict) == len(target_indices):
                    break
            frame_idx += 1
            
        return frames_dict
    
    def _read_with_seeking(self, indices: List[int]):
        """Read frames with seeking (efficient for random access)."""
        frames_dict = {}
        
        # Reopen container for clean seeking
        self.container.close()
        self.container = av.open(self.uri, metadata_errors='ignore')
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = 'AUTO'
        self.stream.thread_count = self.num_threads
        
        for idx in indices:
            if idx < 0 or idx >= self._num_frames:
                continue
                
            # Calculate timestamp for seeking
            timestamp = int(idx / self._fps / float(self.stream.time_base))
            
            try:
                # Seek to the frame
                self.container.seek(timestamp, stream=self.stream)
                
                # Decode frames until we get the right one
                target_pts = int(idx * self.stream.duration / max(self._num_frames, 1))
                
                for frame in self.container.decode(video=0):
                    frame_idx = int(frame.pts * self._num_frames / max(self.stream.duration, 1))
                    if frame_idx >= idx:
                        frames_dict[idx] = frame.to_ndarray(format='rgb24')
                        break
                        
            except (av.AVError, Exception) as e:
                # If seeking fails, fall back to sequential read for this index
                self.container.close()
                self.container = av.open(self.uri, metadata_errors='ignore')
                self.stream = self.container.streams.video[0]
                
                current_idx = 0
                for frame in self.container.decode(video=0):
                    if current_idx == idx:
                        frames_dict[idx] = frame.to_ndarray(format='rgb24')
                        break
                    current_idx += 1
                    if current_idx > idx:
                        break
                        
        return frames_dict
    
    def get_batch(self, indices: List[int]):
        """
        Get a batch of frames (decord-compatible method).
        
        Args:
            indices: List of frame indices
            
        Returns:
            numpy array of shape (N, H, W, C)
        """
        return self._get_frames(indices)
    
    def get_avg_fps(self):
        """Get average FPS of the video."""
        return self._fps
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'container'):
            try:
                self.container.close()
            except:
                pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.__del__()


# For compatibility with: import decord; decord.VideoReader(...)
__all__ = ['VideoReader', 'cpu', 'NDArray']

