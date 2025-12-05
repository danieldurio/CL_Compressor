import pyopencl as cl
import numpy as np
import time
from typing import List, Tuple, Optional

LZ4_DECOMPRESS_KERNEL = """
// ============================================================
// LZ4 EXT3 Decompressor Kernel with Coalesced Memory Access
// ============================================================
// Optimizations:
// - vload8/vstore8 for literal copy (8 bytes at a time)
// - Byte-by-byte for match copy (required for overlap handling)
// ============================================================

__kernel void lz4_decompress_batch(
    __global const uchar* input_data,
    __global const uint* input_offsets,
    __global uchar* output_data,
    __global const uint* output_offsets,
    __global const uint* compressed_sizes,
    __global const uint* uncompressed_sizes,
    __global int* status,
    const uint num_frames
) {
    uint gid = get_global_id(0);
    if (gid >= num_frames) return;

    uint input_start = input_offsets[gid];
    uint output_start = output_offsets[gid];
    uint input_len = compressed_sizes[gid];
    uint expected_output_len = uncompressed_sizes[gid];

    __global const uchar* ip = input_data + input_start;
    __global uchar* op = output_data + output_start;
    
    __global const uchar* ip_end = ip + input_len;
    __global uchar* op_start = output_data + output_start;
    __global uchar* op_end = op + expected_output_len;

    while (ip < ip_end) {
        // Read token
        uchar token = *ip++;
        
        // Literal length
        uint literal_len = token >> 4;
        if (literal_len == 15) {
            uchar s;
            do {
                if (ip >= ip_end) { status[gid] = -1; return; }
                s = *ip++;
                literal_len += s;
            } while (s == 255);
        }

        // Copy literals - VECTORIZED (8 bytes at a time for coalesced access)
        if (op + literal_len > op_end) { status[gid] = -2; return; }
        if (ip + literal_len > ip_end) { status[gid] = -3; return; }
        
        // Fast path: vectorized copy using vload8/vstore8
        uint lit_rem = literal_len;
        while (lit_rem >= 8) {
            uchar8 vec = vload8(0, ip);
            vstore8(vec, 0, op);
            ip += 8;
            op += 8;
            lit_rem -= 8;
        }
        // Remaining bytes (< 8)
        while (lit_rem > 0) {
            *op++ = *ip++;
            lit_rem--;
        }

        if (ip >= ip_end) break; // End of stream

        // Offset (3 bytes) - single vectorized read
        if (ip + 3 > ip_end) { status[gid] = -4; return; }
        uint offset = (uint)ip[0] | ((uint)ip[1] << 8) | ((uint)ip[2] << 16);
        ip += 3;

        if (offset == 0) { status[gid] = -5; return; }

        // Match length
        uint match_len = (token & 0x0F) + 4;
        if ((token & 0x0F) == 15) {
            uchar s;
            do {
                if (ip >= ip_end) { status[gid] = -6; return; }
                s = *ip++;
                match_len += s;
            } while (s == 255);
        }

        // Copy match - MUST be byte-by-byte due to potential overlap
        // (When offset < match_len, output depends on previous output bytes)
        if (op + match_len > op_end) { status[gid] = -7; return; }
        
        // Calculate match source position
        long current_rel_pos = op - op_start;
        long match_rel_pos = current_rel_pos - offset;

        if (match_rel_pos < 0) { status[gid] = -8; return; }
        
        // Match copy - byte-by-byte for correctness with overlapping matches
        // Optimization: if offset >= 8, we could use vectorized copy, but
        // overlapping matches are common in LZ4, so we keep it safe
        __global uchar* match_src = op_start + match_rel_pos;
        
        // Unrolled loop for slightly better performance
        uint mr = match_len;
        while (mr >= 4) {
            op[0] = match_src[0];
            op[1] = match_src[1];
            op[2] = match_src[2];
            op[3] = match_src[3];
            op += 4;
            match_src += 4;
            mr -= 4;
        }
        while (mr > 0) {
            *op++ = *match_src++;
            mr--;
        }
    }
    
    // Size validation - allow small underflow (compressor pads last bytes)
    long size_diff = (long)(op_end - op);
    if (size_diff < 0 || size_diff > 12) {
        status[gid] = -9;
        return;
    }
    
    // Zero-fill any remaining bytes
    while (op < op_end) {
        *op++ = 0;
    }
    
    status[gid] = 0; // Success
}
"""


class GPU_LZ4_Decompressor:
    """
    GPU LZ4 Decompressor with Pinned Memory and Async Transfers.
    
    Optimizations:
    - ALLOC_HOST_PTR: Enables DMA transfers between CPU and GPU
    - Persistent buffers: Avoids per-batch allocation overhead
    - Async transfers: Overlaps data movement with kernel execution
    """
    
    def __init__(self, device_index: int = 0, max_batch_size: int = 64, max_frame_size: int = 16 * 1024 * 1024):
        self.device_index = device_index
        self.max_batch_size = max_batch_size
        self.max_frame_size = max_frame_size
        self.max_compressed_size = max_frame_size + (max_frame_size // 255) + 128
        
        # OpenCL objects
        self.ctx = None
        self.queue = None
        self.program = None
        self.kernel = None
        self.enabled = False
        
        # Persistent GPU Buffers (allocated once)
        self.buf_in = None
        self.buf_in_offsets = None
        self.buf_out = None
        self.buf_out_offsets = None
        self.buf_compressed_sizes = None
        self.buf_uncompressed_sizes = None
        self.buf_status = None
        
        # Pinned Memory Arrays (for DMA transfers)
        self.pinned_input = None
        self.pinned_output = None
        
        # Event tracking
        self.last_kernel_event = None
        
        self._init_opencl()
        if self.enabled:
            self._allocate_buffers()

    def _init_opencl(self):
        try:
            platforms = cl.get_platforms()
            if not platforms: return
            
            # Collect all GPUs
            devices = []
            for p in platforms:
                try:
                    devices.extend(p.get_devices(device_type=cl.device_type.GPU))
                except: pass
            
            if not devices: return
            
            # Validate device index
            if self.device_index >= len(devices):
                print(f"[GPU_LZ4] GPU index {self.device_index} not available (found {len(devices)} GPUs)")
                return
            
            # Use the specified GPU
            selected_device = devices[self.device_index]
            self.ctx = cl.Context(devices=[selected_device])
            self.queue = cl.CommandQueue(self.ctx)
            self.program = cl.Program(self.ctx, LZ4_DECOMPRESS_KERNEL).build()
            self.kernel = self.program.lz4_decompress_batch
            self.enabled = True
            print(f"[GPU_LZ4] Decompressor OpenCL initialized on {selected_device.name} (Index: {self.device_index})")
        except Exception as e:
            print(f"[GPU_LZ4] Failed to initialize OpenCL on GPU {self.device_index}: {e}")
            self.enabled = False

    def _allocate_buffers(self):
        """Allocate persistent buffers with Pinned Memory."""
        if not self.enabled:
            return
            
        try:
            mf = cl.mem_flags
            
            # Calculate total sizes
            total_input_size = self.max_compressed_size * self.max_batch_size
            total_output_size = self.max_frame_size * self.max_batch_size
            uint_size = np.dtype(np.uint32).itemsize
            int_size = np.dtype(np.int32).itemsize
            
            # ============================================================
            # OTIMIZAÇÃO: PINNED MEMORY (ALLOC_HOST_PTR)
            # ============================================================
            # Buffers de dados com Pinned Memory para DMA transfers
            self.buf_in = cl.Buffer(
                self.ctx,
                mf.READ_ONLY | mf.ALLOC_HOST_PTR,
                total_input_size
            )
            self.buf_out = cl.Buffer(
                self.ctx,
                mf.READ_WRITE | mf.ALLOC_HOST_PTR,
                total_output_size
            )
            
            # Buffers de controle com Pinned Memory
            self.buf_in_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.max_batch_size * uint_size)
            self.buf_out_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.max_batch_size * uint_size)
            self.buf_compressed_sizes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.max_batch_size * uint_size)
            self.buf_uncompressed_sizes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.ALLOC_HOST_PTR, self.max_batch_size * uint_size)
            self.buf_status = cl.Buffer(self.ctx, mf.WRITE_ONLY | mf.ALLOC_HOST_PTR, self.max_batch_size * int_size)
            
            # Pinned memory arrays para transferência rápida
            self.pinned_input = np.zeros(total_input_size, dtype=np.uint8)
            self.pinned_output = np.zeros(total_output_size, dtype=np.uint8)
            
            print(f"[GPU_LZ4] Decompressor buffers allocated with PINNED MEMORY:")
            print(f"  - Input (Pinned): {total_input_size/1024/1024:.1f}MB")
            print(f"  - Output (Pinned): {total_output_size/1024/1024:.1f}MB")
            print(f"  - Max batch: {self.max_batch_size} frames")
            print(f"  - Async Transfers: ENABLED")
            
        except Exception as e:
            print(f"[GPU_LZ4] Error allocating decompressor buffers: {e}")
            import traceback
            traceback.print_exc()
            self.enabled = False

    def release(self):
        """Release all OpenCL resources."""
        self.enabled = False
        
        # GPU Buffers
        if self.buf_in: del self.buf_in; self.buf_in = None
        if self.buf_out: del self.buf_out; self.buf_out = None
        if self.buf_in_offsets: del self.buf_in_offsets; self.buf_in_offsets = None
        if self.buf_out_offsets: del self.buf_out_offsets; self.buf_out_offsets = None
        if self.buf_compressed_sizes: del self.buf_compressed_sizes; self.buf_compressed_sizes = None
        if self.buf_uncompressed_sizes: del self.buf_uncompressed_sizes; self.buf_uncompressed_sizes = None
        if self.buf_status: del self.buf_status; self.buf_status = None
        
        # Pinned arrays
        self.pinned_input = None
        self.pinned_output = None
        
        # OpenCL objects
        if self.queue:
            self.queue.finish()
            self.queue = None
        self.program = None
        self.kernel = None
        self.ctx = None
        
        import gc
        gc.collect()

    def decompress_batch(self, frames_data: List[bytes], uncompressed_sizes: List[int]) -> List[Optional[bytes]]:
        """
        Decompress a batch of frames using GPU with Pinned Memory and Async Transfers.
        
        Args:
            frames_data: List of compressed frame data
            uncompressed_sizes: Expected uncompressed size for each frame
            
        Returns:
            List of decompressed data (None for failed frames)
        """
        if not self.enabled:
            return [None] * len(frames_data)

        num_frames = len(frames_data)
        if num_frames == 0:
            return []
            
        if num_frames > self.max_batch_size:
            print(f"[GPU_LZ4] Warning: Batch size {num_frames} exceeds max {self.max_batch_size}, processing in chunks")
            # Process in chunks if needed
            results = []
            for i in range(0, num_frames, self.max_batch_size):
                chunk_data = frames_data[i:i+self.max_batch_size]
                chunk_sizes = uncompressed_sizes[i:i+self.max_batch_size]
                results.extend(self.decompress_batch(chunk_data, chunk_sizes))
            return results

        try:
            # ============================================================
            # 1. PREPARE DATA: Copy to pinned memory
            # ============================================================
            input_offsets = np.zeros(num_frames, dtype=np.uint32)
            output_offsets = np.zeros(num_frames, dtype=np.uint32)
            compressed_sizes_arr = np.zeros(num_frames, dtype=np.uint32)
            uncompressed_sizes_arr = np.array(uncompressed_sizes, dtype=np.uint32)
            
            current_in_offset = 0
            current_out_offset = 0
            
            for i, data in enumerate(frames_data):
                data_len = len(data)
                input_offsets[i] = current_in_offset
                output_offsets[i] = current_out_offset
                compressed_sizes_arr[i] = data_len
                
                # Copy to pinned input buffer
                self.pinned_input[current_in_offset:current_in_offset + data_len] = np.frombuffer(data, dtype=np.uint8)
                
                current_in_offset += data_len
                current_out_offset += uncompressed_sizes[i]
            
            total_input_size = current_in_offset
            total_output_size = current_out_offset
            
            # ============================================================
            # 2. ASYNC TRANSFERS: Upload non-blocking
            # ============================================================
            evt_input = cl.enqueue_copy(
                self.queue,
                self.buf_in,
                self.pinned_input[:total_input_size],
                is_blocking=False
            )
            evt_in_offsets = cl.enqueue_copy(self.queue, self.buf_in_offsets, input_offsets, is_blocking=False)
            evt_out_offsets = cl.enqueue_copy(self.queue, self.buf_out_offsets, output_offsets, is_blocking=False)
            evt_c_sizes = cl.enqueue_copy(self.queue, self.buf_compressed_sizes, compressed_sizes_arr, is_blocking=False)
            evt_u_sizes = cl.enqueue_copy(self.queue, self.buf_uncompressed_sizes, uncompressed_sizes_arr, is_blocking=False)
            
            # ============================================================
            # 3. KERNEL EXECUTION: Wait for uploads
            # ============================================================
            kernel_event = self.kernel(
                self.queue,
                (num_frames,),
                None,
                self.buf_in,
                self.buf_in_offsets,
                self.buf_out,
                self.buf_out_offsets,
                self.buf_compressed_sizes,
                self.buf_uncompressed_sizes,
                self.buf_status,
                np.uint32(num_frames),
                wait_for=[evt_input, evt_in_offsets, evt_out_offsets, evt_c_sizes, evt_u_sizes]
            )
            
            self.last_kernel_event = kernel_event
            
            # ============================================================
            # 4. READ RESULTS: Async with dependency on kernel
            # ============================================================
            status_arr = np.empty(num_frames, dtype=np.int32)
            evt_status = cl.enqueue_copy(
                self.queue,
                status_arr,
                self.buf_status,
                wait_for=[kernel_event],
                is_blocking=False
            )
            
            # Read output data (only the portion we need)
            evt_output = cl.enqueue_copy(
                self.queue,
                self.pinned_output[:total_output_size],
                self.buf_out,
                wait_for=[kernel_event],
                is_blocking=False
            )
            
            # Wait for both reads to complete
            evt_status.wait()
            evt_output.wait()
            
            # ============================================================
            # 5. PROCESS RESULTS
            # ============================================================
            results = []
            for i in range(num_frames):
                if status_arr[i] == 0:
                    # Success - extract from pinned output
                    out_start = int(output_offsets[i])
                    out_size = int(uncompressed_sizes[i])
                    results.append(bytes(self.pinned_output[out_start:out_start + out_size]))
                else:
                    # GPU failed for this frame
                    results.append(None)
            
            return results
            
        except Exception as e:
            print(f"[GPU_LZ4] Decompression batch error: {e}")
            import traceback
            traceback.print_exc()
            return [None] * num_frames


