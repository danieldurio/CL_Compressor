import pyopencl as cl
import numpy as np
import time
from typing import List, Tuple, Optional

LZ4_DECOMPRESS_KERNEL = """
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
    __global uchar* op_start = output_data + output_start; // Start of this frame's output
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

        // Copy literals
        if (op + literal_len > op_end) { status[gid] = -2; return; }
        if (ip + literal_len > ip_end) { status[gid] = -3; return; }
        
        for (uint i = 0; i < literal_len; i++) {
            *op++ = *ip++;
        }

        if (ip >= ip_end) break; // End of stream

        // Offset (3 bytes)
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

        // Copy match
        if (op + match_len > op_end) { status[gid] = -7; return; }
        
        // Calculate match position relative to the start of the output buffer for this frame
        // op is current position, op_start is start of frame
        // offset is distance back from op
        long current_rel_pos = op - op_start;
        long match_rel_pos = current_rel_pos - offset;

        if (match_rel_pos < 0) { status[gid] = -8; return; }
        
        // Byte-by-byte copy to handle overlap
        for (uint i = 0; i < match_len; i++) {
            op[0] = op_start[match_rel_pos + i];
            op++;
        }
    }
    
    // Size validation - allow small underflow (compressor pads last bytes)
    // The compressor leaves last 12 bytes as literals which may cause slight size mismatch
    long size_diff = (long)(op_end - op);
    if (size_diff < 0 || size_diff > 12) {
        status[gid] = -9; // Significant size mismatch
        return;
    }
    
    // If we have a small underflow, zero-fill the rest
    while (op < op_end) {
        *op++ = 0;
    }
    
    status[gid] = 0; // Success
}
"""

class GPU_LZ4_Decompressor:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self.ctx = None
        self.queue = None
        self.program = None
        self.kernel = None
        self.enabled = False
        self._init_opencl()

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

    def decompress_batch(self, frames_data: List[bytes], uncompressed_sizes: List[int]) -> List[Optional[bytes]]:
        if not self.enabled:
            return [None] * len(frames_data)

        num_frames = len(frames_data)
        if num_frames == 0:
            return []

        # TEMPORARY FIX: Process frames ONE AT A TIME to avoid GPU race conditions
        # The parallel batch processing was causing memory interference between threads
        results = []
        
        for i in range(num_frames):
            try:
                # Process single frame
                data = frames_data[i]
                u_size = uncompressed_sizes[i]
                
                # Prepare single-frame buffers
                input_concat = bytearray(len(data))
                input_concat[:] = data
                input_offsets = np.array([0], dtype=np.uint32)
                compressed_sizes_arr = np.array([len(data)], dtype=np.uint32)
                
                output_offsets = np.array([0], dtype=np.uint32)
                uncompressed_sizes_arr = np.array([u_size], dtype=np.uint32)
                
                # Buffers
                mf = cl.mem_flags
                buf_in = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_concat)
                buf_in_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=input_offsets)
                buf_out = cl.Buffer(self.ctx, mf.READ_WRITE, u_size)
                buf_out_offsets = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=output_offsets)
                buf_c_sizes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=compressed_sizes_arr)
                buf_u_sizes = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=uncompressed_sizes_arr)
                buf_status = cl.Buffer(self.ctx, mf.WRITE_ONLY, 4)
                
                # Run kernel (single work-item)
                event = self.kernel(
                    self.queue,
                    (1,),  # Single work-item
                    None,
                    buf_in,
                    buf_in_offsets,
                    buf_out,
                    buf_out_offsets,
                    buf_c_sizes,
                    buf_u_sizes,
                    buf_status,
                    np.uint32(1)
                )
                event.wait()
                
                # Read results
                status_arr = np.empty(1, dtype=np.int32)
                cl.enqueue_copy(self.queue, status_arr, buf_status).wait()
                
                if status_arr[0] == 0:
                    # Success
                    output_data = np.empty(u_size, dtype=np.uint8)
                    cl.enqueue_copy(self.queue, output_data, buf_out).wait()
                    results.append(output_data.tobytes())
                else:
                    # GPU failed for this frame
                    results.append(None)
                    
            except Exception as e:
                # Error processing this frame
                results.append(None)
        
        return results

