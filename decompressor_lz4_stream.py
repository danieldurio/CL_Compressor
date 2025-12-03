"""
Decompressor LZ4 GPU para Streams (stdout).

Descomprime volumes .gpu e escreve dados para stdout.
Permite uso em pipelines Unix/Windows.

Uso:
    # Descomprimir para arquivo
    python decompressor_lz4_stream.py -i arquivo.gpu.001 > saida.bin
    
    # Descomprimir tar diretamente
    python decompressor_lz4_stream.py -i backup.gpu.001 | tar xf -
    
    # Descomprimir e transferir via rede
    python decompressor_lz4_stream.py -i data.gpu.001 | ssh server "cat > data.bin"
    
    # Windows: redirecionar para arquivo
    python decompressor_lz4_stream.py -i arquivo.gpu.001 > saida.bin
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

from iotools import VolumeReader, FrameMeta, load_index_from_volume
from gpu_lz4_decompressor import GPU_LZ4_Decompressor

def decompress_to_stdout(input_base: Path) -> int:
    """
    Descomprime volumes e escreve para stdout.
    
    Args:
        input_base: Caminho base dos volumes (ex: arquivo.gpu.001)
        
    Returns:
        0 se sucesso, 1 se erro
    """
    # Configurar stdout para modo binário
    if os.name == 'nt':
        # Windows: forçar modo binário
        import msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    
    stdout_binary = sys.stdout.buffer
    
    # Carregar índice
    print(f"[Decompressor] Carregando índice de {input_base}...", file=sys.stderr)
    
    try:
        index_data = load_index_from_volume(input_base)
    except Exception as e:
        print(f"ERRO ao carregar índice: {e}", file=sys.stderr)
        return 1
    
    files = index_data.get("files", [])
    frames = index_data.get("frames", [])
    params = index_data.get("params", {})
    
    if not frames:
        print("ERRO: Nenhum frame encontrado no índice.", file=sys.stderr)
        return 1
    
    print(f"[Decompressor] Total de frames: {len(frames)}", file=sys.stderr)
    print(f"[Decompressor] Modo de compressão: {params.get('compression_mode', 'unknown')}", file=sys.stderr)
    
    # Determinar base do volume (remover .001, .002, etc)
    base_str = str(input_base)
    if base_str.endswith(('.001', '.002', '.003', '.004', '.005', '.006', '.007', '.008', '.009')):
        volume_base = Path(base_str[:-4])
    else:
        volume_base = input_base
    
    # Inicializar leitor de volumes
    reader = VolumeReader(volume_base)
    
    # Detectar GPUs para decompressão
    import pyopencl as cl
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        devices.extend(p.get_devices(device_type=cl.device_type.GPU))
    
    decompressor = None
    if devices:
        print(f"[Decompressor] Detectadas {len(devices)} GPUs. Usando GPU para decompressão.", file=sys.stderr)
        try:
            decompressor = GPU_LZ4_Decompressor(device_index=0)
            if decompressor.enabled:
                print(f"[GPU_LZ4] Decompressor ativado.", file=sys.stderr)
            else:
                decompressor = None
                print(f"[Decompressor] Falha ao inicializar GPU. Usando CPU fallback.", file=sys.stderr)
        except Exception as e:
            print(f"[Decompressor] Erro ao inicializar GPU: {e}. Usando CPU fallback.", file=sys.stderr)
            decompressor = None
    else:
        print("[Decompressor] Nenhuma GPU detectada. Usando CPU fallback.", file=sys.stderr)
    
    # Obter frame_modes se disponível
    frame_modes = params.get("frame_modes", {})
    if isinstance(frame_modes, dict):
        # Converter keys para int se necessário
        frame_modes = {int(k) if isinstance(k, str) else k: v for k, v in frame_modes.items()}
    
    # Estatísticas
    total_frames = len(frames)
    total_bytes_written = 0
    lz4_frames = 0
    raw_frames = 0
    
    print(f"[Decompressor] Iniciando descompressão para stdout...", file=sys.stderr)
    print("-" * 70, file=sys.stderr)
    
    # Processar frames em ordem
    for i, frame_meta in enumerate(frames):
        # Determinar modo do frame
        frame_mode = frame_modes.get(frame_meta.frame_id, "unknown")
        
        # Ler dados comprimidos do volume
        try:
            compressed_data = reader.read_frame(frame_meta)
        except Exception as e:
            print(f"\nERRO ao ler frame {frame_meta.frame_id}: {e}", file=sys.stderr)
            return 1
        
        # Descomprimir se necessário
        if frame_mode == "lz_ext3_gpu" and decompressor:
            # Descompressão GPU
            try:
                uncompressed_data = decompressor.decompress_single(
                    compressed_data,
                    frame_meta.uncompressed_size
                )
                lz4_frames += 1
            except Exception as e:
                print(f"\nERRO ao descomprimir frame {frame_meta.frame_id} (GPU): {e}", file=sys.stderr)
                print("Tentando CPU fallback...", file=sys.stderr)
                # Fallback para LZ4 CPU
                import lz4.frame
                try:
                    uncompressed_data = lz4.frame.decompress(compressed_data)
                    lz4_frames += 1
                except Exception as e2:
                    print(f"ERRO CPU fallback: {e2}", file=sys.stderr)
                    return 1
        elif frame_mode == "lz_ext3_gpu":
            # Sem GPU, usar CPU
            import lz4.frame
            try:
                uncompressed_data = lz4.frame.decompress(compressed_data)
                lz4_frames += 1
            except Exception as e:
                print(f"\nERRO ao descomprimir frame {frame_meta.frame_id} (CPU): {e}", file=sys.stderr)
                return 1
        else:
            # RAW - não comprimido
            uncompressed_data = compressed_data
            raw_frames += 1
        
        # Validar tamanho
        if len(uncompressed_data) != frame_meta.uncompressed_size:
            print(f"\nERRO: Tamanho descomprimido incorreto no frame {frame_meta.frame_id}", file=sys.stderr)
            print(f"Esperado: {frame_meta.uncompressed_size}, Obtido: {len(uncompressed_data)}", file=sys.stderr)
            return 1
        
        # Escrever para stdout
        try:
            stdout_binary.write(uncompressed_data)
            stdout_binary.flush()
        except Exception as e:
            print(f"\nERRO ao escrever para stdout: {e}", file=sys.stderr)
            return 1
        
        total_bytes_written += len(uncompressed_data)
        
        # Progress feedback (stderr para não poluir stdout)
        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            progress_pct = ((i + 1) / total_frames) * 100
            mb_written = total_bytes_written / (1024 * 1024)
            print(f"\r[Progress] {i+1}/{total_frames} frames ({progress_pct:.1f}%) | {mb_written:.1f} MB escritos", 
                  end='', file=sys.stderr)
    
    print("", file=sys.stderr)  # Nova linha após progress
    print("-" * 70, file=sys.stderr)
    print(f"[Decompressor] Descompressão concluída!", file=sys.stderr)
    print(f"[Stats] LZ4 frames: {lz4_frames} | RAW frames: {raw_frames}", file=sys.stderr)
    print(f"[Stats] Total escrito: {total_bytes_written / (1024*1024):.2f} MB", file=sys.stderr)
    
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decompressor LZ4 GPU para Streams (stdout)",
        epilog="Exemplo: python decompressor_lz4_stream.py -i arquivo.gpu.001 > saida.bin"
    )
    parser.add_argument("-i", "--input", required=True, help="Volume de entrada (.gpu.001)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input).resolve()
    
    if not input_path.exists():
        print(f"ERRO: Arquivo não encontrado: {input_path}", file=sys.stderr)
        return 1
    
    print("=" * 70, file=sys.stderr)
    print("DECOMPRESSOR LZ4 GPU STREAM", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Entrada: {input_path}", file=sys.stderr)
    print(f"Saída: stdout (stream binário)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    return decompress_to_stdout(input_path)

if __name__ == "__main__":
    raise SystemExit(main())
