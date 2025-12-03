"""
Descompressor para o formato archive_final.gpu (Dedup + Dict++ + Patterns).

Suporta:
- Deduplicação (copia de arquivos originais)
- RLE2 (Inverse Map + Inverse RLE)
- Pattern+RLE (Inverse Pattern + Inverse RLE)
- RAW

Uso:
    python decompressor_dedup.py F:\\saida_final.gpu.index.json -o F:\\restaurado
"""

from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import struct

def rle_decode(data: bytes) -> bytes:
    """
    Decodifica RLE do GPUBlockRLECompressor.
    Formato: Sequência de trios [count_lo, count_hi, symbol]
    - count: uint16 (little endian)
    - symbol: uint8
    """
    out = bytearray()
    i = 0
    n = len(data)
    
    if n % 3 != 0:
        print(f"Aviso: Dados RLE com tamanho {n} não é múltiplo de 3. Pode estar corrompido.")
    
    while i + 2 < n:
        # Ler count (uint16 LE)
        count = data[i] | (data[i+1] << 8)
        symbol = data[i+2]
        i += 3
        
        if count > 0:
            out.extend(bytes([symbol]) * count)
            
    return bytes(out)

def map_decode(data: bytes, dictionary: List[int]) -> bytes:
    """
    Reverte o mapeamento de bytes.
    data: bytes mapeados (0 = mais frequente, 1 = segundo...)
    dictionary: lista de bytes originais ordenados por frequência.
    """
    # Criar tabela de lookup inversa?
    # O valor 'v' no data mapeia para dictionary[v].
    # Se v >= len(dictionary), então v mapeia para v (se não foi mapeado?)
    # O GPUByteMapper mapeia TODOS os 256 bytes.
    # O dictionary tem 'top_bytes'. Os que não estão no top_bytes?
    # O preprocessor.py constrói top_bytes.
    # O GPUByteMapper constrói um mapa completo 0..255.
    # Os bytes que estão no dicionário ganham índices 0..N.
    # Os que NÃO estão, ganham índices N..255 (em ordem original).
    
    # Reconstruir o mapa completo
    full_map = list(dictionary)
    used = set(dictionary)
    
    # Preencher o resto
    for b in range(256):
        if b not in used:
            full_map.append(b)
            
    # Agora full_map[i] é o byte original para o símbolo i
    # Traduzir
    return bytes([full_map[b] for b in data])

def pattern_decode(data: bytes, patterns: List[bytes]) -> bytes:
    """
    Decodifica stream com tokens de padrão.
    0x00..0xFE: Índice do padrão
    0xFF: Escape (próximo byte é literal)
    """
    out = bytearray()
    i = 0
    n = len(data)
    
    while i < n:
        token = data[i]
        i += 1
        
        if token == 0xFF:
            # Literal
            if i < n:
                out.append(data[i])
                i += 1
        elif token < len(patterns):
            # Padrão
            out.extend(patterns[token])
        else:
            # Token inválido (ou padrão não existente), trata como literal?
            # Deveria ser erro, mas vamos copiar
            out.append(token)
            
    return bytes(out)

def delta_decode(data: bytes) -> bytes:
    """
    Reverte Delta Encoding.
    out[0] = in[0]
    out[i] = out[i-1] + in[i]
    """
    try:
        import numpy as np
        arr_in = np.frombuffer(data, dtype=np.uint8)
        # cumsum reverte a diferença
        # Mas cuidado com overflow/wrap-around. O numpy cumsum padrão promove para int maior.
        # Precisamos de cumsum em uint8 com wrap-around.
        # np.cumsum(dtype=uint8) não faz wrap-around como esperado em algumas versões?
        # Teste rápido: np.array([1, 255], dtype=uint8).cumsum(dtype=uint8) -> [1, 0] (correto 1+255=256=0)
        
        # O problema é que diff faz arr[i] - arr[i-1].
        # Se arr[i] < arr[i-1], dá negativo -> wrap around.
        # Na volta, somamos.
        
        res = np.cumsum(arr_in, dtype=np.uint8)
        return res.tobytes()
    except ImportError:
        # Fallback python puro
        out = bytearray(len(data))
        if not data: return bytes(out)
        
        val = 0
        for i, b in enumerate(data):
            val = (val + b) & 0xFF
            out[i] = val
        return bytes(out)

def main() -> int:
    parser = argparse.ArgumentParser(description="Descompressor Dedup + Patterns (Embedded Index)")
    parser.add_argument("archive_base", help="Caminho base do arquivo (ex: F:\\saida_final ou F:\\saida_final.gpu.001)")
    parser.add_argument("-o", "--output", required=True, help="Pasta de destino")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Tentar descobrir o último volume
    # O usuário pode passar "saida_final" ou "saida_final.gpu.001"
    base_path = Path(args.archive_base).resolve()
    
    # Se passou um arquivo existente, tenta achar a base
    # Ex: saida.gpu.001 -> base = saida.gpu
    # Mas o padrão é saida.gpu.001, .002...
    
    # Vamos procurar todos os volumes .001, .002 na pasta e pegar o último.
    # Assumindo que o nome segue padrão <nome>.<num>
    
    parent_dir = base_path.parent
    name_stem = base_path.name
    
    # Se o usuário passou o .001, remove a extensão numérica
    import re
    match = re.match(r"(.*)\.(\d{3})$", name_stem)
    if match:
        prefix = match.group(1) # saida.gpu
    else:
        prefix = name_stem # saida.gpu (se passou a base)
        
    # Listar volumes
    volumes = sorted(parent_dir.glob(f"{prefix}.*"))
    # Filtrar apenas numéricos
    volumes = [v for v in volumes if re.match(r".*\.\d{3}$", v.name)]
    
    if not volumes:
        print(f"Erro: Nenhum volume encontrado com prefixo '{prefix}' em {parent_dir}")
        return 1
        
    last_vol = volumes[-1]
    print(f"Volumes encontrados: {len(volumes)}. Último: {last_vol.name}")
    
    # Ler Footer do último volume
    # [Offset (8)] [Size (8)] [Magic (8)] = 24 bytes
    import struct
    import zlib
    
    with open(last_vol, "rb") as f:
        f.seek(0, 2) # Fim
        file_size = f.tell()
        if file_size < 24:
            print("Erro: Arquivo muito pequeno para conter footer.")
            return 1
            
        f.seek(-24, 2)
        footer = f.read(24)
        
        offset, size, magic = struct.unpack('<QQ8s', footer)
        
        if magic != b'GPU_IDX1':
            print(f"Erro: Assinatura inválida no footer: {magic}")
            print("Este arquivo pode não conter um índice embutido ou é uma versão antiga.")
            return 1
            
        print(f"Índice encontrado: Offset={offset}, Size={size}")
        
        f.seek(offset)
        compressed_index = f.read(size)
        
    try:
        index_bytes = zlib.decompress(compressed_index)
        index = json.loads(index_bytes.decode('utf-8'))
        print("Índice carregado e descomprimido com sucesso.")
    except Exception as e:
        print(f"Erro ao descomprimir/ler índice: {e}")
        return 1
        
    # Carregar metadados
    files = index["files"]
    frames = index["frames"]
    params = index["params"]
    
    # Reconstruir dicionário
    dict_data = index["dictionary"]
    top_bytes = dict_data["top_bytes"]
    
    # Reconstruir dicionário de padrões
    patterns = []
    if "pattern_dictionary" in params:
        # Formato: {"pattern_length": 4, "patterns": ["hex", ...]}
        p_dict = params["pattern_dictionary"]
        patterns = [bytes.fromhex(p) for p in p_dict["patterns"]]
        print(f"Padrões carregados: {len(patterns)}")
        if patterns:
            print(f"DEBUG: Pattern[0] len={len(patterns[0])} content={patterns[0].hex()}")
            print(f"DEBUG: Pattern Length param: {p_dict.get('pattern_length')}")
        
    # Mapa de frames por ID
    frames_map = {f["frame_id"]: f for f in frames}
    
    # Modos de compressão por frame
    frame_modes = params.get("frame_modes", {})
    # Se frame_modes for string (chaves JSON são strings), converter chaves para int
    frame_modes = {int(k): v for k, v in frame_modes.items()}
    
    # Cache de volumes abertos
    volumes = {}
    base_dir = parent_dir
    
    print(f"Restaurando {len(files)} arquivos para {output_dir}...")
    
    current_file_idx = 0
    current_file_pos = 0 # Quanto já escrevemos no arquivo atual
    
    # Abrir primeiro arquivo
    if not files:
        print("Nenhum arquivo no índice.")
        return 0
        
    curr_file_entry = files[0]
    curr_fp = None
    
    # Função para escrever dados no arquivo atual (e pular para próximos se encher)
    def write_data(data: bytes):
        nonlocal current_file_idx, current_file_pos, curr_fp, curr_file_entry
        
        data_pos = 0
        data_len = len(data)
        
        while data_pos < data_len:
            # Se arquivo atual acabou ou é duplicata (já tratado), avançar
            while current_file_idx < len(files):
                curr_file_entry = files[current_file_idx]
                
                # Se é duplicata, não escrevemos dados do stream nele (ele não está no stream!)
                if curr_file_entry.get("is_duplicate"):
                    # Restaurar duplicata agora
                    src_rel = curr_file_entry["original"]
                    dst_path = output_dir / curr_file_entry["path_rel"]
                    src_path = output_dir / src_rel
                    
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                    except FileNotFoundError:
                        print(f"Aviso: Original {src_rel} não encontrado para duplicata {curr_file_entry['path_rel']}")
                        
                    current_file_idx += 1
                    current_file_pos = 0
                    continue
                
                # Arquivo normal (não duplicata)
                remaining = curr_file_entry["size"] - current_file_pos
                if remaining > 0:
                    break # Precisa de dados
                
                # Arquivo cheio, fechar e ir pro próximo
                if curr_fp:
                    curr_fp.close()
                    curr_fp = None
                
                current_file_idx += 1
                current_file_pos = 0
            
            if current_file_idx >= len(files):
                break # Acabaram os arquivos
                
            # Abrir arquivo se necessário
            if curr_fp is None:
                p = output_dir / curr_file_entry["path_rel"]
                p.parent.mkdir(parents=True, exist_ok=True)
                curr_fp = open(p, "wb")
            
            # Escrever o que cabe
            remaining = curr_file_entry["size"] - current_file_pos
            chunk_size = min(data_len - data_pos, remaining)
            
            curr_fp.write(data[data_pos : data_pos + chunk_size])
            
            current_file_pos += chunk_size
            data_pos += chunk_size
            
            # Se encheu, fecha (o loop vai abrir o próximo na próxima iteração)
            if current_file_pos >= curr_file_entry["size"]:
                curr_fp.close()
                curr_fp = None
                current_file_idx += 1
                current_file_pos = 0

    # Iterar frames e descomprimir
    sorted_frames = sorted(frames, key=lambda x: x["frame_id"])
    
    for fr in sorted_frames:
        fid = fr["frame_id"]
        vol_name = fr["volume_name"]
        offset = fr["offset"]
        c_size = fr["compressed_size"]
        u_size = fr["uncompressed_size"]
        
        # Ler dados comprimidos
        vol_path = base_dir / vol_name
        if vol_path not in volumes:
            pass
            
        with open(vol_path, "rb") as f:
            f.seek(offset)
            c_data = f.read(c_size)
            
        # Descobrir modo
        mode = frame_modes.get(fid, "raw") # Default raw se não achar
        
        # Decodificar
        if mode == "raw":
            u_data = c_data
        elif mode == "rle1":
            u_data = rle_decode(c_data)
        elif mode == "rle2":
            # RLE -> Map Decode
            rle_out = rle_decode(c_data)
            u_data = map_decode(rle_out, top_bytes)
        elif mode == "pattern_rle":
            # RLE -> Pattern Decode
            rle_out = rle_decode(c_data)
            u_data = pattern_decode(rle_out, patterns)
        elif mode == "delta_rle":
            # RLE -> Delta Decode
            rle_out = rle_decode(c_data)
            u_data = delta_decode(rle_out)
        else:
            print(f"Modo desconhecido {mode} no frame {fid}, tratando como RAW")
            u_data = c_data
            
        if len(u_data) != u_size:
            print(f"\n[ERRO] Frame {fid} ({mode}):")
            print(f"       Esperado: {u_size}")
            print(f"       Obtido:   {len(u_data)}")
            print(f"       Comp:     {len(c_data)}")
            if 'rle_out' in locals():
                print(f"       RLE Out:  {len(rle_out)}")
                print(f"       RLE Head: {rle_out[:20].hex()}")
            # print(f"       Head:     {u_data[:20].hex()}")

            
        # Escrever no fluxo de arquivos
        write_data(u_data)
        
        if fid % 10 == 0:
            print(f"Processado Frame {fid}...", end="\r")
            
    print("\nDescompressão concluída.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
