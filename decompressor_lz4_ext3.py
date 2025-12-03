"""
decompressor_lz4_ext3.py
Pure-Python LZ4 Decompressor for Extended Format (3-byte offsets)

Suporta:
- Offsets de 3 bytes (janela de 16MB)
- Formato lz_ext3_gpu gerado pelo GPU kernel customizado
"""

def decompress_lz4_ext3(compressed_data: bytes, uncompressed_size: int) -> bytes:
    """
    Descomprime dados LZ4 com offsets de 3 bytes.
    
    Args:
        compressed_data: Dados comprimidos pelo GPU kernel lz_ext3
        uncompressed_size: Tamanho esperado após descompressão
        
    Returns:
        Dados descomprimidos
        
    Raises:
        ValueError: Se os dados estiverem corrompidos
    """
    MIN_MATCH = 4
    
    output = bytearray(uncompressed_size)
    ip = 0  # Input position
    op = 0  # Output position
    input_len = len(compressed_data)
    
    while ip < input_len:
        # Ler token
        token = compressed_data[ip]
        ip += 1
        
        # Literal length (4 bits superiores)
        literal_len = token >> 4
        
        # Extended literal length
        if literal_len == 15:
            while ip < input_len:
                byte = compressed_data[ip]
                ip += 1
                literal_len += byte
                if byte != 255:
                    break
        
        # Copiar literais
        if literal_len > 0:
            if ip + literal_len > input_len:
                raise ValueError(f"Input overflow: ip={ip}, literal_len={literal_len}, input_len={input_len}")
            
            if op + literal_len > uncompressed_size:
                raise ValueError(f"Output overflow: op={op}, literal_len={literal_len}, uncompressed_size={uncompressed_size}")
            
            output[op:op + literal_len] = compressed_data[ip:ip + literal_len]
            op += literal_len
            ip += literal_len
        
        # Fim do stream?
        if ip >= input_len:
            break
        
        # Ler offset (3 bytes, little-endian)
        if ip + 3 > input_len:
            raise ValueError(f"Incomplete offset at ip={ip}")
        
        offset = (compressed_data[ip] | 
                 (compressed_data[ip + 1] << 8) | 
                 (compressed_data[ip + 2] << 16))
        ip += 3
        
        if offset == 0:
            raise ValueError(f"Invalid offset=0 at position {ip-3}")
        
        # Match length (4 bits inferiores)
        match_len = (token & 0x0F) + MIN_MATCH
        
        # Extended match length
        if (token & 0x0F) == 15:
            while ip < input_len:
                byte = compressed_data[ip]
                ip += 1
                match_len += byte
                if byte != 255:
                    break
        
        # Copiar match
        if offset > op:
            raise ValueError(f"Offset points before output start: offset={offset}, op={op}")
        
        match_pos = op - offset
        
        if op + match_len > uncompressed_size:
            raise ValueError(f"Match overflow: op={op}, match_len={match_len}, uncompressed_size={uncompressed_size}")
        
        # Copiar byte a byte para suportar overlapping copies (RLE)
        for i in range(match_len):
            output[op] = output[match_pos]
            op += 1
            match_pos += 1
    
    if op != uncompressed_size:
        raise ValueError(f"Uncompressed size mismatch: expected={uncompressed_size}, got={op}")
    
    return bytes(output)
