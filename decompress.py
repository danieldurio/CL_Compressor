def lz_decompress(data: bytes, expected_size: int) -> bytes:
    """
    Descompressor para o formato LZ simples usado no compressor LZ.

    Formato de entrada:
        repetição de tokens:
            - Literal:
                0x00 | L (1..255) | L bytes
            - Match:
                0x01 | offset_lo | offset_hi | length (3..255)

    :param data: stream de tokens.
    :param expected_size: tamanho total esperado do frame descomprimido.
    :return: bytes do frame original.
    """
    out = bytearray()
    i = 0
    n = len(data)

    while i < n and len(out) < expected_size:
        if i >= n:
            raise RuntimeError("[Decompress] Fim inesperado do stream LZ (tipo).")

        token_type = data[i]
        i += 1

        if token_type == 0x00:
            # Literal
            if i >= n:
                raise RuntimeError("[Decompress] Fim inesperado do stream LZ (len literal).")
            length = data[i]
            i += 1
            if length == 0:
                raise RuntimeError("[Decompress] Literal com length=0.")
            if i + length > n:
                raise RuntimeError("[Decompress] Fim inesperado do stream LZ (dados literal).")

            out.extend(data[i:i + length])
            i += length

        elif token_type == 0x01:
            # Match
            if i + 3 > n:
                raise RuntimeError("[Decompress] Fim inesperado do stream LZ (match header).")

            offset = data[i] | (data[i + 1] << 8)
            i += 2
            length = data[i]
            i += 1

            if offset == 0:
                raise RuntimeError("[Decompress] Match com offset=0.")
            if length < 3:
                raise RuntimeError("[Decompress] Match com length < 3.")

            start = len(out) - offset
            if start < 0:
                raise RuntimeError("[Decompress] Match aponta para antes do início do buffer.")

            for _ in range(length):
                if start >= len(out):
                    raise RuntimeError("[Decompress] Match ultrapassa o buffer atual.")
                out.append(out[start])
                start += 1

        else:
            raise RuntimeError(f"[Decompress] Tipo de token LZ inválido: {token_type}.")

    if len(out) != expected_size:
        raise RuntimeError(
            f"[Decompress] Tamanho LZ diferente do esperado: {len(out)} vs {expected_size}."
        )

    return bytes(out)
