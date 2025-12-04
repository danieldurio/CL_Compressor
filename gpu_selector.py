"""
GPU Selector - Módulo de Seleção Interativa de GPUs

Permite ao usuário escolher quais GPUs usar antes de iniciar
operações de compressão/descompressão.

Uso:
    from gpu_selector import prompt_gpu_selection, get_filtered_devices
    
    excluded = prompt_gpu_selection()
    devices = get_filtered_devices(excluded)
"""

from typing import List, Dict, Tuple, Optional

# Cache global para índices excluídos (persiste durante a sessão)
_excluded_indices: Optional[List[int]] = None
_selection_done: bool = False


def list_available_gpus() -> List[Dict]:
    """
    Lista todas as GPUs OpenCL disponíveis com informações detalhadas.
    
    Returns:
        Lista de dicionários com informações de cada GPU:
        [
            {
                "index": 0,
                "name": "NVIDIA T1000 8GB",
                "platform": "NVIDIA CUDA",
                "vendor": "NVIDIA Corporation",
                "vram_mb": 8192,
                "compute_units": 14
            },
            ...
        ]
    """
    try:
        import pyopencl as cl
    except ImportError:
        return []
    
    try:
        platforms = cl.get_platforms()
        if not platforms:
            return []
        
        gpus = []
        global_index = 0
        
        for platform in platforms:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
            except:
                continue
            
            for device in devices:
                gpu_info = {
                    "index": global_index,
                    "name": device.name.strip(),
                    "platform": platform.name.strip(),
                    "vendor": platform.vendor.strip(),
                    "vram_mb": device.global_mem_size // (1024 * 1024),
                    "compute_units": device.max_compute_units
                }
                gpus.append(gpu_info)
                global_index += 1
        
        return gpus
        
    except Exception as e:
        print(f"[GPU Selector] Erro ao listar GPUs: {e}")
        return []


def print_gpu_list(gpus: List[Dict]) -> None:
    """Imprime lista formatada de GPUs disponíveis."""
    print("\n" + "="*60)
    print("GPUs DISPONÍVEIS")
    print("="*60)
    
    for gpu in gpus:
        print(f"  [{gpu['index']}] {gpu['name']}")
        print(f"      Plataforma: {gpu['platform']}")
        print(f"      VRAM: {gpu['vram_mb']} MB | CUs: {gpu['compute_units']}")
    
    print("="*60)


def prompt_gpu_selection(force_prompt: bool = False) -> List[int]:
    """
    Prompt interativo para seleção de GPUs.
    
    O prompt só aparece se:
    - Houver mais de 1 GPU disponível
    - A seleção ainda não foi feita nesta sessão (ou force_prompt=True)
    
    Args:
        force_prompt: Se True, força o prompt mesmo se já foi feito
        
    Returns:
        Lista de índices de GPUs a DESATIVAR (excluir)
    """
    global _excluded_indices, _selection_done
    
    # Retornar cache se já foi feita a seleção
    if _selection_done and not force_prompt:
        return _excluded_indices if _excluded_indices else []
    
    gpus = list_available_gpus()
    
    # Sem GPUs ou apenas 1 GPU - sem necessidade de prompt
    if len(gpus) <= 1:
        _excluded_indices = []
        _selection_done = True
        if len(gpus) == 1:
            print(f"[GPU Selector] GPU única detectada: {gpus[0]['name']}")
        return []
    
    # Mostrar GPUs disponíveis
    print_gpu_list(gpus)
    
    # Perguntar se deseja usar todas
    while True:
        try:
            response = input("\nUsar todas as GPUs? (s/n): ").strip().lower()
            if response in ['s', 'sim', 'y', 'yes', '']:
                _excluded_indices = []
                _selection_done = True
                print("[GPU Selector] Usando todas as GPUs disponíveis.")
                return []
            elif response in ['n', 'nao', 'não', 'no']:
                break
            else:
                print("Resposta inválida. Digite 's' para sim ou 'n' para não.")
        except (EOFError, KeyboardInterrupt):
            print("\n[GPU Selector] Operação cancelada. Usando todas as GPUs.")
            _excluded_indices = []
            _selection_done = True
            return []
    
    # Solicitar índices a desativar
    max_index = len(gpus) - 1
    
    while True:
        try:
            indices_str = input(f"Insira os índices das GPUs a DESATIVAR (0-{max_index}, separados por vírgula): ").strip()
            
            if not indices_str:
                print("[GPU Selector] Nenhum índice informado. Usando todas as GPUs.")
                _excluded_indices = []
                _selection_done = True
                return []
            
            # Parse dos índices
            excluded = []
            parts = indices_str.replace(' ', '').split(',')
            
            for part in parts:
                if not part:
                    continue
                try:
                    idx = int(part)
                    if 0 <= idx <= max_index:
                        if idx not in excluded:
                            excluded.append(idx)
                    else:
                        print(f"  Aviso: Índice {idx} ignorado (fora do range 0-{max_index})")
                except ValueError:
                    print(f"  Aviso: '{part}' não é um número válido, ignorado.")
            
            # Verificar se não excluiu todas
            if len(excluded) >= len(gpus):
                print("Erro: Você não pode desativar TODAS as GPUs!")
                print("Pelo menos uma GPU deve permanecer ativa, ou será usado fallback CPU.")
                continue
            
            # Confirmar exclusões
            if excluded:
                print("\n[GPU Selector] GPUs a serem DESATIVADAS:")
                for idx in excluded:
                    gpu = gpus[idx]
                    print(f"  [{idx}] {gpu['name']}")
                
                print("\n[GPU Selector] GPUs ATIVAS:")
                for gpu in gpus:
                    if gpu['index'] not in excluded:
                        print(f"  [{gpu['index']}] {gpu['name']}")
                
                confirm = input("\nConfirmar seleção? (s/n): ").strip().lower()
                if confirm in ['s', 'sim', 'y', 'yes', '']:
                    _excluded_indices = excluded
                    _selection_done = True
                    return excluded
                else:
                    print("Seleção cancelada. Tente novamente.\n")
                    continue
            else:
                _excluded_indices = []
                _selection_done = True
                return []
                
        except (EOFError, KeyboardInterrupt):
            print("\n[GPU Selector] Operação cancelada. Usando todas as GPUs.")
            _excluded_indices = []
            _selection_done = True
            return []


def get_filtered_devices(excluded_indices: Optional[List[int]] = None):
    """
    Retorna dispositivos OpenCL filtrados (excluindo os índices especificados).
    
    Args:
        excluded_indices: Lista de índices a excluir. 
                         Se None, usa o cache da última seleção.
    
    Returns:
        Lista de dispositivos OpenCL (cl.Device) filtrados
    """
    try:
        import pyopencl as cl
    except ImportError:
        return []
    
    if excluded_indices is None:
        excluded_indices = _excluded_indices if _excluded_indices else []
    
    try:
        platforms = cl.get_platforms()
        if not platforms:
            return []
        
        devices = []
        global_index = 0
        
        for platform in platforms:
            try:
                platform_devices = platform.get_devices(device_type=cl.device_type.GPU)
            except:
                continue
            
            for device in platform_devices:
                if global_index not in excluded_indices:
                    devices.append(device)
                global_index += 1
        
        return devices
        
    except Exception as e:
        print(f"[GPU Selector] Erro ao filtrar dispositivos: {e}")
        return []


def get_enabled_device_indices(excluded_indices: Optional[List[int]] = None) -> List[int]:
    """
    Retorna índices dos dispositivos habilitados.
    
    Args:
        excluded_indices: Lista de índices a excluir.
                         Se None, usa o cache da última seleção.
    
    Returns:
        Lista de índices de GPUs habilitadas
    """
    gpus = list_available_gpus()
    
    if excluded_indices is None:
        excluded_indices = _excluded_indices if _excluded_indices else []
    
    return [gpu['index'] for gpu in gpus if gpu['index'] not in excluded_indices]


def reset_selection() -> None:
    """Reseta o cache de seleção, permitindo novo prompt."""
    global _excluded_indices, _selection_done
    _excluded_indices = None
    _selection_done = False


def get_excluded_indices() -> List[int]:
    """Retorna a lista atual de índices excluídos (do cache)."""
    return _excluded_indices if _excluded_indices else []


# Convenience function para uso externo
def ensure_gpu_selection() -> List[int]:
    """
    Garante que a seleção de GPU foi feita.
    Chama o prompt se ainda não foi feito.
    
    Returns:
        Lista de índices excluídos
    """
    if not _selection_done:
        return prompt_gpu_selection()
    return _excluded_indices if _excluded_indices else []
