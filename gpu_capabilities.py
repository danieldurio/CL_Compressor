import pyopencl as cl

def get_recommended_batch_size(frame_size_mb: int = 16) -> int:
    """
    Calcula o batch size recomendado baseado nas GPUs disponíveis.
    
    Estratégia:
    - Detecta todas as GPUs OpenCL disponíveis
    - Calcula a "Recomendação Conservadora" para cada GPU
    - Retorna 2/3 do menor valor (caso múltiplas GPUs)
    - Fallback: 8 frames se nenhuma GPU for detectada
    
    Args:
        frame_size_mb: Tamanho do frame em MB (padrão: 16MB)
        
    Returns:
        Batch size recomendado (int)
    """
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("[GPU Capabilities] Nenhuma plataforma OpenCL encontrada. Usando batch size padrão: 8")
            return 8
        
        min_conservative_recommendation = None
        
        for p in platforms:
            try:
                devices = p.get_devices(device_type=cl.device_type.GPU)
            except:
                continue  # Ignorar plataformas sem GPUs
                
            if not devices:
                continue
                
            for d in devices:
                # Compute Units
                cus = d.max_compute_units
                
                # Memória Global
                mem_global = d.global_mem_size
                mem_global_mb = mem_global / 1024 / 1024
                
                # Heurística: mesma lógica do print_gpu_info()
                ideal_threads_min = cus * 4
                safe_mem_mb = mem_global_mb * 0.8
                max_frames_mem = safe_mem_mb / frame_size_mb
                
                # Recomendação Conservadora
                conservative_recommendation = int(min(ideal_threads_min, max_frames_mem))
                
                # Encontrar o menor valor entre todas as GPUs (safe)
                if min_conservative_recommendation is None:
                    min_conservative_recommendation = conservative_recommendation
                else:
                    min_conservative_recommendation = min(min_conservative_recommendation, conservative_recommendation)
        
        if min_conservative_recommendation is None:
            # Nenhuma GPU detectada
            print("[GPU Capabilities] Nenhuma GPU detectada. Usando batch size padrão: 8")
            return 8
        
        # Retornar 2/3 do valor conservador (arredondado para baixo)
        # Garantir mínimo de 4 frames
        recommended = max(4, int(min_conservative_recommendation * 2 / 3))
        
        return recommended
        
    except Exception as e:
        print(f"[GPU Capabilities] Erro ao calcular batch size: {e}. Usando padrão: 8")
        return 8

def print_gpu_info():
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("Nenhuma plataforma OpenCL encontrada.")
            return

        print("="*60)
        print("ANÁLISE DE CAPACIDADE GPU (OpenCL)")
        print("="*60)

        for p in platforms:
            print(f"Plataforma: {p.name} ({p.vendor})")
            
            devices = p.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                print("  [Sem GPUs nesta plataforma]")
                continue
                
            for d in devices:
                print(f"\n  Dispositivo: {d.name}")
                
                # Cores (Compute Units)
                cus = d.max_compute_units
                print(f"    - Compute Units (CUs): {cus}")
                
                # Clock
                clock = d.max_clock_frequency
                print(f"    - Clock Máximo: {clock} MHz")
                
                # Memória
                mem_global = d.global_mem_size
                mem_global_mb = mem_global / 1024 / 1024
                print(f"    - VRAM Global: {mem_global_mb:.0f} MB")
                
                # Max Work Group (Threads por bloco)
                max_wg = d.max_work_group_size
                print(f"    - Max Threads por Grupo: {max_wg}")
                
                # Heurística de Batch Size
                # LZ4 Kernel usa 1 thread por frame.
                # Para saturar a GPU, queremos ter pelo menos (CUs * Multiplicador) threads ativas.
                # GPUs modernas suportam milhares de threads em voo para esconder latência.
                # Um bom chute inicial é tentar ocupar todos os CUs com pelo menos 4-8 threads cada.
                
                ideal_threads_min = cus * 4
                ideal_threads_max = cus * 32
                
                # Limite de memória (Frame de 16MB)
                frame_size_mb = 16
                # Deixar 20% livre ou 1GB livre
                safe_mem_mb = mem_global_mb * 0.8
                max_frames_mem = safe_mem_mb / frame_size_mb
                
                print(f"    --------------------------------------------------")
                print(f"    [Estimativa para Batch Size - Frames de {frame_size_mb}MB]")
                print(f"    - Capacidade de Processamento (Threads): {ideal_threads_min} a {ideal_threads_max}")
                print(f"    - Capacidade de Memória (VRAM): ~{max_frames_mem:.0f} frames")
                
                recommended = min(ideal_threads_max, max_frames_mem)
                print(f"    -> Recomendação Teórica: {int(recommended)} frames (se couber na VRAM)")
                print(f"    -> Recomendação Conservadora: {int(min(ideal_threads_min, max_frames_mem))} frames")
                print(f"    --------------------------------------------------")

    except Exception as e:
        print(f"Erro ao consultar GPUs: {e}")

if __name__ == "__main__":
    print_gpu_info()
