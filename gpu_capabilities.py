import pyopencl as cl

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
