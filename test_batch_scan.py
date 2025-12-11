import os
import sys
import shutil
import time
from pathlib import Path

sys.path.append(os.getcwd())
try:
    import iotools
    import config_loader
except ImportError:
    sys.path.append(r"e:\TheStorage\runtime\compressor")
    import iotools
    import config_loader

def create_structure(base_dir):
    print(f"Criando {base_dir}...")
    if base_dir.exists(): shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True)
    total = 0
    # ~250 pastas (para testar batches de 50)
    for i in range(50):
        d = base_dir / f"root_{i}"
        d.mkdir()
        (d / "f.txt").write_text("ok")
        total += 1
        for j in range(4):
            SUB = d / f"sub_{j}"
            SUB.mkdir()
            (SUB/"f.txt").write_text("ok")
            total += 1
    return total

def main():
    test_dir = Path("tmp_batch_test")
    try:
        expected = create_structure(test_dir)
        
        # Force configs
        iotools.PRE_SCAN_TARGET_DIRS = 100 # Forçar split
        iotools.NUM_SCAN_WORKERS = 4
        # iotools.TASK_BATCH_SIZE é local, não editável fácil sem alterar source. 
        # Mas podemos ver no log "Batch Size: 50".
        
        print("\n--- Test Batch Scan ---")
        start = time.time()
        files = iotools.scan_directory_parallel(test_dir)
        dur = time.time() - start
        
        print(f"\nResult: {len(files)} files found (Expected: {expected}) in {dur:.3f}s")
        assert len(files) == expected
        print("[SUCCESS]")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        if test_dir.exists(): shutil.rmtree(test_dir)

if __name__ == "__main__":
    main()
