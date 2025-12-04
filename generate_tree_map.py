"""
Gerador de Mapa de Árvore (Tree Map) a partir de arquivos compactados.

Lê o footer de arquivos .001 (LZ4 GPU compressor) e gera um HTML standalone
com visualização hierárquica do uso de espaço (tree size) e estatísticas
de compressão.

Uso:
    python generate_tree_map.py arquivo.001 -o output.html

Sem dependências externas - usa apenas biblioteca padrão do Python.
"""

from __future__ import annotations
import struct
import zlib
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


def format_bytes(size: int) -> str:
    """Formata bytes para formato legível."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def read_footer(archive_path: Path) -> Optional[Dict[str, Any]]:
    """
    Lê o footer do arquivo compactado e retorna o índice descomprimido.
    
    Footer format (last 24 bytes):
    - 8 bytes: offset (Q, little-endian)
    - 8 bytes: size (Q, little-endian)
    - 8 bytes: magic signature (should be b'GPU_IDX1')
    """
    try:
        with open(archive_path, 'rb') as f:
            # Ir para o final do arquivo
            f.seek(0, 2)
            file_size = f.tell()
            
            if file_size < 24:
                print(f"Erro: Arquivo muito pequeno para conter footer ({file_size} bytes)")
                return None
            
            # Ler footer (últimos 24 bytes)
            f.seek(-24, 2)
            footer = f.read(24)
            
            # Decodificar footer
            offset, size, magic = struct.unpack('<QQ8s', footer)
            
            if magic != b'GPU_IDX1':
                print(f"Erro: Assinatura inválida no footer: {magic}")
                print("Este arquivo pode não ser um arquivo compactado válido.")
                return None
            
            print(f"✓ Footer encontrado: Offset={offset}, Size={size} bytes")
            
            # Ler índice comprimido
            f.seek(offset)
            compressed_index = f.read(size)
            
            # Descomprimir índice
            index_bytes = zlib.decompress(compressed_index)
            index = json.loads(index_bytes.decode('utf-8'))
            
            print(f"✓ Índice carregado: {len(index.get('files', []))} arquivos, {len(index.get('frames', []))} frames")
            
            return index
            
    except Exception as e:
        print(f"Erro ao ler footer: {e}")
        return None


class TreeNode:
    """Nó da árvore hierárquica de pastas."""
    
    def __init__(self, name: str):
        self.name = name
        self.size = 0
        self.total_items = 0
        self.total_files = 0
        self.total_folders = 0
        self.children: List[TreeNode] = []
        self.is_folder = True
    
    def add_file(self, path_parts: List[str], file_size: int):
        """Adiciona um arquivo à árvore."""
        if not path_parts:
            return
        
        if len(path_parts) == 1:
            # É um arquivo direto neste nível
            self.size += file_size
            self.total_items += 1
            self.total_files += 1
        else:
            # É uma pasta
            folder_name = path_parts[0]
            
            # Procurar ou criar child
            child = None
            for c in self.children:
                if c.name == folder_name:
                    child = c
                    break
            
            if child is None:
                child = TreeNode(folder_name)
                self.children.append(child)
            
            # Adicionar recursivamente
            child.add_file(path_parts[1:], file_size)
            
            # Acumular tamanhos e contagens
            self.size += file_size
            self.total_items += 1
    
    def finalize(self):
        """Finaliza a árvore calculando contagens de pastas."""
        for child in self.children:
            child.finalize()
            self.total_folders += child.total_folders
        
        if self.children:
            self.total_folders += len(self.children)
    
    def sort_children(self):
        """Ordena children por tamanho (maiores primeiro)."""
        self.children.sort(key=lambda x: x.size, reverse=True)
        for child in self.children:
            child.sort_children()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário para JSON."""
        return {
            'name': self.name,
            'size': self.size,
            'size_formatted': format_bytes(self.size),
            'total_items': self.total_items,
            'total_files': self.total_files,
            'total_folders': self.total_folders,
            'children': [c.to_dict() for c in self.children]
        }


def build_tree(files: List[Dict[str, Any]]) -> TreeNode:
    """Constrói árvore hierárquica a partir da lista de arquivos."""
    root = TreeNode("ROOT")
    
    for file_entry in files:
        # Ignorar duplicatas (já contabilizadas no original)
        if file_entry.get('is_duplicate'):
            continue
        
        path = file_entry['path_rel']
        size = file_entry['size']
        
        # Normalizar separadores e dividir path
        path = path.replace('\\', '/')
        parts = [p for p in path.split('/') if p]
        
        root.add_file(parts, size)
    
    root.finalize()
    root.sort_children()
    
    return root


def calculate_compression_stats(index: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula estatísticas de compressão do arquivo."""
    files = index.get('files', [])
    frames = index.get('frames', [])
    params = index.get('params', {})
    frame_modes = params.get('frame_modes', {})
    
    # Total original (soma dos tamanhos de todos os arquivos não-duplicados)
    total_original = sum(f['size'] for f in files if not f.get('is_duplicate'))
    
    # Total comprimido (soma dos compressed_size de todos os frames)
    total_compressed = sum(f['compressed_size'] for f in frames)
    
    # Contar volumes únicos
    volumes = set(f['volume_name'] for f in frames)
    
    # Contar por modo de compressão
    mode_counts = {}
    mode_original_sizes = {}
    mode_compressed_sizes = {}
    
    for frame in frames:
        fid = frame['frame_id']
        mode = frame_modes.get(str(fid), 'lz_ext3_gpu')
        
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        mode_compressed_sizes[mode] = mode_compressed_sizes.get(mode, 0) + frame['compressed_size']
        mode_original_sizes[mode] = mode_original_sizes.get(mode, 0) + frame['uncompressed_size']
    
    # Calcular ratios
    compression_ratio = (total_original / total_compressed) if total_compressed > 0 else 1.0
    space_saved = total_original - total_compressed
    space_saved_percent = (space_saved / total_original * 100) if total_original > 0 else 0
    
    return {
        'total_original': total_original,
        'total_original_formatted': format_bytes(total_original),
        'total_compressed': total_compressed,
        'total_compressed_formatted': format_bytes(total_compressed),
        'compression_ratio': compression_ratio,
        'space_saved': space_saved,
        'space_saved_formatted': format_bytes(space_saved),
        'space_saved_percent': space_saved_percent,
        'total_files': len([f for f in files if not f.get('is_duplicate')]),
        'total_duplicates': len([f for f in files if f.get('is_duplicate')]),
        'total_frames': len(frames),
        'volume_count': len(volumes),
        'volumes': sorted(volumes),
        'mode_counts': mode_counts,
        'mode_original_sizes': mode_original_sizes,
        'mode_compressed_sizes': mode_compressed_sizes,
    }


def generate_html(tree_data: Dict[str, Any], stats: Dict[str, Any], output_path: Path):
    """Gera HTML standalone com visualização da árvore e estatísticas."""
    
    html_template = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mapa de Árvore - {archive_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
        }}

        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 30px 0;
        }}

        .header h1 {{
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #4ecca3, #2a9d8f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .header p {{
            color: #999;
            font-size: 16px;
            margin-bottom: 15px;
        }}

        .section {{
            background: rgba(22, 33, 62, 0.8);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(78, 204, 163, 0.1);
        }}

        .section h2 {{
            margin-bottom: 24px;
            color: #fff;
            font-size: 24px;
            font-weight: 600;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.03);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid rgba(78, 204, 163, 0.1);
        }}

        .stat-label {{
            color: #999;
            font-size: 13px;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stat-value {{
            color: #4ecca3;
            font-size: 24px;
            font-weight: 700;
        }}

        .stat-value.big {{
            font-size: 28px;
        }}

        .compression-bar {{
            width: 100%;
            height: 30px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
            margin-top: 10px;
        }}

        .compression-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4ecca3, #2a9d8f);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 14px;
        }}

        .mode-list {{
            list-style: none;
        }}

        .mode-item {{
            padding: 12px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .mode-item:last-child {{
            border-bottom: none;
        }}

        .mode-name {{
            color: #ddd;
            font-weight: 500;
        }}

        .mode-count {{
            color: #999;
            font-size: 14px;
        }}

        .mode-ratio {{
            color: #4ecca3;
            font-weight: 600;
            margin-left: 15px;
        }}

        .tree-list {{
            list-style: none;
        }}

        .tree-item {{
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .tree-item-header {{
            display: block;
            padding: 12px 8px;
            cursor: pointer;
            transition: background 0.2s ease;
        }}

        .tree-item-header:hover {{
            background: rgba(78, 204, 163, 0.05);
        }}

        .tree-header-top {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
        }}

        .tree-toggle {{
            width: 20px;
            color: #4ecca3;
            font-weight: bold;
            flex-shrink: 0;
            user-select: none;
        }}

        .tree-icon {{
            width: 30px;
            font-size: 18px;
            flex-shrink: 0;
        }}

        .tree-name {{
            flex: 1;
            color: #ddd;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .tree-size {{
            width: 120px;
            color: #4ecca3;
            font-weight: 600;
            text-align: right;
            flex-shrink: 0;
        }}

        .tree-percent {{
            width: 80px;
            color: #999;
            text-align: right;
            font-size: 13px;
            flex-shrink: 0;
        }}

        .tree-items {{
            width: 100px;
            color: #999;
            text-align: right;
            font-size: 13px;
            flex-shrink: 0;
        }}

        .tree-progress-bar {{
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 4px;
        }}

        .tree-progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4ecca3, #2a9d8f);
            border-radius: 2px;
            transition: width 0.3s ease;
        }}

        .tree-children {{
            display: block;
            margin-left: 30px;
        }}

        .tree-children.collapsed {{
            display: none;
        }}

        .footer {{
            text-align: center;
            color: #666;
            margin-top: 50px;
            padding: 20px;
            font-size: 13px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .footer strong {{
            color: #4ecca3;
        }}

        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 32px;
            }}

            .tree-percent,
            .tree-items {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Mapa de Árvore do Arquivo</h1>
            <p>{archive_name}</p>
        </div>

        <!-- Estatísticas de Compressão -->
        <div class="section">
            <h2>📊 Estatísticas de Compressão</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Tamanho Original</div>
                    <div class="stat-value">{total_original_formatted}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Tamanho Comprimido</div>
                    <div class="stat-value">{total_compressed_formatted}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Razão de Compressão</div>
                    <div class="stat-value big">{compression_ratio:.2f}x</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Espaço Economizado</div>
                    <div class="stat-value">{space_saved_formatted}</div>
                    <div class="compression-bar">
                        <div class="compression-bar-fill" style="width: {space_saved_percent:.1f}%">
                            {space_saved_percent:.1f}%
                        </div>
                    </div>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Total de Arquivos</div>
                    <div class="stat-value">{total_files:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Arquivos Duplicados</div>
                    <div class="stat-value">{total_duplicates:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total de Frames</div>
                    <div class="stat-value">{total_frames:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Volumes</div>
                    <div class="stat-value">{volume_count}</div>
                </div>
            </div>
        </div>

        <!-- Breakdown por Modo de Compressão -->
        <div class="section">
            <h2>🔧 Breakdown por Modo de Compressão</h2>
            <ul class="mode-list">
                {mode_breakdown}
            </ul>
        </div>

        <!-- Resumo Geral -->
        <div class="section">
            <h2>📊 Resumo da Árvore</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Tamanho Total</div>
                    <div class="stat-value">{tree_size_formatted}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total de Itens</div>
                    <div class="stat-value">{tree_total_items:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Arquivos</div>
                    <div class="stat-value">{tree_total_files:,}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Pastas</div>
                    <div class="stat-value">{tree_total_folders:,}</div>
                </div>
            </div>
        </div>

        <!-- Árvore Hierárquica -->
        <div class="section">
            <h2>📁 Árvore Hierárquica (Maiores Primeiro)</h2>
            <p style="color: #999; font-size: 13px; margin-bottom: 15px;">
                💡 Clique em uma pasta para recolher/expandir. Barras verdes mostram proporção do espaço total.
            </p>
            <ul class="tree-list" id="tree-list"></ul>
        </div>

        <div class="footer">
            ⚡ Gerado por <strong>generate_tree_map.py</strong> | 
            Compressor: <strong>LZ4 GPU Dedup</strong>
        </div>
    </div>

    <script>
        const treeData = {tree_json};

        function renderTreeList(node, parentElement, depth) {{
            depth = depth || 0;
            const li = document.createElement('li');
            li.className = 'tree-item';

            const hasChildren = node.children && node.children.length > 0;

            const percentage = (node.size / treeData.size * 100).toFixed(1);

            const header = document.createElement('div');
            header.className = 'tree-item-header';
            header.style.paddingLeft = (depth * 30 + 8) + 'px';

            const headerTop = document.createElement('div');
            headerTop.className = 'tree-header-top';

            const toggle = document.createElement('span');
            toggle.className = 'tree-toggle';
            toggle.textContent = hasChildren ? '-' : '';
            headerTop.appendChild(toggle);

            const icon = document.createElement('span');
            icon.className = 'tree-icon';
            icon.textContent = '📁';
            headerTop.appendChild(icon);

            const name = document.createElement('span');
            name.className = 'tree-name';
            name.textContent = node.name;
            name.title = node.name;
            headerTop.appendChild(name);

            const percent = document.createElement('span');
            percent.className = 'tree-percent';
            percent.textContent = percentage + '%';
            headerTop.appendChild(percent);

            const items = document.createElement('span');
            items.className = 'tree-items';
            items.textContent = node.total_items.toLocaleString('pt-BR') + ' itens';
            headerTop.appendChild(items);

            const size = document.createElement('span');
            size.className = 'tree-size';
            size.textContent = node.size_formatted;
            headerTop.appendChild(size);

            header.appendChild(headerTop);

            const progressBar = document.createElement('div');
            progressBar.className = 'tree-progress-bar';
            const progressFill = document.createElement('div');
            progressFill.className = 'tree-progress-fill';
            progressFill.style.width = percentage + '%';
            progressBar.appendChild(progressFill);
            header.appendChild(progressBar);

            li.appendChild(header);

            if (hasChildren) {{
                const childrenContainer = document.createElement('ul');
                childrenContainer.className = 'tree-children';

                node.children.forEach(function (child) {{
                    renderTreeList(child, childrenContainer, depth + 1);
                }});

                li.appendChild(childrenContainer);

                header.addEventListener('click', function (e) {{
                    e.stopPropagation();
                    childrenContainer.classList.toggle('collapsed');
                    toggle.textContent = childrenContainer.classList.contains('collapsed') ? '+' : '-';
                }});
            }}

            parentElement.appendChild(li);
        }}

        window.addEventListener('DOMContentLoaded', function () {{
            const treeList = document.getElementById('tree-list');
            renderTreeList(treeData, treeList);
        }});
    </script>
</body>
</html>
"""
    
    # Preparar mode breakdown HTML
    mode_breakdown_html = []
    for mode, count in stats['mode_counts'].items():
        original_size = stats['mode_original_sizes'].get(mode, 0)
        compressed_size = stats['mode_compressed_sizes'].get(mode, 0)
        ratio = (original_size / compressed_size) if compressed_size > 0 else 1.0
        
        mode_name_display = {
            'lz_ext3_gpu': 'LZ4 GPU (Extended)',
            'lz4_gpu': 'LZ4 GPU',
            'lz4': 'LZ4',
            'raw': 'RAW (Sem compressão)'
        }.get(mode, mode)
        
        mode_breakdown_html.append(f"""
                <li class="mode-item">
                    <span class="mode-name">{mode_name_display}</span>
                    <div>
                        <span class="mode-count">{count:,} frames</span>
                        <span class="mode-ratio">{ratio:.2f}x</span>
                    </div>
                </li>
        """)
    
    # Render HTML
    html_content = html_template.format(
        archive_name=stats.get('archive_name', 'Arquivo Compactado'),
        total_original_formatted=stats['total_original_formatted'],
        total_compressed_formatted=stats['total_compressed_formatted'],
        compression_ratio=stats['compression_ratio'],
        space_saved_formatted=stats['space_saved_formatted'],
        space_saved_percent=stats['space_saved_percent'],
        total_files=stats['total_files'],
        total_duplicates=stats['total_duplicates'],
        total_frames=stats['total_frames'],
        volume_count=stats['volume_count'],
        mode_breakdown=''.join(mode_breakdown_html),
        tree_size_formatted=tree_data['size_formatted'],
        tree_total_items=tree_data['total_items'],
        tree_total_files=tree_data['total_files'],
        tree_total_folders=tree_data['total_folders'],
        tree_json=json.dumps(tree_data, ensure_ascii=False)
    )
    
    # Escrever HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✓ HTML gerado com sucesso: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gera mapa de árvore (tree map) HTML a partir de arquivo compactado"
    )
    parser.add_argument(
        'archive',
        type=str,
        help='Caminho para o arquivo .001 do arquivo compactado'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='tree_map.html',
        help='Caminho do arquivo HTML de saída (padrão: tree_map.html)'
    )
    
    args = parser.parse_args()
    
    archive_path = Path(args.archive).resolve()
    output_path = Path(args.output).resolve()
    
    if not archive_path.exists():
        print(f"Erro: Arquivo não encontrado: {archive_path}")
        return 1
    
    print(f"📂 Lendo arquivo: {archive_path.name}")
    print(f"📄 Saída HTML: {output_path.name}\n")
    
    # Ler footer e índice
    index = read_footer(archive_path)
    if index is None:
        return 1
    
    # Construir árvore
    print("\n🌳 Construindo árvore hierárquica...")
    tree_root = build_tree(index.get('files', []))
    tree_data = tree_root.to_dict()
    print(f"✓ Árvore construída: {tree_data['total_items']:,} itens, {tree_data['total_folders']:,} pastas")
    
    # Calcular estatísticas
    print("\n📊 Calculando estatísticas de compressão...")
    stats = calculate_compression_stats(index)
    stats['archive_name'] = archive_path.name
    print(f"✓ Compressão: {stats['compression_ratio']:.2f}x | Economia: {stats['space_saved_percent']:.1f}%")
    
    # Gerar HTML
    print("\n🎨 Gerando HTML...")
    generate_html(tree_data, stats, output_path)
    
    print(f"\n{'='*60}")
    print(f"✅ Concluído com sucesso!")
    print(f"{'='*60}")
    print(f"📁 Arquivo gerado: {output_path}")
    print(f"💾 Tamanho do HTML: {format_bytes(output_path.stat().st_size)}")
    print(f"\n💡 Abra o arquivo HTML em um navegador para visualizar o mapa.")
    
    return 0


if __name__ == '__main__':
    exit(main())
