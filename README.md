# 🚀 CL_Compressor: Motor de Compressão e Deduplicação Acelerado por GPU

O **CL_Compressor** é uma solução de arquivamento de alto desempenho, projetada para processar grandes volumes de dados com eficiência e velocidade superiores. Ele combina uma arquitetura de pipeline assíncrona, um motor de deduplicação inteligente de múltiplos estágios e um kernel de compressão LZ4 totalmente personalizado e acelerado por GPU (OpenCL).

Desenvolvido para cenários de **backup, ingestão de dados em larga escala e arquivamento versionado**, o CL_Compressor transforma o gargalo de I/O e processamento em um fluxo de trabalho otimizado, aproveitando o poder de processamento paralelo das Unidades de Processamento Gráfico (GPUs).

## ✨ Inovações e Tecnologias Chave

O projeto se destaca por uma série de otimizações que garantem o máximo *throughput* e a melhor taxa de compressão possível:

### 1. Kernel LZ4 Personalizado com Janela Estendida (OpenCL)

O coração do sistema é um kernel LZ4 customizado, implementado em **OpenCL**, que supera as limitações da implementação padrão:
*   **Janela de Match Estendida:** Utiliza *offsets* de 3 bytes, expandindo a janela de busca de *matches* para **16 MB** (contra 64 KB do LZ4 padrão). Isso melhora drasticamente a taxa de compressão para arquivos grandes e repetitivos.
*   **Busca Top-K Adaptativa:** O algoritmo de busca de *matches* na tabela de *hash* utiliza uma técnica **Top-K** (configurável via `HASH_CANDIDATES`), que prioriza encontrar o melhor *match* possível, aumentando o *ratio* de compressão sem comprometer a velocidade devido ao paralelismo da GPU.
*   **Otimização de Saída Antecipada (*Early-Exit*):** A lógica de compressão para de buscar *matches* melhores assim que encontra um com o comprimento definido por `GOOD_ENOUGH_MATCH`, equilibrando *ratio* e velocidade.

### 2. Deduplicação Inteligente de Múltiplos Estágios

Para minimizar o I/O e o custo computacional do *hashing* completo, o processo de deduplicação emprega um filtro de quatro estágios antes de calcular o *hash* completo na GPU:
1.  **Filtro por Tamanho:** Agrupa arquivos pelo tamanho.
2.  **Filtro 2 Bytes Iniciais:** Verifica os dois primeiros bytes.
3.  **Filtro 2 Bytes Finais:** Verifica os dois últimos bytes.
4.  **Filtro 3 Bytes Centrais:** Verifica três bytes ao redor do centro do arquivo.

Somente os arquivos que passam por esses filtros rápidos (e baratos) são submetidos ao **cálculo de *hash* FNV-1a 64-bit paralelo na GPU**, garantindo que o motor de deduplicação seja excepcionalmente rápido e eficiente.

### 3. Otimização de Buffer e Read-Ahead (I/O Assíncrono)

O sistema utiliza um motor de I/O assíncrono com *buffers* configuráveis (`READ_BUFFER_BATCHES` e `WRITE_BUFFER_BATCHES`) para desacoplar o processamento da GPU da latência do disco. Isso implementa um mecanismo de **Read-Ahead** (leitura antecipada) e **Write-Behind** (escrita atrasada), mantendo a GPU sempre alimentada com dados e o *throughput* de escrita constante.

### 4. Auto-Skip Adaptativo (Otimização de Incompressibilidade)

O sistema incorpora uma otimização para dados incompressíveis. Se o tamanho do *frame* comprimido na GPU não for menor que o tamanho original, o sistema automaticamente armazena o *frame* em seu **formato RAW (não comprimido)**. Isso evita o desperdício de tempo de processamento e espaço de armazenamento em dados que não podem ser efetivamente comprimidos, atuando como um mecanismo de **"auto-skip"** para blocos incompressíveis.

## ⚙️ Fluxo de Processamento Completo (Pipeline)

O processo de compressão segue um pipeline de 8 estágios, projetado para máxima paralelização e eficiência:

| Fase | Título | Descrição e Otimizações |
| :--- | :--- | :--- |
| **1** | **Scan Assíncrono & Metadata** | Traversal de diretório multi-threaded. Coleta metadados (timestamps, permissões, tamanho) e utiliza **Read-Ahead** para arquivos grandes. Emite *jobs* para a fila. |
| **2** | **Motor de Chunking** | Segmentação de dados usando janela rolante (*rolling-window*). Suporte a janela estendida e heurísticas adaptativas para produzir blocos otimizados para deduplicação. |
| **3** | **Deduplicação Multi-Nível** | Aplica o filtro de 4 estágios (Tamanho, 2 Bytes Iniciais, 2 Bytes Finais, 3 Bytes Centrais) seguido por **Hashing FNV-1a 64-bit paralelo na GPU**. Inclui **Auto-Skip** para dados repetitivos e rastreamento de referências. |
| **4** | **Inicialização do Pipeline** | Ativação de *pools* de *workers* CPU/GPU baseada na configuração. Balanceamento de carga centralizado para a fila de blocos. |
| **5** | **Estágio de Compressão** | **Caminho GPU:** Utiliza o kernel LZ4 personalizado (OpenCL) com busca Top-K e lógica de *early-exit*. **Caminho CPU:** Fallback otimizado para LZ4 em caso de indisponibilidade ou erro da GPU. |
| **6** | **Montagem de Blocos** | Reordena os blocos processados em um fluxo de saída linear. Integra referências de deduplicação e anexa os resultados da compressão. |
| **7** | **Mapeamento de Metadados** | Serializa o índice global (tabela de blocos, offsets, tamanhos originais, referências de *hash*). O índice é **comprimido com zlib**. |
| **8** | **Escrita de Volumes & Footer** | Escreve o *payload* (blocos comprimidos) em volumes multi-parte (`.001`, `.002`, etc.). O índice comprimido é **embutido diretamente no último volume** do arquivo, com um *footer* fixo (`GPU_IDX1`) para localização rápida. |

## 🛠️ Guia de Configuração (`config.txt`)

O arquivo `config.txt` centraliza todos os parâmetros de *tuning* para o sistema. Abaixo estão os itens essenciais e seus propósitos:

### Compressão LZ4 GPU

| Parâmetro | Descrição | Impacto |
| :--- | :--- | :--- |
| `FORCE_CPU_MODE` | Força o uso exclusivo da CPU, ignorando a GPU. | Debugging ou sistemas sem GPU. |
| `COMPRESSOR_BATCH_SIZE` | Número de *frames* processados por vez na GPU. | **Performance:** Afeta o uso de VRAM e o *throughput* da GPU. |
| `GPU_FALLBACK_ENABLED` | Habilita o *fallback* automático para CPU em caso de erro na GPU. | **Estabilidade:** Garante a conclusão do processo. |
| `DECOMPRESSOR_BATCH_SIZE` | Número de *frames* por *batch* na descompressão. | **Performance:** Afeta o uso de VRAM na descompressão. |
| `MAX_WORKER_THREADS` | Número de *threads* paralelas para descompressão. | **Paralelismo:** Equilíbrio entre paralelismo CPU-GPU e contenção de OpenCL. |

### Otimização de I/O e Workers

| Parâmetro | Descrição | Impacto |
| :--- | :--- | :--- |
| `READ_BUFFER_BATCHES` | Número de *batches* em *buffer* para **leitura antecipada (Read-Ahead)**. | **RAM/I/O:** Maior valor reduz espera por I/O de leitura, mas aumenta o uso de RAM. |
| `WRITE_BUFFER_BATCHES` | Número de *batches* em *buffer* para **escrita atrasada (Write-Behind)**. | **RAM/I/O:** Maior valor reduz espera por I/O de escrita, mas aumenta o uso de RAM. |
| `NUM_SCAN_WORKERS` | Número de *workers* paralelos para *scanning* de diretórios. | **Velocidade de Scan:** Acelera a fase inicial em HDDs grandes. |
| `NUM_IO_WORKERS` | Número de *workers* I/O para leitura paralela de bytes (Fases de Deduplicação). | **Velocidade de Deduplicação:** Leitura paralela mais rápida durante filtros byte-a-byte. |
| `NUM_READERS` | Número de *threads* de leitura para alimentar o *hash* GPU (Fase 5). | **Throughput GPU:** Mantém a GPU sempre ocupada, melhorando o *throughput*. |
| `BUFFER_SIZE` | Tamanho do *buffer* da fila entre *readers* e GPU. | **Latência:** Maior *buffer* evita que a GPU fique ociosa esperando dados. |

### Parâmetros Avançados do Kernel LZ4 (Tuning Fino)

Estes parâmetros controlam o comportamento do kernel OpenCL e afetam diretamente o *ratio* de compressão e a velocidade. **A alteração requer a recompilação do kernel OpenCL.**

| Parâmetro | Descrição | Impacto no Ratio/Velocidade |
| :--- | :--- | :--- |
| `HASH_LOG` | Log2 do número de entradas base na tabela de *hash*. | **Ratio:** Maior valor = Mais memória GPU, melhor *ratio* para dados grandes. |
| `HASH_CANDIDATES` | Número de posições candidatas por entrada de *hash* (**Top-K**). | **Ratio:** Maior valor = Melhor *ratio* (encontra *matches* mais longos), mas mais lento. |
| `GOOD_ENOUGH_MATCH` | Comprimento de *match* considerado "bom o suficiente" para parar a busca. | **Velocidade:** Menor valor = Mais rápido (aceita *matches* curtos). Maior valor = Melhor *ratio*, mais lento. |

## 📊 Performance e Resultados

A aceleração por GPU proporciona ganhos de desempenho significativos:

*   **Velocidade de Compressão:** O LZ4 GPU pode atingir **2–3+ GB/s**, dependendo da placa.
*   **Redução Total:** A combinação de Deduplicação + LZ4 resulta tipicamente em uma **redução total de 60–85%** no tamanho do arquivo.
*   **Deduplicação:** O *hashing* paralelo na GPU reduz drasticamente a sobrecarga da CPU.

> **Exemplo de Log de Produção:**
>
> ```
> [Dedup Final] Encontradas 1532 duplicatas reais.
> [Dedup Final] Economia potencial: 50.35 MB
>
> [Compressor] | LZ_EXT3_GPU=32 (43.24%) | RAW=42 (56.76%) | Redução = 43.2%
> Dados escritos:   1182.6 MB
> Velocidade média: 110.8 MB/s
> ```

## 🛠️ Requisitos e Uso

### Requisitos

*   **Python 3.9+**
*   **PyOpenCL**
*   **Numpy**
*   **LZ4**
*   **Zlib**
*   **Qualquer GPU compatível com CUDA/OpenCL** (*Recomendado:* NVIDIA GTX 1050 Ti ou superior).

### Instalação

```bash
# Clone o repositório
git clone https://github.com/danieldurio/CL_Compressor
cd CL_Compressor

# Instale as dependências
pip install pyopencl numpy lz4
```

### Compressão

Use o script principal `compressor_lz4_dedup.py`:

```bash
python compressor_lz4_dedup.py <pasta_origem> -o <nome_arquivo_saida>
# Exemplo: python compressor_lz4_dedup.py /home/user/meus_arquivos -o backup_2025
# Isso criará volumes como backup_2025.001, backup_2025.002, etc.
```

### Descompressão

Use o script `decompressor_lz4.py` apontando para o primeiro volume (`.001`):

```bash
python decompressor_lz4.py <arquivo_saida.001> -o <pasta_destino>
# Exemplo: python decompressor_lz4.py backup_2025.001 -o /home/user/restauracao
```

## 🗺️ Roadmap

O projeto está em constante evolução. Planos futuros incluem:

*   Tamanho de janela adaptativo.
*   Ferramenta de reparo para volumes ausentes.
*   Adicionar VSS ( Windows )

---

## 🤝 Contribuições

Contribuições, *pull requests*, relatórios de problemas e sugestões são muito bem-vindos! Este é um projeto experimental, e sua ajuda é essencial para a melhoria contínua.
