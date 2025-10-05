# 🚀 IADT-Fase-3

<div align="center">

[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/umtaldejr/iadt-fase-3-finetune)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-blue)](https://huggingface.co/datasets/umtaldejr/IADT-Fase-3-dataset-sample)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

_Fine-tuning do modelo Gemma 3 (27B) para geração de descrições detalhadas de produtos Amazon_

</div>

## 📋 Índice

- [Objetivo](#-objetivo)
- [Arquitetura da Solução](#️-arquitetura-da-solução)
- [Evolução das Iterações](#-evolução-das-iterações)
- [Resultados Finais](#-resultados-finais)
- [Como Usar](#-como-usar)
- [Melhorias Futuras](#-melhorias-futuras)
- [Links e Recursos](#-links-e-recursos)

## 🎯 Objetivo

Este projeto implementa o **fine-tuning do modelo Gemma 3 (27B)** utilizando o dataset "The AmazonTitles-1.3MM" para criar um modelo especializado em **gerar descrições detalhadas de produtos** a partir de seus títulos.

**Meta**: Desenvolver um modelo capaz de transformar títulos simples de produtos em descrições ricas e informativas, mantendo precisão e naturalidade.

## 🏗️ Arquitetura da Solução

### 🤖 Modelo Base

| Componente                | Especificação                                            |
| ------------------------- | -------------------------------------------------------- |
| **Modelo**                | `unsloth/gemma-3-27b-it` (Gemma 3 27B Instruction-Tuned) |
| **Biblioteca**            | Unsloth (otimização de memória e velocidade)             |
| **Quantização**           | 4-bit (QLoRA)                                            |
| **Parâmetros Totais**     | 27,489,164,912                                           |
| **Parâmetros Treináveis** | 56,758,272 (0.21%)                                       |
| **GPU Utilizada**         | A100 (Google Colab)                                      |

### 📊 Dataset

| Aspecto              | Detalhes                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Fonte**            | AmazonTitles-1.3MM (`trn.json`)                                                                              |
| **Tamanho Original** | 2,248,619 exemplos processados                                                                               |
| **Tamanho Final**    | 10,000 exemplos processados                                                                                  |
| **Formato**          | Conversações multi-turn com 7 padrões diferentes                                                             |
| **Repositório**      | [umtaldejr/IADT-Fase-3-dataset-sample](https://huggingface.co/datasets/umtaldejr/IADT-Fase-3-dataset-sample) |

## 🚧 Evolução das Iterações

### 🔍 Detalhes das Iterações

#### v1 - Setup Inicial e Baseline

**Objetivo**: Criar pipeline completo de pré-processamento de dados e implementar fine-tuning do modelo Gemma 3 (27B), estabelecendo a baseline do projeto.

**Hipótese**: Um pipeline simples de processamento combinado com fine-tuning usando LoRA será suficiente para validar a viabilidade do projeto e gerar um modelo capaz de criar descrições básicas de produtos.

**Configuração do Dataset**:

- **Tamanho**: 5,000 exemplos
- **Formato**: Conversações simples (single-turn)
- **Padrão único**: `"Describe: [título]"` → `"[descrição]"`
- **Processamento**:
  ```python
  # Pipeline de processamento
  1. Leitura do arquivo trn.json (formato JSONL)
  2. Decodificação HTML (html.unescape)
  3. Limpeza de espaços em branco (.strip())
  4. Validação (título e conteúdo não vazios)
  5. Sampling aleatório de 5,000 registros
  6. Conversão para formato de conversação
  7. Upload para HuggingFace Hub (privado)
  ```

**Estrutura do Dataset**:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Describe: [título do produto]"
    },
    {
      "from": "gpt",
      "value": "[descrição detalhada do produto]"
    }
  ]
}
```

**Configuração do Modelo**:

- **Modelo base**: `unsloth/gemma-3-27b-it`
- **Biblioteca**: Unsloth (otimização de memória e velocidade)
- **Quantização**: 4-bit (QLoRA)
- **Max sequence length**: 2048 tokens
- **Parâmetros totais**: 27,489,164,912
- **Parâmetros treináveis**: 56,758,272 (0.21%)
- **GPU**: A100 (Google Colab)

**Parâmetros LoRA**:

```python
r = 8                    # Rank das matrizes LoRA
lora_alpha = 8           # Scaling factor (alpha == r)
lora_dropout = 0         # Sem dropout
bias = "none"            # Sem bias adicional
random_state = 3407      # Seed para reprodutibilidade
```

**Módulos Fine-tuned**:

- ✅ **Language Layers**: Ativado
- ✅ **Attention Modules**: Ativado
- ✅ **MLP Modules**: Ativado
- ❌ **Vision Layers**: Desativado (apenas texto)

**Hiperparâmetros de Treinamento**:

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
effective_batch_size = 8        # (2 × 4 × 1 GPU)
max_steps = 30                  # Teste inicial limitado
learning_rate = 2e-4
warmup_steps = 5
weight_decay = 0.01
lr_scheduler_type = "linear"
optimizer = "adamw_8bit"
seed = 3407
```

**O que foi implementado**:

- ✅ Notebook `dataset_processing_pipeline.ipynb` para processamento de dados
- ✅ Notebook `IADT_Fase_3.ipynb` para fine-tuning baseado no Unsloth
- ✅ Pipeline automatizado de processamento do dataset
- ✅ Carregamento do modelo Gemma 3 (27B) com 4-bit quantization
- ✅ Configuração de LoRA para fine-tuning eficiente
- ✅ Integração com HuggingFace Hub
- ✅ Sistema de logging de métricas
- ✅ Arquivo `.gitignore` para dados sensíveis

**Análise das Métricas**:

- **Steps executados**: 30 (~2.4% de uma época)
- **Loss inicial**: 4.100
- **Loss mínimo**: 1.974 (step 30)
- **Loss final**: 2.044
- **Redução**: 51.9%
- **Convergência**: Rápida e consistente
- **Tempo de treinamento**: ~5 minutos

![](metrics/v1_loss_plot.png)

**Limitações**:

- ⚠️ **Single-turn apenas**: Sem conversações multi-turno ou contexto
- ⚠️ **Baixa diversidade**: Formato único "Describe:" sem variações
- ⚠️ **Tamanho limitado**: 5,000 de 1.3M disponíveis (0.38%)
- ⚠️ **Treinamento incompleto**: 30 de ~1,250 steps possíveis (2.4%)
- ⚠️ **Sem validação**: Sem dataset de validação para monitorar generalização
- ⚠️ **Sem métricas qualitativas**: BLEU, ROUGE não implementadas

**Resultado**: Pipeline completo estabelecido com sucesso. Dataset inicial gerado e publicado. Modelo baseline funcional capaz de gerar descrições básicas. Próximas versões devem focar em: (1) aumentar steps de treinamento e (2) melhorar diversidade do dataset.

---

#### v2 - Aumento de Steps

**Objetivo**: Executar treinamento completo de 1 época para avaliar convergência adequada do modelo

**Hipótese**: Aumentar o número de steps permitirá melhor convergência e redução do loss, resultando em descrições de produtos mais precisas e coerentes.

**Configuração**:

- **Modelo**: Mesma configuração da v1
- **LoRA**: `r=8, alpha=8, dropout=0`
- **Hiperparâmetros**:
  ```python
  per_device_train_batch_size = 2
  gradient_accumulation_steps = 4
  effective_batch_size = 8
  num_train_epochs = 1        # ← Mudança: época completa
  learning_rate = 2e-4
  warmup_steps = 5
  weight_decay = 0.01
  optimizer = "adamw_8bit"
  ```

**O que mudou**:

- ✅ **Treinamento**: Alterado de `max_steps=30` para `num_train_epochs=1`
- ✅ **Steps totais**: 625 (época completa com 5,000 exemplos)
- ✅ **Dataset mantido**: v1 (🔄 **single-turn**, padrão "Describe:")
- ❌ **Formato**: Ainda sem diversidade de conversação

**Análise das Métricas**:

- **Steps executados**: 625 (1 época completa)
- **Loss inicial**: 4.098
- **Loss mínimo**: 0.661 (step 528)
- **Loss final**: 2.108
- **Redução máxima**: 83.9%
- **Melhoria vs v1**: +66.5% (loss mínimo 1.974 → 0.661) ✅ **Melhoria significativa**
- **Convergência**: Instável, loss mínimo excelente mas final alto (overfitting)

![](metrics/v2_loss_plot.png)

**Limitações**:

- ⚠️ **Single-turn mantido**: Ainda sem conversações multi-turno
- ⚠️ **Baixa diversidade**: Padrão único "Describe:" sem variações
- ⚠️ **Treinamento limitado**: Apenas 1 época pode não ser suficiente
- ⚠️ **Sem validação**: Sem métricas de validação para avaliar overfitting
- ⚠️ **Formato repetitivo**: Pode limitar capacidade de generalização

---

#### v3 - Melhoria no Dataset

**Objetivo**: Aumentar a diversidade do dataset com múltiplos padrões de conversação para melhorar a capacidade de generalização do modelo

**Hipótese**: Um dataset com maior variedade de prompts (diferentes estilos de perguntas) permitirá que o modelo aprenda a responder de forma mais natural e adaptável a diferentes contextos.

**Configuração**:

- **Modelo**: Mesma configuração das versões anteriores
- **LoRA**: `r=8, alpha=8, dropout=0`
- **Hiperparâmetros**: Idênticos à v2
  ```python
  per_device_train_batch_size = 2
  gradient_accumulation_steps = 4
  num_train_epochs = 1
  learning_rate = 2e-4
  optimizer = "adamw_8bit"
  ```

**O que mudou**:

- ✅ **Dataset expandido**: 5,000 → 10,000 exemplos (100% de aumento)
- 🔄 **Formato revolucionário**: Single-turn → **Multi-turn** (conversações complexas)
- ✅ **7 padrões de conversação implementados**:
  - `basic_description` (20%): "Describe: [título]" - **Single-turn**
  - `information_request` (15%): "I need information about..." - **Single-turn**
  - `detailed_inquiry` (15%): "Can you provide details..." - **Single-turn**
  - `feature_analysis` (15%): "What are the key features..." - **Single-turn**
  - `casual_question` (15%): "What can you tell me..." - **Single-turn**
  - `summary_request` (10%): "Give me a summary..." - **Single-turn**
  - `multi_turn` (10%): **Conversações multi-turno** com follow-ups
- ✅ **Processamento aprimorado**: Decodificação HTML melhorada
- ✅ **Diversidade**: 90% single-turn + 10% multi-turn

**Análise das Métricas**:

- **Steps executados**: 625 (1 época com 10,000 exemplos)
- **Loss inicial**: 3.994 (menor que v2!)
- **Loss mínimo**: 0.820 (step 102)
- **Loss final**: 1.843
- **Redução máxima**: 79.5%
- **Melhoria vs v2**: -24.0% (loss mínimo 0.661 → 0.820) ⚠️ **Leve piora no mínimo**
- **Observação**: Dataset diversificado melhorou estabilidade, mas loss mínimo foi menor

![](metrics/v3_loss_plot.png)

**Limitações**:

- ⚠️ **Treinamento limitado**: Ainda apenas 1 época de treinamento
- ⚠️ **Multi-turn minoritário**: Apenas 10% das conversações são multi-turn
- ⚠️ **Loss similar**: Loss final similar à v2 (pode precisar de mais épocas)
- ⚠️ **Sem avaliação qualitativa**: Respostas geradas não avaliadas
- ⚠️ **Proporção não otimizada**: Distribuição dos padrões pode não ser ideal

**Resultado**: Dataset significativamente melhorado com 7 padrões diferentes. A diversidade está pronta para ser explorada com treinamento mais longo.

---

#### v4 - Treinamento Completo (FINAL)

**Objetivo**: Executar treinamento completo com 2 épocas para maximizar o aprendizado do modelo com o dataset diversificado

**Hipótese**: Com o dataset melhorado (v3) e treinamento de 2 épocas completas, o modelo alcançará convergência ótima e melhor capacidade de gerar descrições de produtos de alta qualidade.

**Configuração**:

- **Modelo**: Mesma configuração das versões anteriores
- **LoRA**: `r=8, alpha=8, dropout=0`
- **Hiperparâmetros**:
  ```python
  per_device_train_batch_size = 2
  gradient_accumulation_steps = 4
  effective_batch_size = 8
  num_train_epochs = 2        # ← Mudança: 2 épocas completas
  learning_rate = 2e-4
  warmup_steps = 5
  weight_decay = 0.01
  optimizer = "adamw_8bit"
  ```

**O que mudou**:

- ✅ **Treinamento estendido**: `num_train_epochs=1` → `num_train_epochs=2`
- ✅ **Steps dobrados**: 1,250 steps totais (2 épocas × 625 steps)
- ✅ **Dataset mantido**: v3 (10,000 exemplos com 7 padrões)
- 🔄 **Multi-turn preservado**: 90% single-turn + 10% multi-turn
- ✅ **Convergência aprimorada**: Treinamento mais longo para melhor aprendizado

**Análise das Métricas**:

- **Steps executados**: 1,250 (2 épocas completas)
- **Loss inicial**: 3.050
- **Loss mínimo**: 0.539 (step 621) 🏆 **Melhor de todas as versões**
- **Loss final**: 1.679
- **Redução máxima**: 82.3%
- **Melhoria vs v3**: +34.3% (loss mínimo 0.820 → 0.539) ✅ **Melhor resultado global**
- **Observações importantes**:
  - 🏆 **Melhor loss mínimo** alcançado em todo o projeto
  - ✅ Treinamento mais longo permitiu convergência superior
  - ⚠️ Overfitting evidente na 2ª época (loss final > mínimo)
  - ✅ Modelo final com melhor capacidade de aprendizado

**Análise de Convergência**:

- **1ª Época (steps 1-625)**: Loss 3.050 → 0.539 (redução de 82.3%)
- **2ª Época (steps 626-1,250)**: Loss continuou baixo, estabilizando em 1.679
- **Padrão**: Convergência rápida na 1ª época, estabilização na 2ª

![](metrics/v4_loss_plot.png)

**Limitações**:

- ⚠️ **Possível overfitting**: Loss final > loss mínimo (1.679 vs 0.539)
- ⚠️ **Multi-turn limitado**: Apenas 10% das conversações são multi-turn
- ⚠️ **Sem validação**: Dataset de validação não implementado
- ⚠️ **Métricas limitadas**: BLEU, ROUGE, avaliação humana ausentes
- ⚠️ **Learning rate**: Não testado com taxa menor para treinamento suave

**Resultado**: Modelo final com melhor convergência e estabilidade. O dataset diversificado (v3) combinado com 2 épocas de treinamento produziu o modelo mais robusto do projeto.

**Modelo publicado**: [umtaldejr/iadt-fase-3-finetune](https://huggingface.co/umtaldejr/iadt-fase-3-finetune)

---

## 🏆 Resultados Finais

### 📈 Métricas de Performance

| Métrica                  | Valor | Observação                      |
| ------------------------ | ----- | ------------------------------- |
| **Loss Inicial**         | 3.050 | Ponto de partida                |
| **Loss Mínimo Global**   | 0.539 | Melhor resultado alcançado      |
| **Loss Final**           | 1.679 | Após 2 épocas completas         |
| **Loss Médio**           | 1.851 | Média durante treinamento       |
| **Redução Máxima**       | 82.3% | Do inicial ao mínimo            |
| **Steps Totais**         | 1,250 | 2 épocas × 625 steps            |
| **Tempo de Treinamento** | ~3.5h | A100 GPU                        |

### 🔗 Modelo Publicado

**HuggingFace**: [umtaldejr/iadt-fase-3-finetune](https://huggingface.co/umtaldejr/iadt-fase-3-finetune)

---

## 🚀 Como Usar

### 📦 Instalação

```bash
# Instalar dependências principais
pip install unsloth transformers datasets trl torch

# Para usar com GPU (recomendado)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 🤖 Carregar Modelo Fine-tuned

```python
from unsloth import FastLanguageModel
import torch

# Carregar modelo e tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "umtaldejr/iadt-fase-3-finetune",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# Configurar chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma",
    map_eos_token = True,
)
```

### 💬 Gerar Descrição de Produto

```python
# Exemplo de uso
messages = [
    {"role": "system", "content": "You are an AI assistant specialized in product descriptions."},
    {"role": "user", "content": "What can you tell me about Gifts & Decor Solar Powered Outdoor Garden Lighthouse?"},
]

# Aplicar template e gerar resposta
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to("cuda")

output = model.generate(
    **inputs,
    max_new_tokens = 512,
    temperature = 1.0,
    top_p = 0.95,
    top_k = 64,
    do_sample = True,
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

### 🎯 Padrões de Prompt Suportados

O modelo foi treinado com 7 padrões diferentes de conversação:

1. **Descrição Básica** (20%): `"Describe: [título]"`
2. **Solicitação de Informação** (15%): `"I need information about..."`
3. **Consulta Detalhada** (15%): `"Can you provide details..."`
4. **Análise de Características** (15%): `"What are the key features..."`
5. **Pergunta Casual** (15%): `"What can you tell me..."`
6. **Pedido de Resumo** (10%): `"Give me a summary..."`
7. **Multi-turn** (10%): Conversações com múltiplas interações

---

## 🔮 Melhorias Futuras

### 🎯 Próximos Passos Prioritários

| Prioridade   | Melhoria                              | Impacto Esperado              |
| ------------ | ------------------------------------- | ----------------------------- |
| 🔴 **Alta**  | Dataset de validação + early stopping | Prevenir overfitting          |
| 🔴 **Alta**  | Métricas qualitativas (BLEU, ROUGE)   | Avaliação objetiva            |
| 🟡 **Média** | Learning rate menor (2e-5)            | Convergência mais suave       |
| 🟡 **Média** | Rank LoRA maior (r=16/32)             | Maior capacidade de adaptação |
| 🟢 **Baixa** | Gradient checkpointing                | Maior batch size              |
| 🟢 **Baixa** | Data augmentation adicional           | Maior robustez                |

### 🧪 Experimentos Futuros

- **Arquitetura**: Testar outros modelos (Llama 3, Mistral)
- **Dataset**: Expandir para outros domínios de produtos
- **Avaliação**: Implementar avaliação humana
- **Deployment**: Criar API REST para inferência
- **Otimização**: Quantização INT8 para produção

---

## 🔗 Links e Recursos

### 🤗 HuggingFace

| Recurso             | Link                                                                                                         | Descrição                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------- |
| **🤖 Modelo Final** | [umtaldejr/iadt-fase-3-finetune](https://huggingface.co/umtaldejr/iadt-fase-3-finetune)                      | Gemma 3 (27B) fine-tuned     |
| **📊 Dataset**      | [umtaldejr/IADT-Fase-3-dataset-sample](https://huggingface.co/datasets/umtaldejr/IADT-Fase-3-dataset-sample) | 10K conversações processadas |

### 📚 Documentação e Referências

| Ferramenta      | Link                                                                  | Uso no Projeto            |
| --------------- | --------------------------------------------------------------------- | ------------------------- |
| **Unsloth**     | [docs.unsloth.ai](https://docs.unsloth.ai/)                           | Otimização de fine-tuning |
| **Gemma 3**     | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | Modelo base utilizado     |
| **LoRA Paper**  | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)          | Técnica de fine-tuning    |
| **QLoRA Paper** | [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)          | Quantização 4-bit         |

### 🛠️ Ferramentas Utilizadas

- **Google Colab** (A100 GPU)
- **Python 3.10+**
- **PyTorch 2.0+**
- **Transformers 4.40+**
- **Unsloth**
- **Datasets**
- **TRL (Transformer Reinforcement Learning)**

---
