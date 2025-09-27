# IADT-Fase-3

## Objetivo

Este projeto tem como objetivo executar o fine-tuning de um foundation model utilizando o dataset "The AmazonTitles-1.3MM". O foco é criar um modelo especializado capaz de descrever produtos baseado em seus títulos.

## Solução

A implementação está baseada no notebook **Gemma 3 (27B)** disponível em [Unsloth Notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks), utilizando a biblioteca Unsloth para otimização de memória e velocidade durante o processo de fine-tuning.

## Estrutura do Projeto

### Dataset Processing Pipeline

O notebook `dataset_processing_pipeline.ipynb` é responsável por:

1. **Processamento do dataset original**: Converte o arquivo `trn.json` (AmazonTitles-1.3MM) em um formato compatível com fine-tuning
2. **Formatação de conversações**: Transforma os dados em formato de conversação similar ao [mlabonne/FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)
3. **Sampling**: Extrai uma amostra de 5.000 conversações do dataset completo
4. **Upload para Hugging Face**: Publica o dataset processado em [umtaldejr/IADT-Fase-3-dataset-sample](https://huggingface.co/datasets/umtaldejr/IADT-Fase-3-dataset-sample)

## Histórico de Alterações

### v1.0.0 - Setup Inicial
- ✅ Criação do pipeline de pré-processamento de dados
- ✅ Implementação da conversão de formato para fine-tuning
- ✅ Sampling de 5.000 conversações do dataset original
- ✅ Upload automático para Hugging Face Hub
- ✅ Formatação compatível com FineTome-100k
- ✅ **Formato inicial do dataset**: Conversações no padrão `"Describe: [título]"` → `"[descrição]"`
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

### Próximos Passos
- [ ] Implementação do notebook de fine-tuning baseado no Gemma 3 (27B)
- [ ] Configuração dos hiperparâmetros de treinamento
- [ ] Execução do treinamento completo
- [ ] Avaliação do modelo fine-tuned
- [ ] Documentação dos resultados
