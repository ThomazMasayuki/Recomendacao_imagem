**Sistema de Recomendação por Similaridade de Imagens**

Este projeto implementa um sistema de recomendação baseado em conteúdo visual, capaz de sugerir imagens semelhantes a partir de uma imagem de consulta. Ele utiliza Deep Learning para extrair características visuais e busca vetorial para encontrar imagens similares.

**Objetivo**

Dado uma imagem de entrada, o sistema retorna as K imagens mais visualmente semelhantes do catálogo.

Esse tipo de sistema é usado em:

- Recomendação de produtos (e-commerce)

- Busca por imagens semelhantes

- Curadoria de conteúdo visual

- Organização de acervos

- Recuperação de imagens por conteúdo (CBIR)

O sistema é dividido em três etapas principais:

- Extração de características (embeddings)

Utiliza o modelo CLIP (OpenAI) para converter cada imagem em um vetor numérico que representa seu conteúdo visual.

- Indexação vetorial

Os vetores são armazenados em um índice utilizando FAISS, que permite buscas rápidas por similaridade.

- Busca por similaridade

Quando uma imagem de consulta é fornecida:

Seu embedding é extraído

O sistema busca os vetores mais próximos no índice

Retorna as imagens mais semelhantes

**Tecnologias utilizadas**

Python

PyTorch

OpenCLIP

FAISS

Pillow

NumPy

tqdm
