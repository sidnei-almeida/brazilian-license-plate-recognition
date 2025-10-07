# 🚗 Brazilian License Plate Recognition System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![GitHub](https://img.shields.io/badge/GitHub-sidnei--almeida-181717?logo=github)](https://github.com/sidnei-almeida)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-saaelmeida93-0A66C2?logo=linkedin)](https://www.linkedin.com/in/saaelmeida93/)

</div>

## 📋 Descrição do Projeto

Sistema avançado de **Reconhecimento Automático de Placas de Veículos (ALPR)** desenvolvido especificamente para placas brasileiras, incluindo o padrão Mercosul. Utiliza um modelo YOLOv8 personalizado treinado com alta precisão para detectar placas em imagens de veículos.

## ✨ Características Principais

- 🔍 **Detecção de Placas**: Modelo YOLOv8 otimizado para placas brasileiras
- 🚗 **Padrão Mercosul**: Suporte completo ao novo formato de placas brasileiro
- 📊 **Interface Interativa**: Aplicação Streamlit com visualizações avançadas
- 📈 **Análise de Performance**: Métricas detalhadas e gráficos interativos
- 🧪 **Teste em Tempo Real**: Interface para testar o modelo com suas próprias imagens
- 📚 **Documentação Completa**: Guias detalhados de uso e desenvolvimento

## 🏗️ Arquitetura do Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Interface     │    │   Modelo YOLOv8  │    │   Processamento │
│   Streamlit     │───▶│   Treinado       │───▶│   de Imagens    │
│                 │    │                  │    │                 │
│ - Navegação     │    │ - Detecção       │    │ - Bounding      │
│ - Visualizações │    │ - Classificação  │    │   Boxes         │
│ - Testes        │    │ - Confiança      │    │ - Confiança     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Performance do Modelo

### Métricas de Treinamento
- **Precisão (Precision)**: 99.69%
- **Recall**: 99.19%
- **mAP@50**: 99.49%
- **mAP@50-95**: 95.56%
- **Melhor Época**: 170/300

### Recursos Utilizados
- **Modelo Base**: YOLOv8s (Small)
- **Dataset**: Especializado em placas brasileiras
- **Épocas de Treinamento**: 300 (com early stopping)
- **Tamanho do Batch**: 16
- **Tamanho da Imagem**: 640x640

## 🚀 Instalação e Execução

### Pré-requisitos

- **Python 3.11+** (incluindo 3.13)
- **pip** (gerenciador de pacotes)
- **🌐 Streamlit Cloud** (recomendado) ou ambiente local

### ⚡ Performance Otimizada

Este sistema foi especialmente otimizado para **Streamlit Cloud**:

- ✅ **Versões CPU** das bibliotecas (menor tamanho)
- ✅ **Sem necessidade de GPU** para funcionamento
- ✅ **Deploy direto** no Streamlit Cloud
- ✅ **Performance adequada** mesmo com recursos limitados

### Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/sidnei-almeida/brazilian-license-plate-recognition.git
   cd brazilian-license-plate-recognition
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

5. **Acesse no navegador:**
   Abra `http://localhost:8501` para visualizar a aplicação.

### 🚀 Deploy no Streamlit Cloud (Recomendado)

Para fazer deploy gratuito no Streamlit Cloud:

1. **Faça fork** deste repositório no GitHub
2. **Acesse** [share.streamlit.io](https://share.streamlit.io)
3. **Conecte** seu repositório GitHub
4. **Configure**:
   - **Main file path**: `app.py`
   - **Python version**: 3.13
   - **Requirements**: já incluído no `requirements.txt`
   - **System packages**: já incluído no `packages.txt`
5. **Deploy!** - O sistema funcionará perfeitamente na nuvem

> 💡 **Notas importantes**:
> - ✅ As imagens de teste são carregadas automaticamente do GitHub
> - ✅ O `packages.txt` instala dependências do sistema necessárias para o OpenCV
> - ✅ O `opencv-python-headless` é usado para evitar conflitos no Streamlit Cloud

## ⚡ Performance

### Otimizado para Streamlit Cloud

| Ambiente | Tempo de Inferência | Recursos | Status |
|----------|-------------------|----------|---------|
| **Streamlit Cloud** | ~3-8 segundos | CPU Compartilhada | ✅ **Otimizado** |
| **Desenvolvimento Local** | ~2-5 segundos | CPU Local | ✅ Suportado |
| **GPU Local** | ~0.5-2 segundos | GPU NVIDIA | ⚠️ Opcional |

### Otimizações Aplicadas

- ✅ **CPU Optimized**: Versões leves das bibliotecas
- ✅ **Memory Efficient**: Uso otimizado de RAM
- ✅ **Streamlit Cloud Ready**: Deploy direto sem configurações
- ✅ **Caching Inteligente**: Modelo pré-carregado para reduzir latência
- ✅ **Batch Processing**: Processamento eficiente para recursos limitados

## 🔧 Troubleshooting

### Problemas Comuns

**❌ "Modelo não encontrado"**
- Certifique-se de que a pasta `plate_detector_v1/weights/` existe
- Verifique se o arquivo `best.pt` está presente

**❌ "Erro de importação do torch/ultralytics"**
```bash
# Reinstalar dependências:
pip uninstall torch torchvision torchaudio ultralytics
pip install -r requirements.txt
```

**❌ "Memória insuficiente no Streamlit Cloud"**
- O sistema foi otimizado para funcionar com recursos limitados
- Se necessário, use imagens menores (o modelo aceita até 640x640)

**❌ "Lentidão no processamento"**
- No Streamlit Cloud: ~3-8 segundos por imagem (normal)
- Localmente: ~2-5 segundos por imagem (CPU)
- Para acelerar: considere usar GPU local (opcional)

### Performance na CPU

O sistema funciona perfeitamente com CPU:
- **Streamlit Cloud**: 3-8 segundos por imagem
- **Desenvolvimento local**: 2-5 segundos por imagem
- **Memória RAM**: ~2-4GB necessários

### Logs e Debug

Para ativar logs detalhados no código:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📁 Estrutura do Projeto

```
brazilian-license-plate-recognition/
│
├── 📁 images/                          # Imagens de teste
│   ├── DCAM0015_JPG_jpg.rf.72c8...jpg
│   ├── DCAM0019_JPG_jpg.rf.4fe1...jpg
│   └── ...
│
├── 📁 plate_detector_v1/               # Modelo treinado
│   ├── weights/
│   │   ├── best.pt                     # Melhor modelo
│   │   └── last.pt                     # Último modelo
│   ├── args.yaml                       # Hiperparâmetros
│   ├── results.csv                     # Métricas por época
│   └── results.png                     # Gráfico de resultados
│
├── 📁 notebooks/                       # Notebooks de treinamento
│   └── 1_YOLOv8_Training_Brazilian_Plates.ipynb
│
├── 📄 app.py                           # Aplicação Streamlit principal
├── 📄 requirements.txt                 # Dependências Python
├── 📄 README.md                        # Este arquivo
└── 📄 plate_detector_v1_summary.json   # Resumo do treinamento
```

## 🎯 Como Usar

### 1. Página Inicial
- Visão geral do sistema
- Principais características
- Gráfico de resultados do treinamento

### 2. Teste do Modelo
- **Seletor Visual**: Escolha imagens usando o `streamlit-image-select`
- **Detecção**: Clique em "Detectar Placas" para processar
- **Resultados**: Visualize bounding boxes e níveis de confiança

### 3. Análise de Resultados
- **Métricas**: Cards com principais indicadores de performance
- **Gráficos Interativos**: Evolução das métricas durante o treinamento
- **Análise de Perdas**: Curvas de treinamento detalhadas

### 4. Sobre o Modelo
- **Arquitetura**: Detalhes técnicos do YOLOv8
- **Hiperparâmetros**: Configurações utilizadas no treinamento
- **Processo**: Explicação passo a passo da detecção

### 5. Sobre os Dados
- **Dataset**: Características do conjunto de treinamento
- **Tipos de Placas**: Exemplos de diferentes formatos
- **Imagens de Teste**: Galeria das imagens disponíveis

## 🛠️ Desenvolvimento

### Estrutura do Modelo YOLOv8

O modelo utiliza a arquitetura YOLOv8 com as seguintes características:

- **Backbone**: CSPDarknet53 modificado
- **Neck**: PAN (Path Aggregation Network)
- **Head**: Cabeça de detecção YOLOv8
- **Tamanho**: Variante "small" (YOLOv8s)

### Processo de Treinamento

1. **Preparação dos Dados**: Dataset formatado no padrão YOLO
2. **Configuração**: Definição de hiperparâmetros
3. **Treinamento**: 300 épocas com early stopping
4. **Validação**: Avaliação em conjunto de validação
5. **Otimização**: Seleção do melhor modelo

### Métricas Utilizadas

- **Precision**: Fração de detecções corretas
- **Recall**: Fração de placas reais detectadas
- **mAP@50**: Mean Average Precision (IoU ≥ 0.5)
- **mAP@50-95**: Mean Average Precision (média IoU 0.5-0.95)

## 🔧 Personalização

### Imagens de Teste

**✅ As imagens de teste são carregadas automaticamente do GitHub!**

- O sistema busca imagens diretamente do repositório
- Não é necessário ter as imagens localmente
- Funciona perfeitamente no Streamlit Cloud
- Cache automático para melhor performance

### Adicionar Novas Imagens

Para adicionar suas próprias imagens de teste:

1. Faça upload via interface **"Upload"** na aba Detector
2. Ou, para adicionar permanentemente:
   - Coloque suas imagens na pasta `images/`
   - Adicione os nomes dos arquivos na lista `EXAMPLE_IMAGES` em `app.py`
   - Faça commit no GitHub
3. As imagens aparecem automaticamente no seletor

### Ajustar Parâmetros do Modelo

Para modificar a confiança mínima ou outros parâmetros:

```python
# Em app.py, linha 54
results = model(image, conf=0.5)  # Ajuste o threshold aqui
```

## 🔧 Troubleshooting

### Erro: `ImportError: libGL.so.1: cannot open shared object file`

Este erro ocorre quando o OpenCV não encontra as bibliotecas gráficas do sistema. **Solução:**

1. **No Streamlit Cloud**: O arquivo `packages.txt` já está configurado para instalar as dependências necessárias
2. **Localmente (Linux)**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   ```
3. **Localmente (Mac)**: Não é necessário, já funciona nativamente
4. **Localmente (Windows)**: Não é necessário, já funciona nativamente

### Erro: Conflito entre `opencv-python` e `opencv-python-headless`

**Solução:** O `requirements.txt` já está configurado para instalar o `opencv-python-headless` antes do `ultralytics`, evitando conflitos.

### Deploy travando no Streamlit Cloud

**Possíveis causas:**
- Tamanho do modelo muito grande
- Falta de memória durante a instalação

**Solução:** O repositório já está otimizado com versões CPU das bibliotecas, que são menores e mais rápidas para instalar.

## 📈 Melhorias Futuras

- [ ] Integração com OCR para leitura de caracteres
- [ ] Suporte a vídeos em tempo real
- [ ] API REST para integração com outros sistemas
- [ ] Aplicativo móvel complementar
- [ ] Otimização adicional do modelo para edge devices

## 🤝 Contribuição

Contribuições são bem-vindas! Siga estes passos:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👨‍💻 Autor

<div align="center">

**Sidnei Almeida**

[![GitHub](https://img.shields.io/badge/GitHub-sidnei--almeida-181717?style=for-the-badge&logo=github)](https://github.com/sidnei-almeida)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-saaelmeida93-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/saaelmeida93/)

Desenvolvedor especializado em Machine Learning e Computer Vision

</div>

## 🙏 Agradecimentos

- **Ultralytics**: Desenvolvedores do YOLOv8
- **Streamlit**: Framework para criação da interface
- **Comunidade Python**: Por bibliotecas e ferramentas excepcionais

## 📞 Suporte

Para suporte e dúvidas:

- 💬 Abra uma [Issue](https://github.com/sidnei-almeida/brazilian-license-plate-recognition/issues)
- 💼 Entre em contato via [LinkedIn](https://www.linkedin.com/in/saaelmeida93/)
- 📧 Discussões no [GitHub Discussions](https://github.com/sidnei-almeida/brazilian-license-plate-recognition/discussions)

---

⭐ **Se este projeto foi útil para você, considere dar uma estrela!** ⭐
