#!/usr/bin/env python3
"""
Script de instalação e configuração do Brazilian License Plate Recognition System
Autor: Sidnei Almeida
GitHub: https://github.com/sidnei-almeida
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ é necessário. Você tem:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} encontrado")
    return True

def check_requirements():
    """Verifica se todas as dependências estão instaladas"""
    try:
        import streamlit
        import ultralytics
        import plotly
        import cv2
        import numpy
        import pandas
        import PIL
        print("✅ Todas as dependências principais estão instaladas")
        print("✅ Sistema otimizado para Streamlit Cloud (CPU)")
        return True
    except ImportError as e:
        print(f"❌ Dependência ausente: {e}")
        return False

def install_requirements():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências (otimizadas para Streamlit Cloud)...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        print("✅ Sistema pronto para Streamlit Cloud")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro na instalação: {e}")
        return False

def check_model_files():
    """Verifica se os arquivos do modelo existem"""
    model_path = "plate_detector_v1/weights/best.pt"
    if os.path.exists(model_path):
        print("✅ Modelo treinado encontrado")
        return True
    else:
        print(f"⚠️  Modelo não encontrado em {model_path}")
        print("ℹ️  Você pode treinar o modelo usando o notebook fornecido")
        return False

def run_application():
    """Executa a aplicação Streamlit"""
    print("🚀 Iniciando aplicação...")
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Aplicação interrompida pelo usuário")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar aplicação: {e}")

def main():
    """Função principal"""
    print("🚗 Brazilian License Plate Recognition - Setup")
    print("=" * 50)

    # Verificar versão do Python
    if not check_python_version():
        sys.exit(1)

    # Verificar se está no diretório correto
    if not os.path.exists("requirements.txt"):
        print("❌ Execute este script a partir do diretório raiz do projeto")
        sys.exit(1)

    # Verificar dependências
    if not check_requirements():
        print("📦 Instalando dependências...")
        if not install_requirements():
            sys.exit(1)

    # Verificar arquivos do modelo
    check_model_files()

    print("\n🎯 Setup concluído! Iniciando aplicação...")
    print("🌐 Aplicação disponível em: http://localhost:8501")
    print("🚀 Sistema otimizado para Streamlit Cloud")
    print("📖 Consulte o README.md para deploy na nuvem")
    print("=" * 50)

    run_application()

if __name__ == "__main__":
    main()
