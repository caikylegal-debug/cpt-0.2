#!/bin/bash
set -e

mkdir -p model

echo "=== ZylunCPT 0.2 API ==="

if [ ! -f "model/zyluncpt_tokenizer.json" ]; then
  echo "ERRO: model/zyluncpt_tokenizer.json não encontrado."
  ls -lah model
  exit 1
fi

if [ ! -f "model/model.safetensors" ]; then
  echo "Baixando model.safetensors do Google Drive..."
  gdown "1QRE7jGNoP961c18s2wYj5GalVnAH4yaI" -O model/model.safetensors
fi

echo "Arquivos do modelo:"
ls -lh model

echo "Iniciando API..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
