# CDR - LLM Provider Setup

## Estado Actual (2026-01-27)

⚠️ **PROBLEMA**: Los créditos gratuitos de HuggingFace ($0.10/mes) se agotaron.
El sistema devuelve error 402 (Payment Required) al intentar usar el Router API.

## Solución Inmediata: Configurar Groq (GRATIS)

Groq ofrece un tier gratuito muy generoso:
- **Llama 3.1 8B**: 14,400 requests/día, 6,000 tokens/minuto
- **Llama 3.3 70B**: 1,000 requests/día, 12,000 tokens/minuto

### Pasos:

1. **Obtener API Key de Groq** (gratuita):
   - Ir a: https://console.groq.com/keys
   - Crear cuenta (o usar GitHub/Google)
   - Generar nueva API key

2. **Configurar en CDR**:
   ```bash
   cd /home/gonzalor/Desktop/Edicion_total_repositorios/cdr
   
   # Editar .env y agregar la key:
   echo "GROQ_API_KEY=gsk_xxx..." >> .env
   ```

3. **Verificar que funciona**:
   ```bash
   curl https://api.groq.com/openai/v1/chat/completions \
     -H "Authorization: Bearer $GROQ_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"llama-3.3-70b-versatile","messages":[{"role":"user","content":"Hi"}],"max_tokens":10}'
   ```

4. **Reiniciar servidor CDR**:
   ```bash
   pkill -f uvicorn
   cd /home/gonzalor/Desktop/Edicion_total_repositorios/cdr
   source .venv/bin/activate
   nohup python3 -m uvicorn cdr.api.routes:create_app --factory --host 0.0.0.0 --port 8000 > /tmp/cdr_backend.log 2>&1 &
   ```

## Alternativa: Recargar Créditos HuggingFace

1. Ir a: https://huggingface.co/settings/billing
2. Agregar método de pago
3. Los créditos se descontarán automáticamente del balance

## Fallback Automático

El sistema CDR ahora soporta fallback automático:
1. **HuggingFace** (si HF_TOKEN está configurado)
2. **Groq** (si GROQ_API_KEY está configurado)
3. **OpenAI** (si OPENAI_API_KEY está configurado)
4. **Anthropic** (si ANTHROPIC_API_KEY está configurado)

Si HuggingFace falla con 402, el sistema intentará Groq automáticamente si está configurado.

## Modelos Recomendados

### HuggingFace Router (cuando hay créditos):
- `Qwen/Qwen2.5-72B-Instruct` - Best quality (default)
- `meta-llama/Llama-3.3-70B-Instruct` - Balanced
- `google/gemma-3-27b-it` - Fast
- `meta-llama/Llama-3.1-8B-Instruct` - Fastest (for bulk ops)

### Groq (FREE - recomendado):
- `llama-3.3-70b-versatile` - Best quality
- `llama-3.1-8b-instant` - Fast, high rate limit

## Cambios Técnicos Realizados

1. **Retry Logic**: Solo reintentos para errores transitorios (429, 500, 502, 503, 504)
2. **No Retry para 402**: Payment Required se detecta inmediatamente, no se reintenta
3. **Throttling**: Semáforo de 3 requests concurrentes para evitar 500s
4. **Fallback Factory**: `create_provider_with_fallback()` intenta proveedores en orden

## Estado de Tests

✅ **495 tests pasando** (2026-01-27)

## Archivos Modificados

- `src/cdr/llm/huggingface_provider.py` - Retry y throttling
- `src/cdr/llm/factory.py` - Fallback automático
- `src/cdr/api/routes.py` - Usa fallback
- `src/cdr/llm/__init__.py` - Exports actualizados
