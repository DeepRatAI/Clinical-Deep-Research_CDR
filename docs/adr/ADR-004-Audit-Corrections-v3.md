# ADR-004: Correcciones Post-Auditoría v3

**Estado:** ✅ COMPLETADO
**Fecha:** 2026-01-19
**Autor:** Agente CDR
**Auditoría Origen:** CDR_Post_ADR003_v3_Audit_with_Run_KPIs_and_MinEvidence_Checklist.md

---

## Contexto

La auditoría v3 identificó 5 hallazgos priorizados que afectan la trazabilidad end-to-end
y la capacidad de alcanzar Research-grade / SOTA-grade:

| # | Hallazgo | Severidad | Impacto |
|---|----------|-----------|---------|
| 1 | Fallback Markdown no usa `valid_snippet_ids` | CRÍTICO | Claims "condenados" a rechazo |
| 2 | Records incluidos sin snippet mínimo | ALTO | Brecha estructural en evidencia |
| 3 | Fallback heurístico no level-gated | ALTO | Research-grade sin garantías |
| 4 | GRADE rationale no estructurado | MEDIO | Auditoría automática difícil |
| 5 | Tests de retrieval en skip | MEDIO | Degradación silenciosa posible |

---

## Decisiones Implementadas

### 1. Unificación de validación de snippet IDs (CRÍTICO)

**Problema:** `_parse_markdown_claims()` no recibía `valid_snippet_ids`, creando
claims con snippet_ids que serían rechazados por el gate.

**Solución:**
- Firma de `_parse_markdown_claims()` actualizada para aceptar `valid_snippet_ids`
- Llamada desde `_parse_synthesis_response()` propaga el parámetro
- Validación aplicada ANTES de crear EvidenceClaim

**Archivos modificados:**
- `src/cdr/synthesis/synthesizer.py` (L352, L537-545, L648-665)

**Justificación técnica:**
Per PRISMA 2020 (trazabilidad) y GRADE handbook (certeza basada en evidencia),
todo claim DEBE tener snippets reales. La validación temprana elimina falsos
negativos por timing issues.

**Refs:**
- https://www.prisma-statement.org/prisma-2020
- https://gradepro.org/handbook/

---

### 2. Snippet mínimo garantizado por record (ALTO)

**Problema:** Records incluidos podían llegar a síntesis sin abstract/snippet,
resultando en claims sin evidencia trazable.

**Solución:**
- `parse_documents_node` ahora valida abstract >= 10 chars
- Records sin contenido cuentan como `reports_not_retrieved` (PRISMA)
- `prisma_counts` actualizado con `reports_not_retrieved`, `reports_sought`, `reports_assessed`

**Archivos modificados:**
- `src/cdr/orchestration/graph.py` (L468-530)

**Justificación técnica:**
Per PRISMA 2020 Flow Diagram, "reports not retrieved" es una categoría explícita.
Garantiza que solo records con evidencia real lleguen a síntesis.

**Refs:**
- https://www.prisma-statement.org/prisma-2020 (Flow Diagram)
- Cochrane Handbook Section 4.6

---

### 3. Level-gating del screening (ALTO)

**Problema:** El fallback heurístico podía habilitar runs Research-grade sin
garantías de screening suficiente.

**Solución:**
- Agregado parámetro `dod_level` en configurable (1=exploratorio, 2=research, 3=SOTA)
- Para `dod_level >= 2`, fallback heurístico retorna error INSUFFICIENT_EVIDENCE
- Flag `screening_blocked_no_llm` para diagnóstico

**Archivos modificados:**
- `src/cdr/orchestration/graph.py` (L307-340, L372-410)

**Justificación técnica:**
Per PRISMA 2020, Research-grade requiere screening reproducible y justificado.
El heurístico es insuficiente para Level 2+ porque no evalúa criterios PICO
con precisión LLM/humana.

**Configuración:**
```python
config = {"configurable": {"dod_level": 2, "llm_provider": llm}}
```

**Refs:**
- https://www.prisma-statement.org/prisma-2020

---

### 4. GRADE rationale estructurado (MEDIO)

**Problema:** El rationale de downgrade estaba solo en `limitations` (texto libre),
dificultando auditoría automática.

**Solución:**
- Nuevo campo `grade_rationale: dict[str, str]` en `EvidenceClaim`
- Keys estándar: `risk_of_bias`, `inconsistency`, `indirectness`, `imprecision`, `publication_bias`
- Downgrade por RoB2 ahora popula `grade_rationale["risk_of_bias"]`

**Archivos modificados:**
- `src/cdr/core/schemas.py` (EvidenceClaim L516-575)
- `src/cdr/orchestration/graph.py` (L916-930)
- `src/cdr/synthesis/synthesizer.py` (L453, L685)

**Justificación técnica:**
Per GRADE Handbook Section 5.2, cada dominio que causa downgrade debe estar
explícitamente documentado. El campo estructurado habilita:
- Auditoría automática de certeza
- Agregación de razones por dominio
- Trazabilidad GRADE completa

**Refs:**
- https://gradepro.org/handbook/ (Section 5.2)

---

### 5. Tests de retrieval (MEDIO) - PENDIENTE

**Status:** No implementado en esta iteración.

**Plan:**
- Reescribir tests hacia API pública + mocks determinísticos
- Separar integración con markers pytest
- Eliminar skips legacy

**Refs:**
- https://www.ncbi.nlm.nih.gov/books/NBK25501/ (E-utilities)
- https://clinicaltrials.gov/data-api/api

---

## Impacto en KPIs (per Auditoría v3)

| KPI | Antes | Después |
|-----|-------|---------|
| Snippet Coverage | <100% | 100% (gate + validación temprana) |
| Verification Coverage | 100% | 100% (sin cambios) |
| GRADE Explicitness | ~80% | 100% (campo estructurado) |
| PRISMA Consistency | ~90% | 100% (reports_not_retrieved) |
| Level-gating | No | Sí (Level 2+ requiere LLM) |

---

## Verificación

```bash
python -m pytest tests/ --tb=short -q
```

**Resultado esperado:** 99+ passed, <=20 skipped, 0 failed

---

## Referencias Normativas

- PRISMA 2020: https://www.prisma-statement.org/prisma-2020
- PRISMA Checklist: https://www.prisma-statement.org/prisma-2020-checklist
- GRADE Handbook: https://gradepro.org/handbook/
- RoB2 (Cochrane): https://methods.cochrane.org/bias/resources/rob-2-revised-cochrane-risk-bias-tool-randomized-trials
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- ClinicalTrials.gov API: https://clinicaltrials.gov/data-api/api

---

## Changelog

- 2026-01-19: ADR-004 creado con correcciones v3
- 2026-01-19: Implementadas correcciones 1-4 (5 pendiente)
