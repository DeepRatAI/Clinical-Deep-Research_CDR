# ADR-003: Correcciones Post-Auditoría Formal v2

**Fecha**: 2026-01-19
**Estado**: ✅ COMPLETADO
**Contexto**: Auditoría formal identificó errores en correcciones anteriores (ADR-002)

---

## Resumen Ejecutivo

La auditoría formal reveló que las correcciones en ADR-002 introdujeron un error crítico:
se rechazaban snippet_ids con formato `_snip_0` que es el formato REAL del sistema.

Esto degrada claims válidos y rompe trazabilidad sin motivo.

---

## Hallazgos Críticos (de la auditoría)

### 1. Snippets REALES filtrados indebidamente

**Problema**: El parser en synthesizer.py rechaza `snippet_id.endswith("_snip_0")`,
pero `{record_id}_snip_0` es el formato canónico generado por `parse_documents_node`.

**Evidencia**: 
- `graph.py` línea 493: `snippet_id=f"{record.record_id}_snip_0"`
- `synthesizer.py` línea 370: `if snippet_id and not snippet_id.endswith("_snip_0")`

**Impacto**: 
- Claims válidos con snippets reales son rechazados
- Degradación artificial de resultados (falsos negativos)
- Ruptura de trazabilidad PRISMA

**Corrección**: Aceptar snippet_id tal como viene; el gate en graph.py ya valida
existencia en state.snippets.

### 2. Placeholders contaminan el flujo

**Problema**: Se crean IDs con sufijo `_placeholder` que no corresponden a snippets reales.

**Impacto**: Claims con soporte pseudo-evidencial pasan al gate y son rechazados,
pero contaminan el flujo y dificultan debugging.

**Corrección**: NO crear claims si no hay supporting_snippet_ids reales.
El gate en graph.py es el validador final de trazabilidad.

### 3. Screening heurístico muy laxo

**Problema**: Umbral PICO de 0.2 (20%) es muy bajo, puede inflar inclusiones.

**Corrección**: Exigir al menos 2 componentes PICO (P+I o P+O) y umbral 0.4.

---

## Decisiones de Diseño

### D1: Formato canónico de snippet_id

```
{record_id}_snip_{index}
```

Donde:
- `record_id`: ID único del registro (ej: `pubmed_12345678`)
- `index`: Índice del snippet (0 para abstract principal, 1+ para párrafos)

Este formato es generado por `parse_documents_node` y debe ser aceptado sin filtros.

### D2: Flujo de validación de snippets

```
LLM genera claims → synthesizer acepta snippet_ids → gate valida existencia
```

La validación de existencia ocurre SOLO en el gate formal de `synthesize_node`,
no en el parser. El synthesizer debe confiar en los IDs proporcionados.

### D3: Política de claims sin snippets

Si el LLM no proporciona snippet_ids válidos:
1. Intentar extraer de `supporting_studies` con formato `{study_id}_snip_0`
2. Si no hay studies, el claim NO se crea
3. NO usar placeholders ni fallbacks

---

## Changelog

| Fecha | Cambio | Estado |
|-------|--------|--------|
| 2026-01-19 | Identificación de errores en ADR-002 | ✅ |
| 2026-01-19 | Corrección filtro _snip_0 - ahora acepta formato real | ✅ |
| 2026-01-19 | Eliminación placeholders - usa _snip_0 real | ✅ |
| 2026-01-19 | Ajuste screening: threshold 0.4 + 2 componentes PICO | ✅ |
| 2026-01-19 | GRADE rationale estandarizado con ref GRADE handbook | ✅ |
| 2026-01-19 | Validación tests: 99 passed, 18 skipped, 0 failed | ✅ |
| 2026-01-19 | POST-AUDIT: Early snippet validation en synthesizer | ✅ |
| 2026-01-19 | POST-AUDIT: Warning "Level 1 only" en fallback heurístico | ✅ |

---

## Correcciones Post-Auditoría (2026-01-19)

### Hallazgo Crítico: Snippet IDs inferidos sin validación

**Problema**: El synthesizer generaba `{study_id}_snip_0` para `supporting_studies`
sin verificar si el snippet realmente existía. Esto creaba claims que luego eran
rechazados por el gate, causando falsos negativos.

**Solución**: 
1. `synthesize()` ahora recibe `valid_snippet_ids: set[str] | None`
2. `_parse_synthesis_response()` filtra snippet_ids ANTES de crear claims
3. Claims con snippets inválidos se rechazan temprano, no en el gate

**Código**:
```python
# synthesize() ahora acepta valid_snippet_ids
result = synthesizer.synthesize(
    state.study_cards,
    state.rob2_results,
    state.question,
    valid_snippet_ids=valid_snippet_ids,  # NEW
)
```

### Hallazgo Alto: Fallback heurístico sin limitación

**Problema**: El fallback heurístico era usado sin warning, pudiendo sesgar PRISMA
en revisiones Research-grade.

**Solución**: Añadido warning explícito y atributo de tracing:
```python
print("[Screen] ⚠️ WARNING: No LLM available - using heuristic screening (Level 1 only)")
span.set_attribute("screening_warning", "heuristic_fallback_level1_only")
```

### Hallazgo Medio: GRADE rationale

**Estado**: El GRADE rationale está en `limitations` con formato estandarizado.
El schema actual no tiene campo `grade_rationale` dedicado.

**Decisión**: Mantener en `limitations` para esta versión. Para v2.0, considerar
añadir campo `grade_rationale: dict[str, str]` al EvidenceClaim schema.

---

## Detalles de Implementación

### Corrección 1: Aceptar _snip_0 como formato REAL

**Archivo**: `src/cdr/synthesis/synthesizer.py`

**Antes** (INCORRECTO):
```python
if snippet_id and not snippet_id.endswith("_snip_0"):
    supporting_snippet_ids.append(snippet_id)
```

**Después** (CORRECTO):
```python
if snippet_id:
    supporting_snippet_ids.append(snippet_id)
```

**Razón**: `{record_id}_snip_0` es el formato canónico generado por `parse_documents_node` 
en `graph.py` línea 504.

### Corrección 2: Eliminar placeholders

**Archivo**: `src/cdr/synthesis/synthesizer.py`

**Antes** (INCORRECTO):
```python
supporting_snippet_ids.append(f"{study_id}_placeholder")
```

**Después** (CORRECTO):
```python
supporting_snippet_ids.append(f"{study_id}_snip_0")
```

**Razón**: El gate en `synthesize_node` valida existencia en `state.snippets`.
Los IDs deben usar el formato real para que la validación funcione.

### Corrección 3: Screening más estricto

**Archivo**: `src/cdr/orchestration/graph.py`

**Cambios**:
- `_calculate_pico_match_score()` ahora retorna `(score, components_matched)`
- Threshold elevado de 0.2 a 0.4
- Requiere mínimo 2 componentes PICO para inclusión
- Mensaje de exclusión detallado con score y componentes

**Razón**: Per PRISMA 2020 y auditoría formal, un solo componente PICO es insuficiente
para determinar elegibilidad.

### Corrección 4: GRADE rationale estandarizado

**Archivo**: `src/cdr/orchestration/graph.py`

**Texto estandarizado**:
```
GRADE DOWNGRADE (Risk of Bias): RoB2 assessment failed for 
studies [X, Y]. Per GRADE handbook section 5.2: unable to assess 
risk of bias leads to mandatory downgrade. Certainty reduced 
from {old} to {new}.
```

**Razón**: Per GRADE handbook section 5.2 y auditoría formal.

---

## Referencias

- PRISMA 2020: https://www.prisma-statement.org/prisma-2020
- GRADE Handbook: https://gradepro.org/handbook/
- Auditoría: CDR_Formal_Audit_and_Subsystem_Acceptance_Checklists.md
