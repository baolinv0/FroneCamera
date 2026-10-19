import type { AnalysisRecord } from './types'

function hasDiagnostic(payload: unknown): boolean {
  return typeof payload === 'object' && payload !== null && typeof (payload as Record<string, unknown>).diagnostic_path === 'string'
}

export function AnalysisPanel({ records }: { records: AnalysisRecord[] }) {
  const grouped = records.reduce<Record<string, AnalysisRecord[]>>((result, record) => {
    ;(result[record.kind] ??= []).push(record)
    return result
  }, {})
  if (!records.length) return <p className="muted">No analysis results yet.</p>
  return <div className="analysis-groups">
    {Object.entries(grouped).sort(([left], [right]) => left.localeCompare(right)).map(([kind, items]) =>
      <section className="analysis-group" key={kind}>
        <h4>{kind}</h4>
        {items.map(item => <details key={item.id}>
          <summary>{item.scene_group_id ?? 'project-level'}{item.image_id ? ` · image ${item.image_id.slice(0, 8)}` : ''}</summary>
          {hasDiagnostic(item.payload) && <a href={`/api/analysis/${item.id}/diagnostic`} target="_blank" rel="noreferrer">Open diagnostic</a>}
          <pre>{JSON.stringify(item.payload, null, 2)}</pre>
        </details>)}
      </section>
    )}
  </div>
}
