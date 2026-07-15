import { FormEvent, useEffect, useState } from 'react'
import { AnalysisPanel } from './AnalysisPanel'
import { api } from './api'
import { PairingGrid } from './PairingGrid'
import type { AnalysisRecord, EvaluationMode, Pairing, PairingSnapshot, Project, ReportRecord, ReportShare, ReviewItem, Task } from './types'
import { WorkflowSteps } from './WorkflowSteps'
import './styles.css'

type AdvancedTab = 'pairing' | 'analysis' | 'review' | 'reports'
const terminalTaskStates = new Set(['SUCCEEDED', 'FAILED'])

export default function App() {
  const [projects, setProjects] = useState<Project[]>([])
  const [current, setCurrent] = useState<Project | null>(null)
  const [pairing, setPairing] = useState<Pairing | null>(null)
  const [snapshots, setSnapshots] = useState<PairingSnapshot[]>([])
  const [analysis, setAnalysis] = useState<AnalysisRecord[]>([])
  const [reviews, setReviews] = useState<ReviewItem[]>([])
  const [reports, setReports] = useState<ReportRecord[]>([])
  const [shares, setShares] = useState<Record<string, ReportShare>>({})
  const [task, setTask] = useState<Task | null>(null)
  const [mode, setMode] = useState<EvaluationMode>('quick')
  const [advancedTab, setAdvancedTab] = useState<AdvancedTab>('pairing')
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const refreshProjects = async () => setProjects(await api.listProjects())
  useEffect(() => { refreshProjects().catch(error => setError(String(error))) }, [])

  async function refreshProject(projectId: string) {
    const project = await api.getProject(projectId)
    setCurrent(project)
    const results = await Promise.allSettled([
      api.pairing(projectId),
      api.pairingSnapshots(projectId),
      api.analysis(projectId),
      api.reviews(projectId),
      api.reports(projectId)
    ])
    if (results[0].status === 'fulfilled') setPairing(results[0].value); else setPairing(null)
    setSnapshots(results[1].status === 'fulfilled' ? results[1].value : [])
    setAnalysis(results[2].status === 'fulfilled' ? results[2].value : [])
    setReviews(results[3].status === 'fulfilled' ? results[3].value : [])
    setReports(results[4].status === 'fulfilled' ? results[4].value : [])
  }

  async function open(project: Project) {
    setError('')
    setMessage('')
    setTask(null)
    setShares({})
    await refreshProject(project.id)
  }

  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    const project = await api.createProject(String(data.get('name')))
    event.currentTarget.reset()
    await refreshProjects()
    await open(project)
  }

  async function addDevice(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!current) return
    const data = new FormData(event.currentTarget)
    await api.addDevice(current.id, {
      name: String(data.get('name')),
      folder_path: String(data.get('folder')),
      canonical_model: String(data.get('model') || '') || undefined
    })
    setMessage('Device registered. Add the remaining devices, then scan the folders.')
    event.currentTarget.reset()
    await refreshProject(current.id)
  }

  async function scan() {
    if (!current) return
    setPairing(await api.scan(current.id))
    setSnapshots([])
    setMessage('The initial scene matching is ready. Check the rows and confirm the test data.')
  }

  async function updateCell(groupId: string, deviceId: string, imageId: string | null) {
    if (current && pairing) {
      setPairing(await api.updatePairing(current.id, {
        expected_version: pairing.version,
        group_id: groupId,
        device_id: deviceId,
        image_id: imageId
      }))
    }
  }

  async function confirm() {
    if (!current || !pairing) return
    setPairing(await api.confirm(current.id, pairing.version))
    setSnapshots(await api.pairingSnapshots(current.id))
    setMessage('Test data confirmed. You can now run the full evaluation with one button.')
  }

  async function runFullEvaluation() {
    if (!current) return
    const queued = await api.runFullEvaluation(current.id, mode)
    setTask(queued)
    setMessage(mode === 'quick'
      ? 'Quick evaluation queued. A shareable final report will be generated automatically.'
      : 'Professional evaluation queued. A draft report will be generated for evidence review.')
  }

  async function pollTask(taskId: string) {
    const refreshed = await api.task(taskId)
    setTask(refreshed)
    if (current && terminalTaskStates.has(refreshed.status)) {
      await refreshProject(current.id)
      await refreshProjects()
      setMessage(refreshed.status === 'SUCCEEDED'
        ? 'Evaluation complete. The report is ready.'
        : `Evaluation failed: ${refreshed.error || 'unknown error'}`)
    }
  }

  useEffect(() => {
    if (!task || terminalTaskStates.has(task.status)) return
    const timer = window.setInterval(() => {
      pollTask(task.id).catch(error => setError(String(error)))
    }, 2000)
    return () => window.clearInterval(timer)
  }, [task?.id, task?.status, current?.id])

  async function resolve(item: ReviewItem, status: string) {
    await api.resolveReview(item.id, status)
    if (current) await refreshProject(current.id)
  }

  async function finalize() {
    if (!current) return
    const report = await api.finalizeReport(current.id)
    await refreshProject(current.id)
    setMessage(`Final report ${report.version} created.`)
  }

  async function createShare(report: ReportRecord) {
    const share = await api.shareReport(report.id)
    setShares(previous => ({ ...previous, [report.id]: share }))
    setMessage('Read-only link created. Other computers on the same network can open it.')
  }

  const finalReport = reports.find(report => report.status === 'final') || reports[0]
  const workflowStep: 1 | 2 | 3 | 4 = !pairing?.confirmed
    ? 1
    : !task && !reports.length
      ? 2
      : task && !terminalTaskStates.has(task.status)
        ? 2
        : !reports.length
          ? 3
          : 4

  return <main className="shell">
    <header>
      <div><h1>FroneCamera</h1><p>Four-step front-camera portrait evaluation</p></div>
      <label>API token<input aria-label="API token" placeholder="optional" onChange={event => localStorage.setItem('fronecamera-token', event.target.value)} /></label>
    </header>

    <WorkflowSteps currentStep={workflowStep} />
    {error && <pre className="error">{error}</pre>}
    {message && <p className="notice">{message}</p>}

    <section className="card compact project-picker">
      <div>
        <h2>Create or open a test</h2>
        <form onSubmit={event => create(event).catch(error => setError(String(error)))}>
          <input name="name" required placeholder="Test name" />
          <button>Create test</button>
        </form>
      </div>
      <div className="project-list">
        {projects.map(project => <button className="project" key={project.id} onClick={() => open(project).catch(error => setError(String(error)))}>
          <b>{project.name}</b><span>{project.status}</span>
        </button>)}
      </div>
    </section>

    {current && <section className="workspace">
      <div className="project-heading card compact">
        <div><h2>{current.name}</h2><span className="badge">{current.status}</span></div>
        <button className="secondary" onClick={() => refreshProject(current.id).catch(error => setError(String(error)))}>Refresh</button>
      </div>

      <section className={`card workflow-card ${workflowStep === 1 ? 'active-step' : ''}`}>
        <div className="step-heading"><span>1</span><div><h3>Create test data</h3><p>Register each phone folder, scan the images, and confirm the matched scenes.</p></div></div>
        <form onSubmit={event => addDevice(event).catch(error => setError(String(error)))} className="device-form">
          <input name="name" required placeholder="Device label" />
          <input name="model" placeholder="Canonical model" />
          <input name="folder" required placeholder="Linux folder path" />
          <button>Add device</button>
        </form>
        <div className="device-summary">
          {(current.devices || []).map(device => <span key={device.id}>{device.name}</span>)}
        </div>
        <div className="actions">
          <button onClick={() => scan().catch(error => setError(String(error)))} disabled={(current.devices?.length || 0) < 2}>Scan and match scenes</button>
          <button onClick={() => confirm().catch(error => setError(String(error)))} disabled={!pairing || pairing.confirmed}>Confirm test data</button>
          {pairing && <span className="muted">{pairing.groups.length} matched group(s), {snapshots.length} frozen snapshot(s)</span>}
        </div>
        {pairing && !pairing.confirmed && <PairingGrid pairing={pairing} onChange={(groupId, deviceId, imageId) => updateCell(groupId, deviceId, imageId).catch(error => setError(String(error)))} />}
        {pairing?.confirmed && <p className="success-line">✓ Test data confirmed.</p>}
      </section>

      <section className={`card workflow-card ${workflowStep === 2 ? 'active-step' : ''}`}>
        <div className="step-heading"><span>2</span><div><h3>Run model evaluation</h3><p>The server runs image audit, objective metrics, dual-model review, evidence adjudication, external corroboration, and attribution.</p></div></div>
        <div className="mode-grid">
          <label className={mode === 'quick' ? 'selected' : ''}>
            <input type="radio" name="mode" checked={mode === 'quick'} onChange={() => setMode('quick')} />
            <b>Quick mode</b>
            <small>Automatically creates a final report. Unresolved low-confidence items remain clearly flagged.</small>
          </label>
          <label className={mode === 'professional' ? 'selected' : ''}>
            <input type="radio" name="mode" checked={mode === 'professional'} onChange={() => setMode('professional')} />
            <b>Professional mode</b>
            <small>Creates a draft and requires key evidence conflicts to be reviewed before finalization.</small>
          </label>
        </div>
        <button className="primary-action" onClick={() => runFullEvaluation().catch(error => setError(String(error)))} disabled={!pairing?.confirmed || Boolean(task && !terminalTaskStates.has(task.status))}>Run full evaluation</button>
        {task && <div className="task">
          <b>{task.status}</b><span>attempts: {task.attempts ?? 0}</span>
          {!terminalTaskStates.has(task.status) && <span>Automatic refresh every 2 seconds</span>}
          {task.error && <pre>{task.error}</pre>}
        </div>}
      </section>

      <section className={`card workflow-card ${workflowStep === 3 ? 'active-step' : ''}`}>
        <div className="step-heading"><span>3</span><div><h3>Generate test report</h3><p>Quick mode finalizes automatically. Professional mode retains the evidence review gate.</p></div></div>
        {!reports.length && <p className="muted">Run the evaluation to generate the report bundle.</p>}
        {reports.map(report => <article className="report-row" key={report.id}>
          <div><b>Report v{report.version}</b> <span className="badge">{report.status}</span></div>
          <div className="actions">
            <a href={`/api/reports/${report.id}/html`} target="_blank" rel="noreferrer">Internal HTML</a>
            <a href={`/api/reports/${report.id}/pdf`} target="_blank" rel="noreferrer">PDF</a>
          </div>
        </article>)}
        {reports.some(report => report.status === 'draft') && <button onClick={() => finalize().catch(error => setError(String(error)))} disabled={reviews.some(item => item.status === 'open')}>Finalize professional report</button>}
      </section>

      <section className={`card workflow-card ${workflowStep === 4 ? 'active-step' : ''}`}>
        <div className="step-heading"><span>4</span><div><h3>View on another computer</h3><p>Create a signed read-only link and open it from a browser that can reach this Linux server.</p></div></div>
        {!finalReport && <p className="muted">A report must be generated first.</p>}
        {finalReport && <div className="share-panel">
          <div><b>Report v{finalReport.version}</b><span className="badge">{finalReport.status}</span></div>
          <button onClick={() => createShare(finalReport).catch(error => setError(String(error)))}>Create read-only link</button>
          {shares[finalReport.id] && <>
            <input readOnly value={`${window.location.origin}${shares[finalReport.id].url}`} aria-label="Read-only report URL" />
            <div className="actions">
              <a href={shares[finalReport.id].url} target="_blank" rel="noreferrer">Open shared report</a>
              <a href={shares[finalReport.id].pdf_url} target="_blank" rel="noreferrer">Open shared PDF</a>
            </div>
          </>}
        </div>}
      </section>

      <details className="card advanced">
        <summary>Advanced review and engineering controls</summary>
        <div className="actions"><a href={`/api/projects/${current.id}/export`}>Export project</a></div>
        <nav className="tabs">{(['pairing', 'analysis', 'review', 'reports'] as AdvancedTab[]).map(name => <button className={advancedTab === name ? 'active' : 'secondary'} key={name} onClick={() => setAdvancedTab(name)}>{name}</button>)}</nav>
        {advancedTab === 'pairing' && <div><h3>Matched scene groups</h3>{pairing ? <PairingGrid pairing={pairing} onChange={(groupId, deviceId, imageId) => updateCell(groupId, deviceId, imageId).catch(error => setError(String(error)))} /> : <p className="muted">No pairing proposal.</p>}</div>}
        {advancedTab === 'analysis' && <div><h3>Structured evidence</h3><AnalysisPanel records={analysis} /></div>}
        {advancedTab === 'review' && <div><h3>Human review queue</h3>{!reviews.length && <p className="muted">No review items.</p>}{reviews.map(item => <article key={item.id}><div className="review-heading"><b>{item.category}</b><span>{item.priority}</span><span>{item.status}</span></div><pre>{JSON.stringify(item.payload, null, 2)}</pre>{item.status === 'open' && <div><button onClick={() => resolve(item, 'accepted')}>Accept</button><button onClick={() => resolve(item, 'insufficient_evidence')}>Insufficient evidence</button><button onClick={() => resolve(item, 'rejected')}>Reject</button></div>}</article>)}</div>}
        {advancedTab === 'reports' && <div><h3>All report versions</h3>{reports.map(report => <pre key={report.id}>{JSON.stringify(report, null, 2)}</pre>)}</div>}
      </details>
    </section>}
  </main>
}
