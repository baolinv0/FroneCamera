import { FormEvent, useEffect, useState } from 'react'
import { AnalysisPanel } from './AnalysisPanel'
import { api } from './api'
import { PairingGrid } from './PairingGrid'
import type { AnalysisRecord, Pairing, PairingSnapshot, Project, ReportRecord, ReviewItem, Task } from './types'
import './styles.css'

type Tab = 'pairing' | 'analysis' | 'review' | 'reports'

export default function App() {
  const [projects, setProjects] = useState<Project[]>([])
  const [current, setCurrent] = useState<Project | null>(null)
  const [pairing, setPairing] = useState<Pairing | null>(null)
  const [snapshots, setSnapshots] = useState<PairingSnapshot[]>([])
  const [analysis, setAnalysis] = useState<AnalysisRecord[]>([])
  const [reviews, setReviews] = useState<ReviewItem[]>([])
  const [reports, setReports] = useState<ReportRecord[]>([])
  const [task, setTask] = useState<Task | null>(null)
  const [tab, setTab] = useState<Tab>('pairing')
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
    await refreshProject(project.id)
  }

  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    await api.createProject(String(data.get('name')))
    event.currentTarget.reset()
    await refreshProjects()
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
    setMessage('Device registered. Scan again to rebuild the pairing proposal.')
    event.currentTarget.reset()
    await refreshProject(current.id)
  }

  async function scan() {
    if (!current) return
    setPairing(await api.scan(current.id))
    setSnapshots([])
    setMessage('Pairing proposal rebuilt. Review every row before confirmation.')
  }

  async function updateCell(groupId: string, deviceId: string, imageId: string | null) {
    if (current && pairing) setPairing(await api.updatePairing(current.id, { expected_version: pairing.version, group_id: groupId, device_id: deviceId, image_id: imageId }))
  }

  async function confirm() {
    if (!current || !pairing) return
    setPairing(await api.confirm(current.id, pairing.version))
    setSnapshots(await api.pairingSnapshots(current.id))
    setMessage('Pairing snapshot frozen. Editing a cell will create a new version after reconfirmation.')
  }

  async function run() {
    if (!current) return
    const queued = await api.run(current.id)
    setTask(queued)
    setMessage(`Queued ${queued.id}. Ensure the Linux worker is running.`)
  }

  async function resolve(item: ReviewItem, status: string) {
    await api.resolveReview(item.id, status)
    if (current) {
      setReviews(await api.reviews(current.id))
      setCurrent(await api.getProject(current.id))
    }
  }

  async function poll() {
    if (!task) return
    const refreshed = await api.task(task.id)
    setTask(refreshed)
    if (current && ['SUCCEEDED', 'FAILED'].includes(refreshed.status)) await refreshProject(current.id)
  }

  async function finalize() {
    if (!current) return
    const report = await api.finalizeReport(current.id)
    setReports(await api.reports(current.id))
    setCurrent(await api.getProject(current.id))
    setMessage(`Final report ${report.version} created.`)
  }

  return <main className="shell">
    <header>
      <div><h1>FroneCamera</h1><p>Evidence-oriented front-camera portrait evaluation</p></div>
      <label>API token<input aria-label="API token" placeholder="optional" onChange={event => localStorage.setItem('fronecamera-token', event.target.value)} /></label>
    </header>
    {error && <pre className="error">{error}</pre>}
    <section className="card compact"><h2>Create project</h2><form onSubmit={event => create(event).catch(error => setError(String(error)))}><input name="name" required placeholder="Project name"/><button>Create</button></form></section>
    <section className="card compact"><h2>Projects</h2>{projects.map(project => <button className="project" key={project.id} onClick={() => open(project).catch(error => setError(String(error)))}><b>{project.name}</b><span>{project.status}</span></button>)}</section>
    {current && <section className="card workspace">
      <div className="project-heading"><div><h2>{current.name}</h2><span className="badge">{current.status}</span></div><button className="secondary" onClick={() => refreshProject(current.id).catch(error => setError(String(error)))}>Refresh</button></div>
      <form onSubmit={event => addDevice(event).catch(error => setError(String(error)))} className="device-form"><input name="name" required placeholder="Device label"/><input name="model" placeholder="Canonical model"/><input name="folder" required placeholder="Linux folder path"/><button>Add device</button></form>
      <div className="actions"><button onClick={() => scan().catch(error => setError(String(error)))}>Scan</button><button onClick={() => confirm().catch(error => setError(String(error)))} disabled={!pairing}>Confirm pairing</button><button onClick={() => run().catch(error => setError(String(error)))} disabled={!pairing?.confirmed}>Queue evaluation</button><a href={`/api/projects/${current.id}/export`}>Export project</a></div>
      {message && <p className="notice">{message}</p>}
      {task && <div className="task"><b>{task.status}</b><span>attempts: {task.attempts ?? 0}</span><button onClick={() => poll().catch(error => setError(String(error)))}>Refresh task</button>{task.error && <pre>{task.error}</pre>}</div>}
      <nav className="tabs">{(['pairing', 'analysis', 'review', 'reports'] as Tab[]).map(name => <button className={tab === name ? 'active' : 'secondary'} key={name} onClick={() => setTab(name)}>{name}</button>)}</nav>
      {tab === 'pairing' && <div><div className="section-heading"><h3>Matched scene groups</h3><span>{snapshots.length} frozen snapshot(s)</span></div>{pairing ? <PairingGrid pairing={pairing} onChange={(groupId, deviceId, imageId) => updateCell(groupId, deviceId, imageId).catch(error => setError(String(error)))}/> : <p className="muted">Register at least two device folders, then scan.</p>}</div>}
      {tab === 'analysis' && <div><h3>Structured evidence</h3><AnalysisPanel records={analysis}/></div>}
      {tab === 'review' && <div><h3>Human review queue</h3>{!reviews.length && <p className="muted">No review items.</p>}{reviews.map(item => <article key={item.id}><div className="review-heading"><b>{item.category}</b><span>{item.priority}</span><span>{item.status}</span></div><pre>{JSON.stringify(item.payload, null, 2)}</pre>{item.status === 'open' && <div><button onClick={() => resolve(item, 'accepted')}>Accept</button><button onClick={() => resolve(item, 'insufficient_evidence')}>Insufficient evidence</button><button onClick={() => resolve(item, 'rejected')}>Reject</button></div>}</article>)}</div>}
      {tab === 'reports' && <div><div className="section-heading"><h3>Reports</h3><button onClick={() => finalize().catch(error => setError(String(error)))} disabled={reviews.some(item => item.status === 'open') || !reports.length}>Finalize</button></div>{!reports.length && <p className="muted">Run an evaluation to generate a draft report.</p>}{reports.map(report => <article key={report.id}><div><b>v{report.version}</b> <span className="badge">{report.status}</span></div><div className="actions"><a href={`/api/reports/${report.id}/html`} target="_blank" rel="noreferrer">Open HTML</a><a href={`/api/reports/${report.id}/pdf`} target="_blank" rel="noreferrer">Open PDF</a></div></article>)}</div>}
    </section>}
  </main>
}
