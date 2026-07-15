import type { AnalysisRecord, Pairing, PairingSnapshot, Project, ReportRecord, ReviewItem, Task } from './types'

const token = () => localStorage.getItem('fronecamera-token') || ''

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body) headers.set('Content-Type', 'application/json')
  if (token()) headers.set('Authorization', `Bearer ${token()}`)
  const response = await fetch(path, { ...init, headers })
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json() as Promise<T>
}

export const api = {
  listProjects: () => request<Project[]>('/api/projects'),
  getProject: (projectId: string) => request<Project>(`/api/projects/${projectId}`),
  createProject: (name: string) => request<Project>('/api/projects', { method: 'POST', body: JSON.stringify({ name }) }),
  addDevice: (projectId: string, body: { name: string; folder_path: string; canonical_model?: string }) => request(`/api/projects/${projectId}/devices`, { method: 'POST', body: JSON.stringify(body) }),
  scan: (projectId: string) => request<Pairing>(`/api/projects/${projectId}/scan`, { method: 'POST' }),
  pairing: (projectId: string) => request<Pairing>(`/api/projects/${projectId}/pairing`),
  pairingSnapshots: (projectId: string) => request<PairingSnapshot[]>(`/api/projects/${projectId}/pairing/snapshots`),
  updatePairing: (projectId: string, body: { expected_version: number; group_id: string; device_id: string; image_id: string | null }) => request<Pairing>(`/api/projects/${projectId}/pairing`, { method: 'PUT', body: JSON.stringify(body) }),
  confirm: (projectId: string, expectedVersion: number) => request<Pairing>(`/api/projects/${projectId}/pairing/confirm`, { method: 'POST', body: JSON.stringify({ expected_version: expectedVersion }) }),
  run: (projectId: string) => request<Task>(`/api/projects/${projectId}/run`, { method: 'POST' }),
  task: (taskId: string) => request<Task>(`/api/tasks/${taskId}`),
  analysis: (projectId: string, kind?: string) => request<AnalysisRecord[]>(`/api/projects/${projectId}/analysis${kind ? `?kind=${encodeURIComponent(kind)}` : ''}`),
  reviews: (projectId: string) => request<ReviewItem[]>(`/api/projects/${projectId}/review-items`),
  resolveReview: (reviewId: string, status: string, note?: string) => request(`/api/review-items/${reviewId}`, { method: 'PATCH', body: JSON.stringify({ status, note }) }),
  reports: (projectId: string) => request<ReportRecord[]>(`/api/projects/${projectId}/reports`),
  finalizeReport: (projectId: string) => request<ReportRecord>(`/api/projects/${projectId}/reports/finalize`, { method: 'POST' })
}
