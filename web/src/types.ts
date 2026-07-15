export type Project = { id: string; name: string; status: string; version: number; devices?: Device[]; group_count?: number; review_item_count?: number; reports?: ReportRecord[] }
export type Device = { id: string; name: string; folder_path: string; canonical_model?: string | null }
export type PairingCell = { image_id: string; filename: string; path: string; width: number; height: number } | null
export type PairingGroup = { id: string; group_id: string; label?: string | null; cells: Record<string, PairingCell>; analyzable: boolean }
export type Pairing = { project_id: string; version: number; confirmed: boolean; devices: Device[]; groups: PairingGroup[]; available_images: Record<string, Exclude<PairingCell, null>[]> }
export type ReviewItem = { id: string; category: string; priority: string; status: string; payload: unknown }
export type Task = { id: string; status: string; attempts?: number; result?: Record<string, unknown> | null; error?: string | null }

export type AnalysisRecord = { id: string; kind: string; scene_group_id: string | null; image_id: string | null; payload: unknown }
export type ReportRecord = { id: string; version: string; status: string; html_path: string; created_at: string }
export type PairingSnapshot = { id: string; version: number; payload: Pairing; created_at: string }
