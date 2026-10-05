import { useEffect, useState } from 'react'
import type { Pairing } from './types'

function PairingThumbnail({ imageId, label }: { imageId: string; label: string }) {
  const assetUrl = `/api/assets/${encodeURIComponent(imageId)}`
  const token = localStorage.getItem('fronecamera-token') || ''
  const [source, setSource] = useState<string | undefined>(token ? undefined : assetUrl)
  useEffect(() => {
    if (!token) { setSource(assetUrl); return }
    const controller = new AbortController()
    let objectUrl: string | undefined
    setSource(undefined)
    fetch(assetUrl, { headers: { Authorization: `Bearer ${token}` }, signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error('Thumbnail unavailable'); return response.blob() })
      .then(blob => { if (!controller.signal.aborted) { objectUrl = URL.createObjectURL(blob); setSource(objectUrl) } })
      .catch(() => { if (!controller.signal.aborted) setSource(undefined) })
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [assetUrl, token])
  return <img src={source} alt={label} loading="lazy" width={120} height={90} style={{ objectFit: 'contain' }} />
}

export function PairingGrid({ pairing, onChange }: { pairing: Pairing; onChange?: (groupId: string, deviceId: string, imageId: string | null) => void }) {
  return <div className="scroll">
    <p role="status">{pairing.confirmed ? 'Manually confirmed pairing' : 'Pairing awaiting manual confirmation'} · version {pairing.version}{pairing.snapshot_id && ` · snapshot ${pairing.snapshot_id}`}</p>
    <table><thead><tr><th>Group</th>{pairing.devices.map(device => <th key={device.id}>{device.name}</th>)}</tr></thead><tbody>{pairing.groups.map(group => <tr key={group.id}>
      <td><b>{group.group_id}</b><small>{group.analyzable ? ' analyzable' : ' insufficient'}</small><p>{group.review_required ? 'Review required' : 'Automatic match'} · confidence {group.matching_confidence == null ? 'unknown' : `${Math.round(group.matching_confidence * 100)}%`}</p>{group.match_notes?.map((note, index) => <p key={index}>{note}</p>)}</td>
      {pairing.devices.map(device => { const cell = group.cells[device.id]; return <td key={device.id}>
        {cell && <PairingThumbnail imageId={cell.image_id} label={`${device.name} ${cell.filename}`} />}
        {onChange ? <select aria-label={`${group.group_id}-${device.name}`} value={cell?.image_id ?? ''} onChange={event => onChange(group.id, device.id, event.target.value || null)}><option value="">MISSING</option>{(pairing.available_images[device.id] ?? []).map(image => <option key={image.image_id} value={image.image_id}>{image.filename}</option>)}</select> : (cell?.filename ?? <span className="missing">MISSING</span>)}
      </td>})}
    </tr>)}</tbody></table>
  </div>
}
