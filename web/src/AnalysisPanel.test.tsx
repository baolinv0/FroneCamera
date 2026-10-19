import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { AnalysisPanel } from './AnalysisPanel'

const records = [
  { id: 'analysis-1', kind: 'image_metrics', scene_group_id: 'group-1', image_id: 'image-1', payload: { whole: { luma_mean: 0.42, highlight_clip_ratio: 0.01 }, diagnostic_path: '/workspace/diagnostic.jpg' } },
  { id: 'analysis-2', kind: 'strategy_claim', scene_group_id: null, image_id: null, payload: { statement: 'Higher global luminance tendency', grade: 'B' } }
]

describe('AnalysisPanel', () => {
  it('groups structured results and links diagnostic assets', () => {
    render(<AnalysisPanel records={records} />)
    expect(screen.getByText('image_metrics')).toBeInTheDocument()
    expect(screen.getByText('strategy_claim')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open diagnostic' })).toHaveAttribute('href', '/api/analysis/analysis-1/diagnostic')
  })
})
