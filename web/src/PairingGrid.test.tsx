import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { PairingGrid } from './PairingGrid'
import type { Pairing } from './types'

const pairing: Pairing = {project_id:'p',version:1,confirmed:false,devices:[{id:'a',name:'A',folder_path:'/a'},{id:'b',name:'B',folder_path:'/b'}],available_images:{a:[{image_id:'i',filename:'1.jpg',path:'/a/1.jpg',width:1,height:1}],b:[]},groups:[{id:'g',group_id:'G001',analyzable:false,matching_confidence:0.55,review_required:true,match_notes:['Different capture sequence; verify manually'],cells:{a:{image_id:'i',filename:'1.jpg',path:'/a/1.jpg',width:1,height:1},b:null}}]}

describe('PairingGrid', () => {
  afterEach(() => { cleanup(); localStorage.clear(); vi.unstubAllGlobals(); vi.restoreAllMocks() })
  it('renders explicit missing cells and review evidence', () => {
    render(<PairingGrid pairing={pairing} />)
    expect(screen.getByText('MISSING')).toBeInTheDocument()
    expect(screen.getByText('Review required · confidence 55%')).toBeInTheDocument()
    expect(screen.getByText('Different capture sequence; verify manually')).toBeInTheDocument()
    expect(screen.getByRole('img', {name:'A 1.jpg'})).toHaveAttribute('src', '/api/assets/i')
    expect(screen.getByRole('status')).toHaveTextContent('awaiting manual confirmation')
  })
  it('edits selected image and retains independent confirmation visibility', () => {
    const onChange = vi.fn()
    const { rerender } = render(<PairingGrid pairing={pairing} onChange={onChange} />)
    fireEvent.change(screen.getByRole('combobox', {name:'G001-A'}), {target:{value:''}})
    expect(onChange).toHaveBeenCalledWith('g', 'a', null)
    rerender(<PairingGrid pairing={{...pairing, confirmed:true, snapshot_id:'s1'}} />)
    expect(screen.getByRole('status')).toHaveTextContent('Manually confirmed pairing · version 1 · snapshot s1')
    expect(screen.getByText('Review required · confidence 55%')).toBeInTheDocument()
  })
  it('loads thumbnails through authenticated transport when a token is configured', async () => {
    localStorage.setItem('fronecamera-token', 'test-token')
    const fetchMock = vi.fn().mockResolvedValue({ok:true, blob:async () => new Blob(['image'])})
    vi.stubGlobal('fetch', fetchMock)
    const createUrl = vi.fn().mockReturnValue('blob:thumbnail')
    const revokeUrl = vi.fn()
    vi.stubGlobal('URL', {createObjectURL:createUrl, revokeObjectURL:revokeUrl})
    const {unmount} = render(<PairingGrid pairing={pairing} />)
    await waitFor(() => expect(screen.getByRole('img', {name:'A 1.jpg'})).toHaveAttribute('src', 'blob:thumbnail'))
    expect(fetchMock).toHaveBeenCalledWith('/api/assets/i', expect.objectContaining({headers:{Authorization:'Bearer test-token'}}))
    unmount()
    expect(revokeUrl).toHaveBeenCalledWith('blob:thumbnail')
  })

})
