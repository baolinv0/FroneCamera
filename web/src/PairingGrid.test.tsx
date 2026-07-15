import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PairingGrid } from './PairingGrid'

describe('PairingGrid', () => {
  it('renders explicit missing cells', () => {
    render(<PairingGrid pairing={{project_id:'p',version:1,confirmed:false,devices:[{id:'a',name:'A',folder_path:'/a'},{id:'b',name:'B',folder_path:'/b'}],available_images:{a:[{image_id:'i',filename:'1.jpg',path:'/a/1.jpg',width:1,height:1}],b:[]},groups:[{id:'g',group_id:'G001',analyzable:false,cells:{a:{image_id:'i',filename:'1.jpg',path:'/a/1.jpg',width:1,height:1},b:null}}]}} />)
    expect(screen.getByText('MISSING')).toBeInTheDocument()
  })
})
