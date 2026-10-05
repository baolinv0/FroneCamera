import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import App from './App'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

// Keep the POST pending until React has finished dispatching the submit event.
function network(initialProject = false) {
  const project = { id: 'project-1', name: 'Portrait test', status: 'CREATED', devices: [] as { id: string; name: string }[] }
  let projects = initialProject ? [project] : []
  let finishPost!: () => void
  const postBoundary = new Promise<void>(resolve => { finishPost = resolve })
  const response = (body: unknown) => new Response(JSON.stringify(body), { status: 200 })
  vi.stubGlobal('fetch', vi.fn(async (path: string, init?: RequestInit) => {
    if (init?.method === 'POST') {
      await postBoundary
      const body = JSON.parse(String(init.body))
      if (path === '/api/projects') {
        project.name = body.name
        projects = [project]
        return response(project)
      }
      if (path === '/api/projects/project-1/devices') {
        const device = { id: 'device-1', ...body }
        project.devices.push(device)
        return response(device)
      }
    }
    if (path === '/api/projects') return response(projects)
    if (path === '/api/projects/project-1') return response(project)
    if (path === '/api/projects/project-1/pairing') return response(null)
    if (['pairing/snapshots', 'analysis', 'review-items', 'reports'].some(suffix => path === `/api/projects/project-1/${suffix}`)) return response([])
    throw new Error(`Unexpected request: ${init?.method || 'GET'} ${path}`)
  }))
  return finishPost
}

test('creating a project asynchronously lists and opens it, and resets the form without an error', async () => {
  const finishPost = network()
  const { container } = render(<App />)
  const name = screen.getByPlaceholderText('Test name')
  fireEvent.change(name, { target: { value: 'New portrait test' } })
  fireEvent.submit(name.closest('form')!)
  expect(name).toHaveValue('New portrait test')
  expect(screen.queryByRole('heading', { name: 'New portrait test' })).not.toBeInTheDocument()

  await act(async () => { finishPost() })

  expect(await screen.findByRole('heading', { name: 'New portrait test' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /New portrait test/ })).toBeInTheDocument()
  expect(screen.getByPlaceholderText('Device label')).toBeInTheDocument()
  expect(name).toHaveValue('')
  expect(container.querySelector('.error')).not.toBeInTheDocument()
})

test('registering a device asynchronously refreshes the device summary and resets the form without an error', async () => {
  const finishPost = network(true)
  const { container } = render(<App />)
  fireEvent.click(await screen.findByRole('button', { name: /Portrait test/ }))
  const name = await screen.findByPlaceholderText('Device label')
  const folder = screen.getByPlaceholderText('Linux folder path')
  const model = screen.getByPlaceholderText('Canonical model')
  fireEvent.change(name, { target: { value: 'Phone A' } })
  fireEvent.change(folder, { target: { value: '/photos/phone-a' } })
  fireEvent.change(model, { target: { value: 'Model A' } })
  fireEvent.submit(name.closest('form')!)
  expect(screen.queryByText('Phone A')).not.toBeInTheDocument()

  await act(async () => { finishPost() })

  await waitFor(() => expect(container.querySelector('.device-summary')).toHaveTextContent('Phone A'))
  expect(screen.getByText(/Device registered/)).toBeInTheDocument()
  expect(name).toHaveValue('')
  expect(folder).toHaveValue('')
  expect(model).toHaveValue('')
  expect(container.querySelector('.error')).not.toBeInTheDocument()
})
