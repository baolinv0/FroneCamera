import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { JSDOM } from 'jsdom'
import { afterEach, describe, expect, it, vi } from 'vitest'

const source = process.env.FRONE_STATIC_CONSOLE_TEST_SOURCE || resolve(process.cwd(), '../src/portrait_eval/static/index.html')
const fallbackHtml = readFileSync(source, 'utf8')
let dom

function prepareConsole(projectName, statement) {
  const fetchMock = vi.fn(async path => ({
    ok: true,
    headers: new Headers({'content-type':'application/json'}),
    json: async () => path === '/api/projects'
      ? [{id:'p1', name:projectName, status:"<img id='project-status-injection' onerror='window.__projectInjected=true'>"}]
      : path.endsWith('/review-items')
        ? [{category:"<img id='review-category-injection' onerror='window.__reviewInjected=true'>", status:'open', payload:{primary_observation:{statement}}}]
        : {project_id:'p1', version:1, confirmed:false, devices:[], groups:[], available_images:{}}
  }))
  dom = new JSDOM(fallbackHtml, {runScripts:'dangerously', url:'http://localhost/', beforeParse(window) {
    window.fetch = fetchMock
    window.__reviewInjected = false
    window.__projectInjected = false
  }})
  return {page:dom.window, fetchMock}
}

describe('actual static review console', () => {
  afterEach(() => { dom?.window.close(); vi.restoreAllMocks() })
  it('renders provider and project markup as plain text without executing handlers', async () => {
    const name = "<img id='project-injection' src='bad' onerror='window.__projectInjected=true'>"
    const statement = "<img id='review-injection' src='bad' onerror='window.__reviewInjected=true'>"
    const {page} = prepareConsole(name, statement)
    await page.loadProjects()
    await page.openProject('p1',name)
    page.document.querySelectorAll('[id$="injection"]').forEach(node => node.dispatchEvent(new page.Event('error')))
    expect(page.__reviewInjected).toBe(false)
    expect(page.__projectInjected).toBe(false)
    expect(page.document.querySelector('[id$="injection"]')).toBeNull()
    expect(page.document.getElementById('projects').textContent).toContain(name)
    expect(page.document.getElementById('reviews').textContent).toContain(statement)
    expect(page.document.getElementById('reviews').textContent).toContain('review-category-injection')
    expect(page.document.getElementById('workspaceTitle').textContent).toBe(name)
  })
  it('opens a quoted project name through an event listener while retaining its exact label', async () => {
    const name = "Phone's camera \"quoted\" <b>label</b>"
    const {page,fetchMock} = prepareConsole(name,'ordinary observation')
    await page.loadProjects()
    const button = page.document.querySelector('#projects button')
    expect(button.getAttribute('onclick')).toBeNull()
    button.click()
    await vi.waitFor(() => expect(page.document.getElementById('reviews').textContent).toContain('ordinary observation'))
    expect(page.document.getElementById('workspaceTitle').textContent).toBe(name)
    expect(fetchMock).toHaveBeenCalledWith('/api/projects/p1/pairing', expect.any(Object))
  })
})
