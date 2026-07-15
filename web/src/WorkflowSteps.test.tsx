import { render, screen } from '@testing-library/react'
import { WorkflowSteps } from './WorkflowSteps'

test('shows the four user-facing workflow steps', () => {
  render(<WorkflowSteps currentStep={2} />)
  expect(screen.getByText('1. Create test data')).toBeInTheDocument()
  expect(screen.getByText('2. Run evaluation')).toBeInTheDocument()
  expect(screen.getByText('3. Generate report')).toBeInTheDocument()
  expect(screen.getByText('4. View on another computer')).toBeInTheDocument()
  expect(screen.getByText('2. Run evaluation').closest('li')).toHaveAttribute('aria-current', 'step')
})
