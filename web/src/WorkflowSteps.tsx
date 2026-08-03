type WorkflowStepsProps = {
  currentStep: 1 | 2 | 3 | 4
}

const steps = [
  '1. Create test data',
  '2. Run evaluation',
  '3. Generate report',
  '4. View on another computer'
] as const

export function WorkflowSteps({ currentStep }: WorkflowStepsProps) {
  return <ol className="workflow-steps" aria-label="Evaluation workflow">
    {steps.map((label, index) => {
      const step = (index + 1) as 1 | 2 | 3 | 4
      return <li
        key={label}
        className={step < currentStep ? 'complete' : step === currentStep ? 'current' : ''}
        aria-current={step === currentStep ? 'step' : undefined}
      >
        <span>{step < currentStep ? '✓' : step}</span>
        <b>{label}</b>
      </li>
    })}
  </ol>
}
