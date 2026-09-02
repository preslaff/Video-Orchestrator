import { NewJobWizard } from '@/components/NewJobWizard'

export default function NewJobPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">New job</h1>
      <NewJobWizard />
    </div>
  )
}
