import { NewJobWizard } from '@/components/NewJobWizard'

/** `/jobs/new?project=<id>` (from a project page) preselects the project. */
export default async function NewJobPage({ searchParams }: { searchParams: Promise<{ project?: string }> }) {
  const { project } = await searchParams
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold">New job</h1>
      <NewJobWizard initialProjectId={project} />
    </div>
  )
}
