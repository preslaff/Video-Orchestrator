import { redirect } from 'next/navigation'

/** Projects moved to the main menu; keep old bookmarks working. */
export default function LegacyProjectsRedirect() {
  redirect('/projects')
}
