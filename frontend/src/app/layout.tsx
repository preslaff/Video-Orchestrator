import type { Metadata } from 'next'
import './globals.css'
import { Providers } from './providers'
import { Nav } from '@/components/Nav'

export const metadata: Metadata = {
  title: 'vorch — Video Orchestrator',
  description: 'Local AI raw-material generator: storyboard → keyframes → clips',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Providers>
          <Nav />
          <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
        </Providers>
      </body>
    </html>
  )
}
