'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState } from 'react'
import { LightboxProvider } from '@/components/Lightbox'

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { staleTime: 5_000, refetchOnWindowFocus: false, retry: 1 },
        },
      }),
  )
  return <QueryClientProvider client={client}><LightboxProvider>{children}</LightboxProvider></QueryClientProvider>
}
