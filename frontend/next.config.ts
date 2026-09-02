import type { NextConfig } from 'next'

// The FastAPI backend (vorch serve). The browser only ever talks to this Next server;
// /api and /media are proxied so everything stays same-origin (no CORS, relative URLs).
const API_ORIGIN = process.env.API_ORIGIN ?? 'http://127.0.0.1:8000'

const nextConfig: NextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: '/api/:path*', destination: `${API_ORIGIN}/api/:path*` },
      { source: '/media/:path*', destination: `${API_ORIGIN}/media/:path*` },
      { source: '/docs', destination: `${API_ORIGIN}/docs` },
      { source: '/openapi.json', destination: `${API_ORIGIN}/openapi.json` },
    ]
  },
  async redirects() {
    return [{ source: '/', destination: '/jobs', permanent: false }]
  },
}

export default nextConfig
