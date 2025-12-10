import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Trail Running Race Predictor',
  description: 'AI-powered race time predictions using Temporal Fusion Transformers',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
