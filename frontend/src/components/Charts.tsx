import { lazy, Suspense } from 'react'

export interface TimePoint {
  month: string
  [key: string]: string | number | null
}

const TrendChartRenderer = lazy(() => import('./TrendChartRenderer'))

export function TrendChart(props: {
  data: TimePoint[]
  lines: Array<{ key: string; label: string; color: string; dashed?: boolean }>
  forecastStart?: string
  height?: number
}) {
  return (
    <Suspense fallback={<div className="state">Preparing chart…</div>}>
      <TrendChartRenderer {...props} />
    </Suspense>
  )
}
