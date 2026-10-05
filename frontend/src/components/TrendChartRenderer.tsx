import {
  CartesianGrid,
  Line,
  LineChart as RechartsLineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { TimePoint } from './Charts'

export default function TrendChartRenderer({
  data,
  lines,
  forecastStart,
  height = 275,
}: {
  data: TimePoint[]
  lines: Array<{ key: string; label: string; color: string; dashed?: boolean }>
  forecastStart?: string
  height?: number
}) {
  const chartData = data.map((point) => ({
    ...point,
    monthLabel: new Intl.DateTimeFormat('en', { month: 'short', year: '2-digit' }).format(new Date(point.month)),
  }))
  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <RechartsLineChart data={chartData} margin={{ top: 8, right: 14, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#edf1f6" vertical={false} />
          <XAxis dataKey="monthLabel" tickLine={false} axisLine={false} tick={{ fill: '#8a96a7', fontSize: 10 }} minTickGap={20} />
          <YAxis tickLine={false} axisLine={false} tick={{ fill: '#8a96a7', fontSize: 10 }} width={55} tickFormatter={(value: number) => new Intl.NumberFormat('en-IN', { notation: 'compact', maximumFractionDigits: 1 }).format(value)} />
          <Tooltip formatter={(value) => new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 }).format(Number(value))} labelStyle={{ color: '#253951' }} contentStyle={{ border: '1px solid #e5eaf1', borderRadius: 8, fontSize: 11 }} />
          {forecastStart && <ReferenceLine x={new Intl.DateTimeFormat('en', { month: 'short', year: '2-digit' }).format(new Date(forecastStart))} stroke="#b8c4d4" strokeDasharray="4 4" />}
          {lines.map((line) => (
            <Line key={line.key} dataKey={line.key} name={line.label} type="monotone" stroke={line.color} strokeWidth={2.4} dot={false} activeDot={{ r: 4 }} strokeDasharray={line.dashed ? '6 5' : undefined} connectNulls />
          ))}
        </RechartsLineChart>
      </ResponsiveContainer>
    </div>
  )
}
