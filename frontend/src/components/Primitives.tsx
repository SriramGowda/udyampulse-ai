import type { ReactNode } from 'react'
import { AlertCircle, ArrowDownRight, ArrowUpRight, Download, FileText, LoaderCircle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export function PageHeader({
  eyebrow,
  title,
  description,
  action,
}: {
  eyebrow: string
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p className="subtitle">{description}</p>
      </div>
      {action}
    </div>
  )
}

export function Panel({
  title,
  description,
  action,
  children,
}: {
  title: string
  description?: string
  action?: ReactNode
  children: ReactNode
}) {
  return (
    <section className="panel">
      <div className="panel-title">
        <div>
          <h2>{title}</h2>
          {description && <p>{description}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

export function MetricCard({
  label,
  value,
  note,
  icon: Icon,
  tone,
}: {
  label: string
  value: string
  note?: string
  icon: LucideIcon
  tone?: 'positive' | 'negative'
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span>{label}</span>
        <span className="metric-icon"><Icon size={15} /></span>
      </div>
      <div className="metric-value">{value}</div>
      {note && (
        <div className={`metric-foot ${tone ?? ''}`}>
          {tone === 'positive' ? <ArrowUpRight size={13} /> : tone === 'negative' ? <ArrowDownRight size={13} /> : null}
          {note}
        </div>
      )}
    </div>
  )
}

export function LoadingState({ label = 'Loading analytics…' }: { label?: string }) {
  return <div className="state"><span className="spinner" />{label}</div>
}

export function ErrorState({ message }: { message: string }) {
  return <div className="state error"><AlertCircle size={16} />{message}</div>
}

export function EmptyState({ label }: { label: string }) {
  return <div className="state empty"><FileText size={18} />{label}</div>
}

export function StateBlock({
  loading,
  error,
  children,
}: {
  loading: boolean
  error: string | null
  children: ReactNode
}) {
  if (loading) return <LoadingState />
  if (error) return <ErrorState message={error} />
  return <>{children}</>
}

export function StatusTag({ value }: { value: string }) {
  return <span className={`tag ${value.toLowerCase()}`}>{value}</span>
}

export function ProbabilityBars({ values }: { values: Record<string, number> }) {
  return (
    <div>
      {Object.entries(values).map(([label, value]) => (
        <div className="probability-row" key={label}>
          <span>{label}</span>
          <div className="probability-track">
            <div className={`probability-fill ${label.toLowerCase()}`} style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
          </div>
          <strong>{(value * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  )
}

export function DownloadAction({ onClick, label = 'Export CSV' }: { onClick: () => void; label?: string }) {
  return <button className="button secondary" onClick={onClick}><Download size={14} />{label}</button>
}

export function DataTable({
  columns,
  rows,
}: {
  columns: Array<{ key: string; label: string; render?: (row: Record<string, unknown>) => ReactNode }>
  rows: Array<Record<string, unknown>>
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead><tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={String(row.month ?? index)}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row) : String(row[column.key] ?? '—')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function BusyButton({
  busy,
  disabled = false,
  children,
  onClick,
  className = '',
}: {
  busy: boolean
  disabled?: boolean
  children: ReactNode
  onClick: () => void
  className?: string
}) {
  return (
    <button className={`button ${className}`} disabled={busy || disabled} onClick={onClick}>
      {busy ? <LoaderCircle size={14} className="spin-icon" /> : null}
      {children}
    </button>
  )
}
