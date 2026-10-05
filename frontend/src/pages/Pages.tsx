import { useState, type FormEvent } from 'react'
import { useOutletContext } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  Banknote,
  BriefcaseBusiness,
  CalendarDays,
  CircleDollarSign,
  Download,
  FileSpreadsheet,
  Landmark,
  RefreshCw,
  Send,
  Sparkles,
  Wallet,
} from 'lucide-react'
import { TrendChart, type TimePoint } from '../components/Charts'
import { AiModeIndicator } from '../components/AiModeIndicator'
import {
  BusyButton,
  DataTable,
  DownloadAction,
  EmptyState,
  ErrorState,
  LoadingState,
  MetricCard,
  PageHeader,
  Panel,
  ProbabilityBars,
  StateBlock,
  StatusTag,
} from '../components/Primitives'
import { formatMoney, formatNumber, formatPercent } from '../utils/format'
import { useApi } from '../hooks/useApi'
import { downloadCsv, request } from '../services/api'
import type { AnomalyResult } from '../types/api'
import type {
  AIStatus,
  BusinessRecord,
  ComplianceResult,
  ForecastResult,
  HealthResult,
  RiskResult,
  ShapFeature,
  SummaryResult,
} from '../types/api'
import type { BusinessOutletContext } from '../layouts/AppLayout'

function useBusinessId() {
  return useOutletContext<BusinessOutletContext>().businessId
}

const businessPath = (id: string, suffix = '') =>
  id ? `/api/business/${encodeURIComponent(id)}${suffix}` : null

function useBusinessKpis(id: string) {
  return useApi<BusinessRecord>(businessPath(id, '/kpis'))
}

function useBusinessHistory(id: string) {
  return useApi<{ business: BusinessRecord; history: BusinessRecord[] }>(businessPath(id))
}

function moneyMetric(label: string, value: number, icon: typeof Banknote, note?: string) {
  return <MetricCard key={label} label={label} value={formatMoney(value)} note={note} icon={icon} />
}

function trendData(history: BusinessRecord[], metric: string, secondary?: string): TimePoint[] {
  return history.map((point) => ({
    month: point.month,
    [metric]: Number(point[metric as keyof BusinessRecord] ?? 0),
    ...(secondary
      ? { [secondary]: Number(point[secondary as keyof BusinessRecord] ?? 0) }
      : {}),
  }))
}

function DashboardAiCard({ businessId }: { businessId: string }) {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SummaryResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function generate() {
    setLoading(true)
    setError(null)
    try {
      setResult(await request<SummaryResult>('/api/ai/summary', {
        method: 'POST',
        body: JSON.stringify({ business_id: businessId }),
      }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Insight generation failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Panel
      title="AI financial insight"
      description="Grounded in this business's selected synthetic metrics"
      action={<AiModeIndicator provider={result?.provider} />}
    >
      {error && <ErrorState message={error} />}
      {result ? (
        <>
          <p className="small text-muted">{result.answer}</p>
          <div className="metric-foot">{result.used_gemini ? 'Generated with Gemini' : result.provider === 'local' ? 'Generated locally with Ollama' : 'Local analytical fallback · no LLM used'}</div>
        </>
      ) : (
        <>
          <p className="small text-muted">Generate a concise explanation of revenue, liquidity, model risk and the six-month forecast.</p>
          <button className="button purple" disabled={loading || !businessId} onClick={generate}>
            {loading ? <span className="spinner" /> : <Sparkles size={14} />}
            Generate insight
          </button>
        </>
      )}
    </Panel>
  )
}

export function DashboardPage() {
  const id = useBusinessId()
  const kpis = useBusinessKpis(id)
  const history = useBusinessHistory(id)
  const risk = useApi<RiskResult>(businessPath(id, '/risk'))
  const anomaly = useApi<AnomalyResult>(businessPath(id, '/anomaly'))
  const forecast = useApi<ForecastResult>(businessPath(id, '/forecast'))

  return (
    <>
      <PageHeader eyebrow="Overview" title="Good morning" description="A clear view of business health, risk signals and what's ahead." />
      <StateBlock loading={kpis.loading} error={kpis.error}>
        {kpis.data && (
          <div className="grid kpi-grid" style={{ marginBottom: 17 }}>
            {moneyMetric('Revenue', kpis.data.revenue, CircleDollarSign, `${formatPercent(kpis.data.revenue_growth)} vs prior month`)}
            {moneyMetric('Operating expenses', kpis.data.expenses, FileSpreadsheet, `${formatPercent(kpis.data.expense_growth)} vs prior month`)}
            {moneyMetric('Profit', kpis.data.profit, Activity, `${formatNumber(kpis.data.profit_margin)}% margin`)}
            {moneyMetric('Cash balance', kpis.data.cash_balance, Wallet, `${formatNumber(kpis.data.cash_ratio)}× cash ratio`)}
          </div>
        )}
      </StateBlock>

      <div className="grid two-col" style={{ marginBottom: 16 }}>
        <Panel title="Revenue & expense trend" description="Monthly performance across the available business history">
          <StateBlock loading={history.loading} error={history.error}>
            {history.data?.history.length ? (
              <TrendChart data={trendData(history.data.history, 'revenue', 'expenses')} lines={[
                { key: 'revenue', label: 'Revenue', color: '#426bc5' },
                { key: 'expenses', label: 'Expenses', color: '#d5a34d' },
              ]} />
            ) : <EmptyState label="No business history is available." />}
          </StateBlock>
        </Panel>
        <Panel title="Risk status" description="Current model classification">
          <StateBlock loading={risk.loading} error={risk.error}>
            {risk.data && (
              <>
                <StatusTag value={risk.data.label} />
                <p className="small text-muted" style={{ marginTop: 13 }}>{risk.data.explanation}</p>
                <ProbabilityBars values={risk.data.probabilities} />
                <a className="button secondary" href="/risk-intelligence">Explore risk factors <ArrowRight size={13} /></a>
              </>
            )}
          </StateBlock>
        </Panel>
      </div>

      <div className="grid two-col" style={{ marginBottom: 16 }}>
        <Panel title="Six-month outlook" description="Damped-trend projections from historical records">
          <StateBlock loading={forecast.loading} error={forecast.error}>
            {forecast.data && (
              <>
                <div className="grid three-col" style={{ marginBottom: 17 }}>
                  <div><div className="eyebrow">Revenue MAE</div><strong>{formatMoney(forecast.data.forecasts.revenue.evaluation.mae)}</strong></div>
                  <div><div className="eyebrow">Revenue RMSE</div><strong>{formatMoney(forecast.data.forecasts.revenue.evaluation.rmse)}</strong></div>
                  <div><div className="eyebrow">Cash MAE</div><strong>{formatMoney(forecast.data.forecasts.cash_balance.evaluation.mae)}</strong></div>
                </div>
                <TrendChart
                  data={[
                    ...forecast.data.forecasts.revenue.history.slice(-12).map((point) => ({ month: point.month, actual: Number(point.revenue) })),
                    ...forecast.data.forecasts.revenue.forecast.map((point) => ({ month: point.month, forecast: Number(point.revenue) })),
                  ]}
                  lines={[
                    { key: 'actual', label: 'Historical revenue', color: '#426bc5' },
                    { key: 'forecast', label: 'Forecast revenue', color: '#7f70c9', dashed: true },
                  ]}
                />
              </>
            )}
          </StateBlock>
        </Panel>
        <div className="stack">
          <Panel title="Latest anomaly signal" description="Isolation Forest result for the latest month">
            <StateBlock loading={anomaly.loading} error={anomaly.error}>
              {anomaly.data && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                  <div><StatusTag value={anomaly.data.status} /><p className="small text-muted" style={{ margin: '11px 0 0' }}>{anomaly.data.explanation}</p></div>
                  <div style={{ textAlign: 'right' }}><div className="eyebrow">Score</div><strong>{anomaly.data.score.toFixed(4)}</strong></div>
                </div>
              )}
            </StateBlock>
          </Panel>
          <DashboardAiCard businessId={id} />
        </div>
      </div>
    </>
  )
}

export function BusinessAnalysisPage() {
  const id = useBusinessId()
  const kpis = useBusinessKpis(id)
  const history = useBusinessHistory(id)
  const record = kpis.data

  return (
    <>
      <PageHeader eyebrow="Business performance" title="Business analysis" description="Financial position, operating trends and working-capital indicators." />
      <StateBlock loading={kpis.loading} error={kpis.error}>
        {record && (
          <>
            <div className="grid kpi-grid" style={{ marginBottom: 16 }}>
              {moneyMetric('Revenue', record.revenue, CircleDollarSign)}
              {moneyMetric('Expenses', record.expenses, FileSpreadsheet)}
              {moneyMetric('Profit', record.profit, Activity)}
              {moneyMetric('Cash balance', record.cash_balance, Wallet)}
              {moneyMetric('Receivables', record.accounts_receivable, BriefcaseBusiness)}
              {moneyMetric('Payables', record.accounts_payable, FileSpreadsheet)}
              {moneyMetric('Loan outstanding', record.loan_outstanding, Landmark)}
              {moneyMetric('Working capital', record.working_capital, Banknote)}
            </div>
            <div className="grid two-col">
              <Panel title="Operating trend" description="Revenue, expenses and profit by month">
                <StateBlock loading={history.loading} error={history.error}>
                  {history.data && <TrendChart data={trendData(history.data.history, 'revenue', 'expenses')} lines={[
                    { key: 'revenue', label: 'Revenue', color: '#426bc5' },
                    { key: 'expenses', label: 'Expenses', color: '#d5a34d' },
                  ]} />}
                </StateBlock>
              </Panel>
              <Panel title="Financial ratios" description="Latest period values">
                <DataTable columns={[
                  { key: 'label', label: 'Indicator' },
                  { key: 'value', label: 'Latest value' },
                ]} rows={[
                  { label: 'Profit margin', value: `${formatNumber(record.profit_margin)}%` },
                  { label: 'Revenue growth', value: formatPercent(record.revenue_growth) },
                  { label: 'Expense growth', value: formatPercent(record.expense_growth) },
                  { label: 'Cash ratio', value: `${formatNumber(record.cash_ratio)}×` },
                  { label: 'Receivable ratio', value: `${formatNumber(record.receivable_ratio * 100)}%` },
                  { label: 'Debt service ratio', value: `${formatNumber(record.debt_service_ratio * 100)}%` },
                  { label: 'Payable ratio', value: `${formatNumber(record.payable_ratio * 100)}%` },
                  { label: 'Inventory ratio', value: `${formatNumber(record.inventory_ratio * 100)}%` },
                ]} />
              </Panel>
            </div>
          </>
        )}
      </StateBlock>
    </>
  )
}

export function RiskPage() {
  const id = useBusinessId()
  const risk = useApi<RiskResult>(businessPath(id, '/risk'))
  const shap = useApi<{ business_id: string; features: ShapFeature[] }>(businessPath(id, '/shap'))
  const maxImpact = Math.max(0, ...(shap.data?.features ?? []).map((item) => item.absolute_impact))

  return (
    <>
      <PageHeader eyebrow="Model intelligence" title="Risk intelligence" description="Risk classification, class probabilities and feature-level SHAP contributions." />
      <div className="grid two-col">
        <Panel title="Risk classification" description="Random Forest output for the latest business record">
          <StateBlock loading={risk.loading} error={risk.error}>
            {risk.data && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 13 }}><StatusTag value={risk.data.label} /><span className="small text-muted">Latest model assessment</span></div>
                <p className="small text-muted" style={{ marginTop: 16 }}>{risk.data.explanation}</p>
                <ProbabilityBars values={risk.data.probabilities} />
              </>
            )}
          </StateBlock>
        </Panel>
        <Panel title="Model context" description="How to interpret this indicator">
          <div className="notice warning">This risk prediction is a synthetic-data indicator—not a credit decision, guarantee of default, or substitute for professional review.</div>
          <div className="stack" style={{ marginTop: 16 }}>
            <div><div className="eyebrow">Model</div><strong className="small">Existing Random Forest classifier</strong></div>
            <div><div className="eyebrow">Explanation method</div><strong className="small">Tree SHAP feature contributions</strong></div>
            <div><div className="eyebrow">Features</div><strong className="small">Current financial metrics and period-over-period growth</strong></div>
          </div>
        </Panel>
      </div>
      <div style={{ marginTop: 16 }}>
        <Panel title="Top risk factors" description="Feature contribution magnitude; positive and negative values indicate contribution direction">
          <StateBlock loading={shap.loading} error={shap.error}>
            {shap.data?.features.length ? (
              <div className="stack">
                {shap.data.features.map((feature) => (
                  <div key={feature.feature} style={{ display: 'grid', gridTemplateColumns: '155px 1fr 84px', gap: 14, alignItems: 'center' }}>
                    <span className="small">{feature.feature.replaceAll('_', ' ')}</span>
                    <div className="probability-track" style={{ height: 9 }}>
                      <div className="probability-fill" style={{ width: `${maxImpact ? feature.absolute_impact / maxImpact * 100 : 0}%`, background: feature.value >= 0 ? '#617bc1' : '#58a48a' }} />
                    </div>
                    <strong className="small" style={{ textAlign: 'right' }}>{feature.value.toFixed(4)}</strong>
                  </div>
                ))}
              </div>
            ) : <EmptyState label="SHAP factors are not available." />}
          </StateBlock>
        </Panel>
      </div>
    </>
  )
}

export function ForecastingPage() {
  const id = useBusinessId()
  const forecast = useApi<ForecastResult>(businessPath(id, '/forecast'))
  const [metric, setMetric] = useState<'revenue' | 'cash_balance'>('revenue')
  const item = forecast.data?.forecasts[metric]
  const title = metric === 'revenue' ? 'Revenue forecast' : 'Cash balance forecast'
  const chartData: TimePoint[] = item
    ? [
        ...item.history.map((point) => ({ month: point.month, historical: Number(point[metric]) })),
        ...item.forecast.map((point) => ({ month: point.month, forecasted: Number(point[metric]) })),
      ]
    : []

  const csvRows = item?.forecast.map((point) => ({
    month: point.month,
    metric,
    forecast_value: point[metric],
  })) ?? []

  return (
    <>
      <PageHeader eyebrow="Planning" title="Forecasting" description="Six-month revenue and cash projections, with chronological holdout error metrics." action={
        <DownloadAction onClick={() => downloadCsv(`${id}_${metric}_forecast.csv`, csvRows)} label="Download forecast" />
      } />
      <div className="panel" style={{ marginBottom: 16, padding: 7, display: 'inline-flex' }}>
        <div className="tabs">
          <button className={`tab${metric === 'revenue' ? ' active' : ''}`} onClick={() => setMetric('revenue')}>Revenue forecast</button>
          <button className={`tab${metric === 'cash_balance' ? ' active' : ''}`} onClick={() => setMetric('cash_balance')}>Cash forecast</button>
        </div>
      </div>
      <Panel title={title} description="Historical observations with the model's six future monthly values">
        <StateBlock loading={forecast.loading} error={forecast.error}>
          {item && (
            <>
              <div className="grid three-col" style={{ marginBottom: 16 }}>
                <MetricCard label="MAE" value={formatMoney(item.evaluation.mae)} note={`${item.evaluation.holdout_months} holdout months`} icon={Activity} />
                <MetricCard label="RMSE" value={formatMoney(item.evaluation.rmse)} note="Chronological evaluation" icon={Activity} />
                <MetricCard label="Latest projection" value={formatMoney(Number(item.forecast.at(-1)?.[metric]))} note="Six months ahead" icon={CalendarDays} />
              </div>
              <TrendChart data={chartData} forecastStart={item.history.at(-1)?.month} lines={[
                { key: 'historical', label: 'Historical', color: '#426bc5' },
                { key: 'forecasted', label: 'Forecast', color: '#7b6dc6', dashed: true },
              ]} height={330} />
              <div style={{ marginTop: 23 }}>
                <h2>Forecast table</h2>
                <DataTable columns={[
                  { key: 'month', label: 'Month', render: (row) => new Date(String(row.month)).toLocaleDateString('en-IN', { month: 'long', year: 'numeric' }) },
                  { key: 'forecast_value', label: 'Forecast value', render: (row) => formatMoney(Number(row[metric])) },
                  { key: 'metric', label: 'Metric', render: () => metric === 'revenue' ? 'Revenue' : 'Cash balance' },
                ]} rows={item.forecast} />
              </div>
            </>
          )}
        </StateBlock>
      </Panel>
      <p className="footer-note">The existing service fits a damped linear trend using up to 24 chronological monthly values; forecasts are synthetic-data estimates.</p>
    </>
  )
}

export function AnomalyPage() {
  const id = useBusinessId()
  const anomaly = useApi<AnomalyResult>(businessPath(id, '/anomaly'))
  const kpis = useBusinessKpis(id)
  const indicators = [
    ['Revenue', 'revenue'],
    ['Expenses', 'expenses'],
    ['Cash balance', 'cash_balance'],
    ['Receivables', 'accounts_receivable'],
    ['Payables', 'accounts_payable'],
    ['Loan outstanding', 'loan_outstanding'],
  ]

  return (
    <>
      <PageHeader eyebrow="Pattern detection" title="Anomaly detection" description="Review the Isolation Forest's latest pattern signal and underlying financial context." />
      <StateBlock loading={anomaly.loading} error={anomaly.error}>
        {anomaly.data && (
          <>
            <div className="grid two-col" style={{ marginBottom: 16 }}>
              <Panel title="Latest detection" description={`Model evaluation for ${new Date(anomaly.data.month).toLocaleDateString('en-IN', { month: 'long', year: 'numeric' })}`}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <div className={`metric-icon ${anomaly.data.status === 'ANOMALY' ? 'negative' : 'positive'}`} style={{ width: 48, height: 48 }}><Activity size={22} /></div>
                  <div><StatusTag value={anomaly.data.status} /><div className="metric-value" style={{ marginTop: 7 }}>{anomaly.data.score.toFixed(4)}</div><div className="metric-foot">Anomaly score</div></div>
                </div>
                <p className="small text-muted" style={{ marginTop: 18 }}>{anomaly.data.explanation}</p>
              </Panel>
              <Panel title="Interpretation" description="A model signal is not a diagnosis">
                <div className="notice">An anomaly is a financial pattern that differs from patterns learned by the existing Isolation Forest. It may warrant review, but does not by itself indicate financial distress.</div>
                <div className="metric-foot" style={{ marginTop: 14 }}>Model output: {anomaly.data.prediction === -1 ? '−1 · flagged' : '1 · not flagged'}</div>
              </Panel>
            </div>
            <Panel title="Financial indicators" description="Latest observed business values supplied alongside the model score">
              <StateBlock loading={kpis.loading} error={kpis.error}>
                {kpis.data && <div className="grid three-col">
                  {indicators.map(([label, key]) => (
                    <MetricCard key={key} label={label} value={formatMoney(Number(kpis.data?.[key as keyof BusinessRecord]))} icon={Banknote} />
                  ))}
                </div>}
              </StateBlock>
            </Panel>
          </>
        )}
      </StateBlock>
    </>
  )
}

export function AInsightsPage() {
  const id = useBusinessId()
  const health = useApi<HealthResult>('/api/health')
  const aiStatus = useApi<AIStatus>('/api/ai/status')
  const [result, setResult] = useState<SummaryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function generate() {
    setLoading(true)
    setError(null)
    try {
      setResult(await request<SummaryResult>('/api/ai/summary', {
        method: 'POST',
        body: JSON.stringify({ business_id: id }),
      }))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'AI summary request failed.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <PageHeader eyebrow="Generative AI" title="AI insights" description="On-demand financial explanations routed through the secure FastAPI backend." />
      <div className="grid two-col">
        <Panel title="Financial explanation" description="Grounded in selected business metrics, model outputs and forecast values" action={<AiModeIndicator provider={result?.provider} />}>
          {result ? <div className="chat-bubble" style={{ maxWidth: '100%' }}>{result.answer}</div> : <div className="notice">Generate a concise natural-language summary. Gemini is optional; Ollama can generate locally when installed, with a local analytical fallback otherwise.</div>}
          {error && <ErrorState message={error} />}
          {result && <div className="metric-foot" style={{ marginTop: 13 }}>{result.used_gemini ? 'Generated by Gemini' : result.provider === 'local' ? 'Generated locally with Ollama' : 'Local analytical summary · no LLM used'}</div>}
          <div className="button-row" style={{ marginTop: 17 }}>
            <BusyButton busy={loading} disabled={!id} onClick={generate} className="purple"><Sparkles size={14} />{result ? 'Regenerate insight' : 'Generate insight'}</BusyButton>
            {result && <button className="button secondary" onClick={() => setResult(null)}><RefreshCw size={13} />Clear</button>}
          </div>
        </Panel>
        <Panel title="Connection status" description="Backend configuration only">
          <StateBlock loading={health.loading} error={health.error}>
            {health.data && (
              <>
                <div className="probability-row"><span>API</span><span className="tag low">{health.data.status}</span><span /></div>
                <div className="probability-row"><span>Dataset</span><span className="tag low">{health.data.data}</span><span /></div>
                <div className="probability-row"><span>Gemini</span><span className={`tag ${health.data.gemini_configured ? 'low' : 'medium'}`}>{health.data.gemini_configured ? 'Configured' : 'Optional · not configured'}</span><span /></div>
              </>
            )}
          </StateBlock>
          <StateBlock loading={aiStatus.loading} error={aiStatus.error}>
            {aiStatus.data && (
              <div className="stack">
                <div className="probability-row"><span>Local model</span><span className={`tag ${aiStatus.data.local_llm_available ? 'low' : 'medium'}`}>{aiStatus.data.local_llm_available ? `${aiStatus.data.model} ready` : 'Not available'}</span><span /></div>
                <div className="probability-row"><span>RAG</span><span className={`tag ${aiStatus.data.retrieval_available ? 'low' : 'medium'}`}>{aiStatus.data.retrieval_available ? 'Local references ready' : 'No local references'}</span><span /></div>
                <div className="notice">Provider status is based on local configuration and Ollama availability; no Gemini connection check is made. Credentials are never returned by the API.</div>
              </div>
            )}
          </StateBlock>
        </Panel>
      </div>
    </>
  )
}

interface ChatEntry {
  question: string
  result?: ComplianceResult
}

export function CompliancePage() {
  const [question, setQuestion] = useState('')
  const [entries, setEntries] = useState<ChatEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const value = question.trim()
    if (!value || loading) return
    setQuestion('')
    setError(null)
    setLoading(true)
    setEntries((current) => [...current, { question: value }])
    try {
      const result = await request<ComplianceResult>('/api/compliance/query', {
        method: 'POST',
        body: JSON.stringify({ question: value }),
      })
      setEntries((current) => current.map((entry, index) => (
        index === current.length - 1 ? { ...entry, result } : entry
      )))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Compliance query failed.')
      setEntries((current) => current.slice(0, -1))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <PageHeader eyebrow="Grounded document assistant" title="Compliance assistant" description="Ask about working capital and financial practices in the local reference library." action={<AiModeIndicator />} />
      <div className="grid two-col">
        <Panel title="Reference chat" description="Answers use retrieved TXT/MD passages and cite their source filenames">
          <div className="chat-log">
            {entries.length === 0 && (
              <div className="notice">Try asking: “How can an MSME monitor cash flow and working capital?” Responses are informational and do not replace official compliance advice.</div>
            )}
            {entries.map((entry, index) => (
              <div className="stack" key={`${entry.question}-${index}`}>
                <div className="chat-bubble user">{entry.question}</div>
                {entry.result ? (
                  <div className="chat-bubble">
                    <strong>Answer</strong>
                    <div>{entry.result.answer}</div>
                    {entry.result.retrieved_documents.length > 0 && (
                      <div className="retrieved-context">
                        <strong>Retrieved document information</strong>
                        {entry.result.retrieved_documents.map((document, documentIndex) => (
                          <p key={`${document.source}-${document.chunk}-${documentIndex}`}>
                            <span className="source-chip"><FileSpreadsheet size={11} />{document.source}</span>
                            {' '}{document.snippet}
                          </p>
                        ))}
                      </div>
                    )}
                    <strong className="compliance-label">Sources</strong>
                    {entry.result.sources.length > 0 ? <div className="source-list">{entry.result.sources.map((source) => <span className="source-chip" key={source}><FileSpreadsheet size={11} />{source}</span>)}</div> : <div className="small text-muted">No relevant local source found.</div>}
                    <strong className="compliance-label">AI Provider</strong>
                    <AiModeIndicator provider={entry.result.provider} />
                  </div>
                ) : loading && index === entries.length - 1 ? <LoadingState label="Searching references and preparing an answer…" /> : null}
              </div>
            ))}
          </div>
          {error && <ErrorState message={error} />}
          <form className="chat-form" onSubmit={submit}>
            <textarea className="textarea" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about cash flow, receivables, or working capital…" maxLength={2000} />
            <BusyButton busy={loading} onClick={() => undefined}><Send size={14} />Send</BusyButton>
          </form>
        </Panel>
        <Panel title="How answers are grounded" description="Local RAG workflow">
          <div className="stack">
            <div><div className="eyebrow">01 · Retrieve</div><p className="small text-muted">The question is ranked against local document chunks using TF-IDF and cosine similarity.</p></div>
            <div><div className="eyebrow">02 · Generate</div><p className="small text-muted">Gemini is tried first; if unavailable, an installed Ollama model generates locally using only the retrieved references.</p></div>
            <div><div className="eyebrow">03 · Cite</div><p className="small text-muted">Generated explanations are kept distinct from retrieved excerpts and returned with their source filenames. Retrieval-only answers remain available without either LLM.</p></div>
          </div>
          <div className="notice warning" style={{ marginTop: 18 }}>This demonstration does not access private GST, banking or government databases. Always confirm regulatory requirements against official sources.</div>
        </Panel>
      </div>
    </>
  )
}

export function ReportsPage() {
  const id = useBusinessId()
  const business = useBusinessHistory(id)
  const forecast = useApi<ForecastResult>(businessPath(id, '/forecast'))
  const history = business.data?.history ?? []
  const forecastRows = forecast.data
    ? ['revenue', 'cash_balance'].flatMap((metric) =>
        forecast.data!.forecasts[metric as 'revenue' | 'cash_balance'].forecast.map((point) => ({
          month: point.month,
          metric,
          forecast_value: point[metric],
        })),
      )
    : []

  return (
    <>
      <PageHeader eyebrow="Exports" title="Reports" description="Download selected-business records and six-month projections for offline review." />
      <div className="grid two-col">
        <Panel title="Business financial history" description={`${history.length} monthly records available`}>
          <StateBlock loading={business.loading} error={business.error}>
            {history.length ? (
              <>
                <div className="grid three-col" style={{ marginBottom: 18 }}>
                  <MetricCard label="Business" value={business.data?.business.business_id ?? '—'} note={business.data?.business.sector} icon={BriefcaseBusiness} />
                  <MetricCard label="Latest revenue" value={formatMoney(business.data?.business.revenue)} note="Selected record" icon={CircleDollarSign} />
                  <MetricCard label="Latest cash" value={formatMoney(business.data?.business.cash_balance)} note="Selected record" icon={Wallet} />
                </div>
                <DownloadAction onClick={() => downloadCsv(`${id}_financial_history.csv`, history)} label="Download history CSV" />
              </>
            ) : <EmptyState label="No business records are available." />}
          </StateBlock>
        </Panel>
        <Panel title="Forecast report" description="Revenue and cash values for the next six months">
          <StateBlock loading={forecast.loading} error={forecast.error}>
            {forecast.data ? (
              <>
                <DataTable columns={[
                  { key: 'month', label: 'Month', render: (row) => new Date(String(row.month)).toLocaleDateString('en-IN', { month: 'short', year: 'numeric' }) },
                  { key: 'metric', label: 'Metric' },
                  { key: 'forecast_value', label: 'Projection', render: (row) => formatMoney(Number(row.forecast_value)) },
                ]} rows={forecastRows} />
                <div style={{ marginTop: 16 }}>
                  <DownloadAction onClick={() => downloadCsv(`${id}_forecast_report.csv`, forecastRows)} label="Download forecast CSV" />
                </div>
              </>
            ) : null}
          </StateBlock>
        </Panel>
      </div>
      <div className="notice" style={{ marginTop: 16 }}><Download size={13} style={{ verticalAlign: 'middle', marginRight: 5 }} /> CSV reports contain synthetic academic demonstration data and model estimates. Validate important decisions independently.</div>
    </>
  )
}
