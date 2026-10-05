export interface BusinessOption {
  business_id: string
  business_name: string
  sector: string
  state: string
  month: string
}

export interface BusinessRecord extends BusinessOption {
  employee_count: number
  revenue: number
  expenses: number
  cash_balance: number
  accounts_receivable: number
  accounts_payable: number
  inventory_value: number
  loan_emi: number
  loan_outstanding: number
  profit: number
  profit_margin: number
  revenue_growth: number | null
  expense_growth: number | null
  working_capital: number
  cash_ratio: number
  receivable_ratio: number
  payable_ratio: number
  debt_service_ratio: number
  inventory_ratio: number
}

export interface HistoryPoint {
  month: string
  [key: string]: string | number | null
}

export interface RiskResult {
  business_id: string
  label: 'LOW' | 'MEDIUM' | 'HIGH'
  probabilities: Record<string, number>
  explanation: string
}

export interface AnomalyResult {
  business_id: string
  month: string
  status: 'NORMAL' | 'ANOMALY'
  prediction: number
  score: number
  explanation: string
  indicators: Record<string, number>
}

export interface ShapFeature {
  feature: string
  value: number
  absolute_impact: number
}

export interface ForecastMetric {
  history: Array<{ month: string; [key: string]: string | number }>
  forecast: Array<{ month: string; [key: string]: string | number }>
  evaluation: {
    metric: string
    mae: number | null
    rmse: number | null
    holdout_months: number
  }
}

export interface ForecastResult {
  business_id: string
  forecasts: {
    revenue: ForecastMetric
    cash_balance: ForecastMetric
  }
}

export interface SummaryResult {
  answer: string
  used_gemini: boolean
  provider: AIProvider
}

export type AIProvider = 'gemini' | 'local' | 'retrieval_fallback' | 'local_summary'

export interface AIStatus {
  gemini_available: boolean
  gemini_configured: boolean
  ollama_installed: boolean
  ollama_running: boolean
  local_llm_available: boolean
  model: string
  retrieval_available: boolean
  offline_ready: boolean
  preferred_provider: AIProvider
}

export interface ComplianceResult {
  answer: string
  sources: string[]
  retrieved_documents: Array<{
    source: string
    chunk: number
    snippet: string
  }>
  used_gemini: boolean
  provider: AIProvider
}

export interface HealthResult {
  status: string
  data: string
  gemini_configured: boolean
}
