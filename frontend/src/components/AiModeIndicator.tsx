import { useApi } from '../hooks/useApi'
import type { AIProvider, AIStatus } from '../types/api'

const providerLabels: Record<AIProvider, string> = {
  gemini: 'Online · Gemini',
  local: 'Offline · Local AI',
  retrieval_fallback: 'Retrieval fallback',
  local_summary: 'Local analytical summary',
}

const statusLabels: Record<AIProvider, string> = {
  ...providerLabels,
  gemini: 'Gemini configured',
}

export function AiModeIndicator({ provider }: { provider?: AIProvider }) {
  const status = useApi<AIStatus>('/api/ai/status')
  const selectedProvider = provider ?? status.data?.preferred_provider
  const label = status.error
    ? 'API unavailable'
    : selectedProvider
      ? (provider ? providerLabels : statusLabels)[selectedProvider]
      : 'Checking providers'
  const tone = selectedProvider === 'gemini' ? 'online' : 'offline'

  return (
    <span className="ai-mode" aria-label={`AI mode: ${label}`}>
      <span className={`dot ${tone}`} />
      <span>AI MODE</span>
      <strong>{label}</strong>
    </span>
  )
}
