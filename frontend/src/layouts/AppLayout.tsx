import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Bot,
  BriefcaseBusiness,
  FileChartColumn,
  LayoutDashboard,
  ShieldAlert,
  Sparkles,
} from 'lucide-react'
import { useApi } from '../hooks/useApi'
import type { BusinessOption, HealthResult } from '../types/api'

const navigation = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/business-analysis', label: 'Business Analysis', icon: BriefcaseBusiness },
  { to: '/risk-intelligence', label: 'Risk Intelligence', icon: ShieldAlert },
  { to: '/forecasting', label: 'Forecasting', icon: Activity },
  { to: '/anomaly-detection', label: 'Anomaly Detection', icon: AlertTriangle },
  { to: '/ai-insights', label: 'AI Insights', icon: Sparkles },
  { to: '/compliance-assistant', label: 'Compliance Assistant', icon: Bot },
  { to: '/reports', label: 'Reports', icon: FileChartColumn },
]

const titles: Record<string, string> = Object.fromEntries(
  navigation.map(({ to, label }) => [to, label]),
)

export interface BusinessOutletContext {
  businessId: string
  selectedBusiness: BusinessOption | undefined
}

export function AppLayout() {
  const { data: businesses } = useApi<BusinessOption[]>('/api/businesses')
  const { data: health } = useApi<HealthResult>('/api/health')
  const [selectedBusinessId, setSelectedBusinessId] = useState('')
  const location = useLocation()
  const selectedBusiness = businesses?.find(
    (business) => business.business_id === selectedBusinessId,
  ) ?? businesses?.[0]
  const businessId = selectedBusiness?.business_id ?? ''
  const pageTitle = titles[location.pathname] ?? 'Dashboard'

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><BarChart3 size={21} strokeWidth={2.4} /></div>
          <div className="brand-copy"><strong>UdyamPulse AI</strong><small>Financial intelligence</small></div>
        </div>
        <div className="nav-section">Workspace</div>
        <nav className="nav-list" aria-label="Main navigation">
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`} key={to} to={to}>
              <Icon size={16} />
              <span className="nav-label">{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note">Predictive financial analytics and compliance intelligence for MSMEs.</div>
          <div className="connection"><span className={`dot${health?.status === 'ok' ? '' : ' offline'}`} />{health?.status === 'ok' ? 'API connected' : 'Checking API status'}</div>
        </div>
      </aside>

      <div className="main-wrap">
        <header className="topbar">
          <div className="topbar-title">{pageTitle}</div>
          <div className="topbar-right">
            <span className="selector-label">Business</span>
            <select
              aria-label="Select business"
              className="business-select"
              value={businessId}
              onChange={(event) => setSelectedBusinessId(event.target.value)}
            >
              {(businesses ?? []).map((business) => (
                <option key={business.business_id} value={business.business_id}>
                  {business.business_name || business.business_id} · {business.business_id}
                </option>
              ))}
            </select>
            <div className="avatar" aria-label="User profile">UP</div>
          </div>
        </header>
        <main className="content">
          <Outlet context={{ businessId, selectedBusiness } satisfies BusinessOutletContext} />
          <div className="footer-note">UdyamPulse AI · Academic demonstration · Synthetic data only</div>
        </main>
      </div>
    </div>
  )
}
