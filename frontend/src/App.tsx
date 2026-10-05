import { Navigate, Route, Routes } from 'react-router-dom'
import { AppLayout } from './layouts/AppLayout'
import {
  AInsightsPage,
  AnomalyPage,
  BusinessAnalysisPage,
  CompliancePage,
  DashboardPage,
  ForecastingPage,
  ReportsPage,
  RiskPage,
} from './pages/Pages'

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/business-analysis" element={<BusinessAnalysisPage />} />
        <Route path="/risk-intelligence" element={<RiskPage />} />
        <Route path="/forecasting" element={<ForecastingPage />} />
        <Route path="/anomaly-detection" element={<AnomalyPage />} />
        <Route path="/ai-insights" element={<AInsightsPage />} />
        <Route path="/compliance-assistant" element={<CompliancePage />} />
        <Route path="/reports" element={<ReportsPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  )
}
