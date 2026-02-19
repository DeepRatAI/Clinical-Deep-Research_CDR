import { Routes, Route } from 'react-router-dom'
import Layout from './components/common/Layout'
import Dashboard from './pages/Dashboard'
import RunDetail from './pages/RunDetail'
import ClaimDetail from './pages/ClaimDetail'
import NewRun from './pages/NewRun'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="runs/new" element={<NewRun />} />
        <Route path="runs/:runId" element={<RunDetail />} />
        <Route path="runs/:runId/claims" element={<RunDetail />} />
        <Route path="runs/:runId/snippets" element={<RunDetail />} />
        <Route path="runs/:runId/hypotheses" element={<RunDetail />} />
        <Route path="runs/:runId/evaluation" element={<RunDetail />} />
        <Route path="runs/:runId/claims/:claimId" element={<ClaimDetail />} />
      </Route>
    </Routes>
  )
}

export default App
