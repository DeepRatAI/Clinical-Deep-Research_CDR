import { Outlet, Link, useLocation } from 'react-router-dom'
import {
  LayoutDashboard,
  FileSearch,
  Settings,
  HelpCircle,
  PlusCircle,
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'New Run', href: '/runs/new', icon: PlusCircle },
]

const secondaryNavigation = [
  { name: 'Settings', href: '/settings', icon: Settings },
  { name: 'Help', href: '/help', icon: HelpCircle },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-white border-r border-gray-200">
        {/* Logo */}
        <div className="flex items-center gap-2 px-6 py-4 border-b border-gray-200">
          <FileSearch className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-lg font-bold text-gray-900">CDR</h1>
            <p className="text-xs text-gray-500">Clinical Deep Research</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex flex-col gap-1 p-4">
          <div className="mb-2">
            <p className="px-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Main
            </p>
          </div>
          {navigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`nav-item ${
                  isActive ? 'nav-item-active' : 'nav-item-inactive'
                }`}
              >
                <item.icon className="w-5 h-5" />
                {item.name}
              </Link>
            )
          })}

          <div className="mt-6 mb-2">
            <p className="px-3 text-xs font-semibold text-gray-400 uppercase tracking-wider">
              System
            </p>
          </div>
          {secondaryNavigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`nav-item ${
                  isActive ? 'nav-item-active' : 'nav-item-inactive'
                }`}
              >
                <item.icon className="w-5 h-5" />
                {item.name}
              </Link>
            )
          })}
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
          <div className="text-xs text-gray-500">
            <p>CDR v0.1.0</p>
            <p className="mt-1">Evidence-First Research</p>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="pl-64">
        <div className="min-h-screen">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
