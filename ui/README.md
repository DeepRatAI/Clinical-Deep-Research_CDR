# CDR UI - Evidence-First Clinical Research Interface

Frontend application for the Clinical Deep Research (CDR) system.

## Overview

This is a React/TypeScript frontend that provides an evidence-first interface for clinical research synthesis. It follows PRISMA-S guidelines for search reporting and enables traceable evidence navigation.

## Features

- **Dashboard**: View all research runs with status, DoD level, claims, and verification coverage
- **New Run**: Create new research runs with PICO-formatted questions
- **Run Detail**: Explore PRISMA flow, claims, snippets, and search strategy
- **Claim Detail**: Drill down into individual claims with supporting snippets and GRADE rationale
- **Evidence-first navigation**: Maximum 2 clicks from claim to source

## Tech Stack

- **React 18** with TypeScript
- **Vite** for fast development and builds
- **Tailwind CSS** for styling
- **TanStack Query** for data fetching and caching
- **React Router v6** for navigation
- **Lucide React** for icons

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm

### Installation

```bash
cd ui
npm install
```

### Development

```bash
npm run dev
```

The app will be available at http://localhost:3000

### Build

```bash
npm run build
```

### Type Check

```bash
npm run typecheck
```

### Lint

```bash
npm run lint
```

## Project Structure

```
ui/
├── src/
│   ├── api/           # API client and types
│   ├── components/    # Reusable components
│   │   ├── claims/    # Claim-related components
│   │   ├── common/    # Shared components (badges, layout)
│   │   ├── prisma/    # PRISMA flow visualization
│   │   └── snippets/  # Snippet display components
│   ├── pages/         # Main page components
│   ├── types/         # TypeScript type definitions
│   └── App.tsx        # Main app component
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## API Integration

The frontend communicates with the CDR backend API. In development, the Vite dev server proxies `/api` requests to `http://localhost:8000`.

### Required API Endpoints

- `GET /api/runs` - List all runs
- `GET /api/runs/{runId}` - Get run detail
- `POST /api/runs` - Create new run
- `GET /api/runs/{runId}/report` - Get report metadata

## Design Principles

1. **Evidence-First**: Evidence (snippets) are central to the experience
2. **Traceability**: Full path from claim → snippet → source record
3. **PRISMA Compliance**: Search strategy and flow always visible
4. **Negative Outcomes**: Explicit display when evidence is insufficient

## License

See main repository LICENSE file.
