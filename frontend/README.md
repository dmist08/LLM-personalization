# Frontend — Cold-Start StyleVector UI

Interactive React application for comparing personalized headline generation methods side-by-side. Built with React 18, Vite, and TailwindCSS.

## Features

- **Side-by-side comparison** of 4 headline generation methods
- **42 Indian journalist** profiles from Times of India and Hindustan Times
- **Chat-style interface** with session history
- **Dark/Light theme** toggle
- **Real-time generation** with loading states and latency display

## Setup

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173
```

The dev server proxies `/api` requests to `http://localhost:5000` (Flask backend).

## Build for Production

```bash
npm run build
# Output in dist/
```

## Deploy to Vercel

1. Import the GitHub repo on [vercel.com](https://vercel.com)
2. Set **Root Directory** to `frontend`
3. Add environment variable:
   - `VITE_API_URL` = your backend URL (e.g., `https://your-backend.railway.app/api`)
4. Deploy — Vercel auto-detects Vite

The `vercel.json` file is pre-configured with:
- Build command: `npm run build`
- Output directory: `dist`
- SPA rewrite rules for React Router

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| React 18 | UI framework |
| Vite | Dev server and bundler |
| TailwindCSS | Utility-first CSS |
| React Router v6 | Client-side routing |
| Axios | HTTP client |
| Material Symbols | Icon set |

## Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | `HomePage` | Landing page with feature overview |
| `/chat` | `ChatPage` | Empty chat — centered input bar |
| `/chat/:sessionId` | `ChatPage` | Chat with generation results |

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── HeadlineCard.jsx    Result card for each generation method
│   │   ├── InputBar.jsx        Article input + publication/author selectors
│   │   └── Sidebar.jsx         Navigation sidebar with chat history
│   ├── context/
│   │   ├── AuthContext.jsx     Guest user authentication
│   │   └── ThemeContext.jsx    Dark/light theme state
│   ├── pages/
│   │   ├── ChatPage.jsx       Main chat interface
│   │   └── HomePage.jsx       Landing page
│   ├── services/
│   │   └── api.js             Axios API client
│   ├── App.jsx                Root component with routing
│   ├── index.css              Global styles + design tokens
│   └── main.jsx               Entry point
├── index.html
├── vite.config.js             Dev proxy + Vite config
├── tailwind.config.js         TailwindCSS customization
├── vercel.json                Vercel deployment config
└── package.json
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | Production only | Backend API base URL. Defaults to `/api` (proxied in dev) |
