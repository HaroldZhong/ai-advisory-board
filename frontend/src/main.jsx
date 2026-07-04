import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import 'katex/dist/katex.min.css'  // Load KaTeX styles globally
import 'highlight.js/styles/github-dark.css'  // Code block syntax theme (see index.css for the dark-panel wrapper)
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
