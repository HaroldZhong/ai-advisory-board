# AI Advisory Board - Frontend

The frontend is a modern React application built with Vite, designed to facilitate complex multi-LLM interactions.

## Key Features

- **Real-time Streaming**: Server-Sent Events (SSE) for granular token-by-token updates from the Council.
- **Dynamic UI**: Components for Council deliberations, Chain of Thought steps, and cost estimation.
- **Responsive Design**: Tailwind CSS for a seamless experience on desktop and mobile.
- **Markdown & Math**: Full support for rendering rich text and LaTeX equations using `react-markdown` and `katex`.

## Project Structure

- `src/components`: UI components (ChatInterface, Sidebar, ModelSelector, etc.)
- `src/contexts`: React Context for global state (Settings, Theme).
- `src/landing`: Marketing landing page (deployed separately on `marketing` branch).
- `src/utils`: Helper functions for cost calculation, formatting, etc.

## Development

Run the development server:

```bash
npm run dev
```

## Build

Build for production:

```bash
npm run build
```
